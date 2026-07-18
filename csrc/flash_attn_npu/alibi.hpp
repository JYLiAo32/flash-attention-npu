/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 *
 * Shared ALiBi (Attention with Linear Biases) construction for FlashAttention-NPU v2.
 * Used by BOTH the forward online-softmax epilogue (online_softmax.hpp) and the backward
 * score-recompute epilogue (fag_epilogue_op.hpp). Keeping one construction guarantees
 * fwd / bwd apply the SAME bias (required: bias is input-independent, so backward only
 * needs to re-apply it when recomputing P = softmax(scale·S + bias); no extra gradient).
 *
 * Math (decisive reference: impl_alibi/refs/alibi.h, the TriDao CUDA implementation):
 *   causal      : bias[i,j,h] = slope_h * j           // j = ABSOLUTE key position (kvSStartIdx + col)
 *                 (causal guarantees i>=j, so -slope*|i-j| = -slope*i + slope*j;
 *                  -slope*i is a per-row constant -> softmax-translation-invariant -> dropped.
 *                  ONLY +slope*j remains: a column-only bias, shared across all query rows of a head.)
 *   non-causal  : bias[i,j,h] = -slope_h * |i - j|    // i = abs query pos (incl. bottom-right
 *                                                         diffS = seqlen_k - seqlen_q); j = abs key pos
 *
 * *** S-block layout is BNS: [B, N_q, S_q, S_k] (head-major). ***
 * The fused GEMM's M axis packs head on the OUTSIDE, sequence on the INSIDE, so for an absolute
 * tile row `absRow` (qSBlockSize = rows-per-head in this tile):
 *     head  = absRow / qSBlockSize     // slow/outer index, in [0, qNBlockSize)
 *     token = absRow % qSBlockSize     // fast/inner index, in [0, qSBlockSize)
 *     i_q   = qPosBase + token         // abs query position (qPosBase already includes diffS)
 * Consequence: sequence is contiguous -> consecutive rows WITHIN a head share the same slope and
 * advance token by 1. The cross-row reuse below exploits exactly this.
 * (qNBlockSize == 1 -- the overwhelmingly common case -- degenerates to a single head: head=0,
 *  token=absRow; BNS and the old BSN decode agree there, so the prior BSN bug was latent.)
 *
 * *** No cross-call state: UB is tight, the alibi scratch is reused for other ops of the current
 * KV-tile and is NOT preserved to the next KV-tile. So each ApplyAlibiRows call rebuilds its bias
 * from scratch; all cross-row state (preSlope / prevHead) is a LOCAL variable. ***
 *
 * Algorithms (full spec: impl_alibi/notes/alibi_method_bns.md):
 *   MASK_CAUSAL : rebuild bare ramp [kvSStartIdx, ...] once per call; local preSlope=1 sentinel;
 *                 per row: on head change Muls(workUb, slope/preSlope) in-place (ratio); Add(row).
 *   NO_MASK/SWA : per head-segment first row build -slope*|j-baseColIdx| from scratch; subsequent
 *                 rows in the same head advance it by the delta=1 split (left -slope / right +slope
 *                 about the new zero baseColIdx); Add(row). Head boundary => rebuild.
 *
 * Absolute column is mandatory for causal (proven by impl_alibi/verify/verify_tiled_causal.py):
 * the online softmax merges KV-tiles with different kvSStartIdx, so slope*kvSStartIdx (a per-tile
 * constant) CANNOT be dropped. We rebuild the ramp [kvSStartIdx, kvSStartIdx+1, ...] every call.
 *
 * Vectorisation: AscendC high-level "count" API (Adds/Muls/Add/Abs/CreateVecIndex with an element
 * count and isSetMask=true). One call processes a whole row of `count` elements; isSetMask handles
 * the sub-vector tail, so the op touches exactly [0, count). NO inner chunk loop is needed.
 *
 * Compile-time gate: ALiBi is compiled in/out by the HAS_ALIBI template parameter, threaded from
 * flash_api.cpp (whether slopes were passed) through the dispatch policy (EpilogueAtlasA2OnlineSoftmaxT
 * for fwd, EpilogueAtlasA2FAGOp / EpilogueAtlasA2SameAbVec for bwd) down to the epilogue. Each call site
 * wraps ApplyAlibiRows in `if constexpr (HAS_ALIBI)`, so when no slopes are provided the entire ALiBi
 * body (and its UB scratch) is compiled away. The slopes GM pointer itself travels as an independent
 * kernel launch arg (like q/k/v); only alibiSlopesBatchStride is routed through tiling.
 *
 * Items tagged [需 NPU 验证] depend on AscendC intrinsic signatures that cannot be confirmed
 * without the Ascend toolchain (this environment has none). The Adds/Muls/Add/Abs count-form is
 * confirmed by the user; CreateVecIndex is the official ramp API (user-confirmed real).
 */
#ifndef COMMON_ALIBI_BIAS_HPP
#define COMMON_ALIBI_BIAS_HPP

// NOTE: this header assumes the includer has already brought in the AscendC runtime
// (kernel_operator.h / catlass.hpp). Both fwd (online_softmax.hpp via catlass) and bwd
// (fag_epilogue_op.hpp via common_header.h) satisfy this. We use fully-qualified AscendC::
// names and add no includes of our own to avoid include-path conflicts between the trees.

namespace Alibi {

// Mask taxonomy relevant to ALiBi bias construction. Defined here, in the shared fwd+bwd
// header, so neither tree has to reach into the other's mask enum: the fwd kernel uses
// KernelCommon::FaiKenel::MaskType, the bwd/varlen path uses FAGTiling::MaskType, but ALiBi
// only cares about the causal-vs-not distinction (plus keeping the SWA case distinct). Each
// caller maps its own mask type to one of these values:
//   NO_MASK      -> non-causal bias  -slope_h * | i_q - (kvSStartIdx + col) |  (always correct)
//   MASK_CAUSAL  -> causal bias       slope_h * (kvSStartIdx + col)            (column-only; valid i>=j)
//   MASK_SWA     -> sliding-window    full |i-j| (window-direction-agnostic; safe for any SWA)
enum class AlibiMaskType : uint32_t {
    NO_MASK = 0,
    MASK_CAUSAL = 1,
    MASK_SWA = 4
};

// Apply ALiBi bias IN PLACE to a contiguous block of `rowNumCurLoop` score rows held in UB.
//
//   scoreUb[scoreOffset + ri*rowStride + c] += bias( absRowStart + ri , c )   for c in [0, columnNumRound)
//
// The score tile is row-major with rowStride = columnNumRound floats per row (rows packed in UB,
// as written by CopySGmToUb). Each row is handled by count-based vector ops over columnNumRound
// elements (isSetMask handles the tail); there is no inner column loop.
//
// Implementation: FULL TEMPLATE SPECIALISATION per AlibiMaskType. The primary template below has NO
// generic body (just a static_assert) -- only the explicit specialisations are callable, so each mask
// type owns a fully independent implementation that can diverge freely. Only one MASK_TYPE is ever
// instantiated per kernel, so there is zero runtime overhead.
//
// Specialisations provided:
//   MASK_CAUSAL : column-only bias slope_h * (kvSStartIdx + col)   -- single workUb, in-place ratio
//   NO_MASK     : full per-row bias -slope_h * | i_q - (kvSStartIdx + col) |
//   MASK_SWA    : PROVISIONAL -- currently identical to NO_MASK (non-causal |i-j|, safe for any window
//                 direction). Replace with SWA-specific handling once its optimal form is decided.
//
// Args (identical for every specialisation):
//   scoreUb / scoreOffset : the fp32 score tile; this chunk's row 0 is at scoreUb[scoreOffset].
//   rowStride             : floats from one row to the next (= columnNumRound).
//   columnNumRound        : padded columns per row (only [0, columnNum) are valid; padding is ignored
//                           by downstream row-max/row-sum which reduce over columnNum, so adding finite
//                           bias into the padded tail is harmless).
//   absRowStart           : ABSOLUTE tile-row index of this chunk's first row
//                           (= rowOffsetThisSubBlock + rowOffsetCurLoop in fwd; respects the 2-AIV split).
//   rowNumCurLoop         : number of rows in this chunk.
//   qSBlockSize / qPosBase: BNS decode params (see file header). head = absRow/qSBlockSize,
//                           token = absRow%qSBlockSize, i_q = qPosBase + token.
//   slopesUb              : slopesUb[h] = slope of head (qNStartIdx + h), h in [0, qNBlockSize).
//                           Loaded by the caller; head decoded above indexes it directly.
//   workUb                : scratch, >= columnNumRound floats (one row). Holds the in-progress bias row,
//                           rebuilt from scratch every call (UB is not preserved across KV-tiles).
//   kvSStartIdx           : absolute key position of column 0 of this KV tile.

// === Helpers (BNS, per-call, no cross-call state) ===

// Build the non-causal bias row in place: workUb[j] = -slope * |j - baseColIdx| for j in [0, N).
// Method-doc §5.1: CreateVecIndex ramp (j - baseColIdx) -> Abs the negative prefix -> Muls by -slope.
__aicore__ inline void BuildAbsBiasRow(AscendC::LocalTensor<float> &workUb,
                                       int64_t baseColIdx, float slope, int32_t N)
{
    AscendC::CreateVecIndex<float>(workUb, static_cast<float>(-baseColIdx), N);  // [需 NPU 验证] workUb[j] = j - baseColIdx
    AscendC::PipeBarrier<PIPE_V>();
    if (baseColIdx >= 0) {
        // Only the prefix j < baseColIdx is negative; flip exactly that (capped at N).
        int64_t absCnt = baseColIdx < static_cast<int64_t>(N) ? baseColIdx : static_cast<int64_t>(N);
        AscendC::Abs<float>(workUb, workUb, static_cast<int32_t>(absCnt));  // [需 NPU 验证] count-form, partial prefix
        AscendC::PipeBarrier<PIPE_V>();
    }
    AscendC::Muls<float>(workUb, workUb, -slope, N);  // [需 NPU 验证] count-form, in-place
    AscendC::PipeBarrier<PIPE_V>();
}

// Advance workUb from bias(zero = baseColIdx-1) to bias(zero = baseColIdx) for the non-causal |i-j|
// form, assuming delta = 1 (guaranteed within a head: token advances by 1 per row -> baseColIdx += 1).
// Method-doc §5.2, unified for delta=1: the per-column delta is
//     Δ[j] = -slope for j < baseColIdx (new zero) ;  +slope for j >= baseColIdx
// so:
//   baseColIdx <= 0 : whole row += +slope      (zero at/left of block; left region empty)
//   baseColIdx >= N : whole row += -slope      (zero at/right of block; right region empty)
//   else            : [0, baseColIdx) += -slope ; [baseColIdx, N) += +slope
__aicore__ inline void AdvanceAbsBiasRow(AscendC::LocalTensor<float> &workUb,
                                         int64_t baseColIdx, float slope, int32_t N, float delta=1.0f)
{
    if (baseColIdx <= 0) {
        AscendC::Adds<float>(workUb, workUb, delta * slope, N);                          // [需 NPU 验证] count-form
    } else if (baseColIdx >= static_cast<int64_t>(N)) {
        AscendC::Adds<float>(workUb, workUb, delta * -slope, N);                         // [需 NPU 验证] count-form
    } else {
        // UB access must be 32B-aligned => the float index must be a multiple of FLOAT_BLOCK_SIZE (=8).
        // The split point baseColIdx is generally NOT aligned, so split at the last aligned boundary
        // <= baseColIdx: alignedLeft = floor(baseColIdx/8)*8, x = baseColIdx % 8. Then:
        //   [0, alignedLeft)          : fully LEFT  -> += -delta*slope
        //   [alignedLeft, N)          : first treat as RIGHT -> += +delta*slope
        //   [alignedLeft, baseColIdx) : the x LEFT cols were over-added by +delta*slope
        //                              -> correct with += -2*delta*slope (net -delta*slope)
        constexpr int32_t FLOAT_BLOCK_SIZE = 8;
        const int32_t bci = static_cast<int32_t>(baseColIdx);
        const int32_t alignedLeft = bci / FLOAT_BLOCK_SIZE * FLOAT_BLOCK_SIZE;
        const int32_t x = bci - alignedLeft;   // = baseColIdx % 8
        if (alignedLeft > 0) {
            AscendC::Adds<float>(workUb, workUb, -delta * slope, alignedLeft);                      // [需 NPU 验证] count-form [0, alignedLeft)
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::Adds<float>(workUb[alignedLeft], workUb[alignedLeft], delta * slope, N - alignedLeft);  // [需 NPU 验证] [alignedLeft, N)
        if (x != 0) {
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds<float>(workUb[alignedLeft], workUb[alignedLeft], -2.0f * delta * slope, x);    // [需 NPU 验证] correct x LEFT cols [alignedLeft, baseColIdx)
        }
    }
    AscendC::PipeBarrier<PIPE_V>();
}

// scoreUb[rowOff] += workUb (one row of N elements).
__aicore__ inline void AddBiasToRow(AscendC::LocalTensor<float> &scoreUb, uint32_t rowOff,
                                    AscendC::LocalTensor<float> &workUb, int32_t N)
{
    AscendC::Add<float>(scoreUb[rowOff], scoreUb[rowOff], workUb, N);  // [需 NPU 验证] count-form
    AscendC::PipeBarrier<PIPE_V>();
}

// Rescale the bias row in place by the slope ratio. Used when the head changes but the query
// position i_q is unchanged (decode: qSBlockSize==1): the bias shape is identical, only the slope
// differs, so workUb (holding -preSlope*shape) becomes -slope*shape via Muls(slope / preSlope).
__aicore__ inline void RescaleBiasRow(AscendC::LocalTensor<float> &workUb,
                                      float slope, float preSlope, int32_t N)
{
    AscendC::Muls<float>(workUb, workUb, slope / preSlope, N);  // [需 NPU 验证] in-place ratio
    AscendC::PipeBarrier<PIPE_V>();
}

// Primary template: no-op for unsupported AlibiMaskType. Supported types (NO_MASK, MASK_CAUSAL,
// MASK_SWA) have explicit specialisations below. Unsupported mask types (e.g. future additions)
// land here and silently do nothing at runtime.
template <AlibiMaskType MASK_TYPE>
__aicore__ inline void ApplyAlibiRows(
    AscendC::LocalTensor<float> &scoreUb, uint32_t scoreOffset,
    uint32_t rowStride, uint32_t columnNumRound,
    uint32_t absRowStart, uint32_t rowNumCurLoop,
    uint32_t qSBlockSize, int64_t qPosBase,
    AscendC::GlobalTensor<float> &slopesGm, uint64_t slopesGmOffset,
    AscendC::LocalTensor<float> &workUb,
    int64_t kvSStartIdx);

// ---- MASK_CAUSAL: column-only bias slope*(kvSStartIdx + col); single workUb, in-place ratio.
//      (causal needs no i_q: the per-row -slope*i term is softmax-invariant and dropped.)
//      UB-tight: rebuild the bare ramp every call (not preserved across KV-tiles); no baseCol cache. ----
template <>
__aicore__ inline void ApplyAlibiRows<AlibiMaskType::MASK_CAUSAL>(
    AscendC::LocalTensor<float> &scoreUb, uint32_t scoreOffset,
    uint32_t rowStride, uint32_t columnNumRound,
    uint32_t absRowStart, uint32_t rowNumCurLoop,
    uint32_t qSBlockSize, int64_t qPosBase,
    AscendC::GlobalTensor<float> &slopesGm, uint64_t slopesGmOffset,
    AscendC::LocalTensor<float> &workUb,
    int64_t kvSStartIdx)
{
    (void)qPosBase;  // unused: causal drops the per-row -slope*i term (no query position needed)
    if (rowNumCurLoop == 0 || qSBlockSize == 0 || columnNumRound == 0) {
        return;
    }
    const int32_t count = static_cast<int32_t>(columnNumRound);

    // Rebuild the bare column ramp [kvSStartIdx, kvSStartIdx+1, ...] every call.
    // preSlope = 1 sentinel: workUb is the bare ramp now, so Muls(workUb, slope/1) => ramp*slope;
    // and slope==1 => bare ramp already equals the bias, so the (slope != preSlope) check skips the
    // Muls and Add(ramp) is correct as-is.
    AscendC::CreateVecIndex<float>(workUb, static_cast<float>(kvSStartIdx), count);  // [需 NPU 验证] ramp
    AscendC::PipeBarrier<PIPE_V>();
    float preSlope = 1.0f;

    for (uint32_t ri = 0; ri < rowNumCurLoop; ++ri) {
        const uint32_t absRow = absRowStart + ri;
        const uint32_t head   = absRow / qSBlockSize;          // BNS: head = slow/outer index
        const float    slope  = slopesGm.GetValue(slopesGmOffset + head);  // [需 NPU 验证] GM->scalar read on AIV
        if (slope != preSlope) {
            RescaleBiasRow(workUb, slope, preSlope, count);
            preSlope = slope;
        }
        AddBiasToRow(scoreUb, scoreOffset + ri * rowStride, workUb, count);
    }
}

// ---- NO_MASK: full non-causal bias -slope*|i_q - (kvSStartIdx + col)| (always correct).
//      Per head-segment: first row builds from scratch (BuildAbsBiasRow); subsequent rows in the same
//      head advance by delta=1 (AdvanceAbsBiasRow); head boundary rebuilds. No cross-call state. ----
template <>
__aicore__ inline void ApplyAlibiRows<AlibiMaskType::NO_MASK>(
    AscendC::LocalTensor<float> &scoreUb, uint32_t scoreOffset,
    uint32_t rowStride, uint32_t columnNumRound,
    uint32_t absRowStart, uint32_t rowNumCurLoop,
    uint32_t qSBlockSize, int64_t qPosBase,
    AscendC::GlobalTensor<float> &slopesGm, uint64_t slopesGmOffset,
    AscendC::LocalTensor<float> &workUb,
    int64_t kvSStartIdx)
{
    if (rowNumCurLoop == 0 || qSBlockSize == 0 || columnNumRound == 0) {
        return;
    }
    const int32_t count = static_cast<int32_t>(columnNumRound);

    // bias(h,t) = -slope_h * shape(i_q), and shape depends ONLY on i_q (= qPosBase + token), NOT on head.
    // So when the head changes but i_q is unchanged (decode: qSBlockSize==1 => token always 0 => i_q
    // constant across all rows), the shape is identical and we only rescale by slope/preSlope -- cheaper
    // than rebuilding. When i_q ALSO changes (prefill multi-head: token wraps at the head boundary),
    // shape changes too and we must rebuild. State is local (UB is not preserved across KV-tiles).
    uint32_t prevHead = 0xFFFFFFFFu;  // sentinel => the first row always rebuilds
    int64_t  prevIq = -1;             // sentinel: i_q is always >= 0, so the first row never matches
    float    preSlope = 0.0f;

    for (uint32_t ri = 0; ri < rowNumCurLoop; ++ri) {
        const uint32_t absRow = absRowStart + ri;
        const uint32_t head   = absRow / qSBlockSize;          // BNS
        const uint32_t token  = absRow % qSBlockSize;          // BNS
        const int64_t  i_q    = qPosBase + static_cast<int64_t>(token);
        const int64_t  baseColIdx = i_q - kvSStartIdx;
        const float    slope  = slopesGm.GetValue(slopesGmOffset + head);  // [需 NPU 验证] GM->scalar read on AIV

        if (prevHead == 0xFFFFFFFFu || head != prevHead) {
            // Slope differs from the previous row (head changed), or this is the first row.
            if (i_q == prevIq) {
                // Head changed but query position unchanged (decode: qSBlockSize==1 => token
                // always 0; prevIq==-1 sentinel => first row never lands here) => bias shape
                // identical, only rescale by the slope ratio.
                RescaleBiasRow(workUb, slope, preSlope, count);
            } else {
                // First row, or head changed AND query position changed (prefill multi-head:
                // token wraps at the head boundary) => rebuild from scratch.
                BuildAbsBiasRow(workUb, baseColIdx, slope, count);
            }
        } else {
            // Same head as the previous row: query position advanced by delta = i_q - prevIq.
            AdvanceAbsBiasRow(workUb, baseColIdx, slope, count, i_q - prevIq);
        }
        AddBiasToRow(scoreUb, scoreOffset + ri * rowStride, workUb, count);
        prevHead = head;
        prevIq = i_q;
        preSlope = slope;
    }
}

// ---- MASK_SWA: PROVISIONAL -- non-causal |i-j|. Safe for any window direction (the |i-j| bias is
// window-direction-agnostic). SWA currently DELEGATES to the NO_MASK specialisation rather than
// duplicating the loop; replace this call with SWA-specific handling once its optimal form is decided.
template <>
__aicore__ inline void ApplyAlibiRows<AlibiMaskType::MASK_SWA>(
    AscendC::LocalTensor<float> &scoreUb, uint32_t scoreOffset,
    uint32_t rowStride, uint32_t columnNumRound,
    uint32_t absRowStart, uint32_t rowNumCurLoop,
    uint32_t qSBlockSize, int64_t qPosBase,
    AscendC::GlobalTensor<float> &slopesGm, uint64_t slopesGmOffset,
    AscendC::LocalTensor<float> &workUb,
    int64_t kvSStartIdx)
{
    // TODO: implement SWA-specific optimized handling.
    ApplyAlibiRows<AlibiMaskType::NO_MASK>(scoreUb, scoreOffset, rowStride, columnNumRound,
                                           absRowStart, rowNumCurLoop, qSBlockSize, qPosBase,
                                           slopesGm, slopesGmOffset, workUb, kvSStartIdx);
}

}  // namespace Alibi

#endif  // COMMON_ALIBI_BIAS_HPP
