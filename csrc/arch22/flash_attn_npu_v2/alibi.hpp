#ifndef COMMON_ALIBI_BIAS_HPP
#define COMMON_ALIBI_BIAS_HPP

namespace Alibi {
enum class AlibiMaskType : uint32_t {
    NO_MASK = 0,
    MASK_CAUSAL = 1,
    MASK_SWA = 4
};

__aicore__ inline void BuildAbsBiasRow(AscendC::LocalTensor<float> &workUb,
                                       int64_t baseColIdx, float slope, int32_t N)
{
    AscendC::CreateVecIndex<float>(workUb, static_cast<float>(-baseColIdx), N);  
    AscendC::PipeBarrier<PIPE_V>();
    if (baseColIdx >= 0) {
        int64_t absCnt = baseColIdx < static_cast<int64_t>(N) ? baseColIdx : static_cast<int64_t>(N);
        AscendC::Abs<float>(workUb, workUb, static_cast<int32_t>(absCnt));
        AscendC::PipeBarrier<PIPE_V>();
    }
    AscendC::Muls<float>(workUb, workUb, -slope, N);
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void AdvanceAbsBiasRow(AscendC::LocalTensor<float> &workUb,
                                         int64_t baseColIdx, float slope, int32_t N, float delta=1.0f)
{
    if (baseColIdx <= 0) {
        AscendC::Adds<float>(workUb, workUb, delta * slope, N);                          
    } else if (baseColIdx >= static_cast<int64_t>(N)) {
        AscendC::Adds<float>(workUb, workUb, delta * -slope, N);                         
    } else {
        constexpr int32_t FLOAT_BLOCK_SIZE = 8;
        const int32_t bci = static_cast<int32_t>(baseColIdx);
        const int32_t alignedLeft = bci / FLOAT_BLOCK_SIZE * FLOAT_BLOCK_SIZE;
        const int32_t x = bci - alignedLeft; 
        if (alignedLeft > 0) {
            AscendC::Adds<float>(workUb, workUb, -delta * slope, alignedLeft);                      
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::Adds<float>(workUb[alignedLeft], workUb[alignedLeft], delta * slope, N - alignedLeft);  
        if (x != 0) {
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds<float>(workUb[alignedLeft], workUb[alignedLeft], -2.0f * delta * slope, x);    
        }
    }
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void AddBiasToRow(AscendC::LocalTensor<float> &scoreUb, uint32_t rowOff,
                                    AscendC::LocalTensor<float> &workUb, int32_t N)
{
    AscendC::Add<float>(scoreUb[rowOff], scoreUb[rowOff], workUb, N); 
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void RescaleBiasRow(AscendC::LocalTensor<float> &workUb,
                                      float slope, float preSlope, int32_t N)
{
    AscendC::Muls<float>(workUb, workUb, slope / preSlope, N);  
    AscendC::PipeBarrier<PIPE_V>();
}

template <AlibiMaskType MASK_TYPE>
__aicore__ inline void ApplyAlibiRows(
    AscendC::LocalTensor<float> &scoreUb, uint32_t scoreOffset,
    uint32_t rowStride, uint32_t columnNumRound,
    uint32_t absRowStart, uint32_t rowNumCurLoop,
    uint32_t qSBlockSize, int64_t qPosBase,
    AscendC::GlobalTensor<float> &slopesGm, uint64_t slopesGmOffset,
    AscendC::LocalTensor<float> &workUb,
    int64_t kvSStartIdx);

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
    (void)qPosBase; 
    if (rowNumCurLoop == 0 || qSBlockSize == 0 || columnNumRound == 0) {
        return;
    }
    const int32_t count = static_cast<int32_t>(columnNumRound);

    AscendC::CreateVecIndex<float>(workUb, static_cast<float>(kvSStartIdx), count); 
    AscendC::PipeBarrier<PIPE_V>();
    float preSlope = 1.0f;

    for (uint32_t ri = 0; ri < rowNumCurLoop; ++ri) {
        const uint32_t absRow = absRowStart + ri;
        const uint32_t head   = absRow / qSBlockSize;          
        const float    slope  = slopesGm.GetValue(slopesGmOffset + head);  
        if (slope != preSlope) {
            RescaleBiasRow(workUb, slope, preSlope, count);
            preSlope = slope;
        }
        AddBiasToRow(scoreUb, scoreOffset + ri * rowStride, workUb, count);
    }
}

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

    uint32_t prevHead = 0xFFFFFFFFu; 
    int64_t  prevIq = -1;             
    float    preSlope = 0.0f;

    for (uint32_t ri = 0; ri < rowNumCurLoop; ++ri) {
        const uint32_t absRow = absRowStart + ri;
        const uint32_t head   = absRow / qSBlockSize;          
        const uint32_t token  = absRow % qSBlockSize;          
        const int64_t  i_q    = qPosBase + static_cast<int64_t>(token);
        const int64_t  baseColIdx = i_q - kvSStartIdx;
        const float    slope  = slopesGm.GetValue(slopesGmOffset + head); 

        if (prevHead == 0xFFFFFFFFu || head != prevHead) {
            if (i_q == prevIq) {
                RescaleBiasRow(workUb, slope, preSlope, count);
            } else {
                BuildAbsBiasRow(workUb, baseColIdx, slope, count);
            }
        } else {
            AdvanceAbsBiasRow(workUb, baseColIdx, slope, count, i_q - prevIq);
        }
        AddBiasToRow(scoreUb, scoreOffset + ri * rowStride, workUb, count);
        prevHead = head;
        prevIq = i_q;
        preSlope = slope;
    }
}

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

} 

#endif 
