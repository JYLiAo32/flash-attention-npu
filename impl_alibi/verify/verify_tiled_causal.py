#!/usr/bin/env python3
"""
Prove the §3 ABSOLUTE-vs-RELATIVE column decision for CAUSAL ALiBi under the
online softmax that merges multiple KV tiles.

Context
-------
In causal attention, alibi = -slope*|i-j| with i>=j always, so it expands to
-slope*i + slope*j. The -slope*i term is a per-ROW constant (identical across
ALL kv-tiles of that row) and softmax is translation-invariant, so it can be
dropped. What remains is slope*j  with j = ABSOLUTE key position.

design.md §3 builds the column bias as [0,1,...,columnNumRound-1] (RELATIVE,
within-tile column index). That is slope*(j - kvSStartIdx) = slope*j - slope*kvSStartIdx.
The dropped slope*kvSStartIdx is a per-TILE constant (kvSStartIdx changes every
kv-stack). The online softmax merges tiles with DIFFERENT per-tile constants,
which distorts the cross-tile normalization -> final P is WRONG whenever
kvSeqlen spans >1 kv-stack.

This script demonstrates it directly with numpy:
  ref        : full unchunked causal alibi attention (ground truth, abs j)
  chunked_abs: online softmax over kv-chunks, bias = slope*(kvStart+col)   [abs]
  chunked_rel: online softmax over kv-chunks, bias = slope*col             [rel, design.md §3]
Expect: chunked_abs == ref  and  chunked_rel != ref  for the multi-chunk case,
        and abs == rel for a single-chunk case (bug only manifests multi-tile).

Run:  python impl_alibi/verify/verify_tiled_causal.py   (numpy only, no Ascend/GPU)
"""
import numpy as np


def _causal_valid(sq, sk, offset):
    # valid[i, j] = not masked = (j <= i + offset)
    i = np.arange(sq)[:, None]
    j = np.arange(sk)[None, :]
    return j <= (i + offset)


def ref_causal_alibi(q, k, v, slopes, scale):
    # full unchunked causal alibi attention; bias = slope*j (abs), -slope*i dropped.
    b, h, sq, d = q.shape
    hk = k.shape[1]
    grp = h // hk
    kb = np.repeat(k, grp, axis=1)
    vb = np.repeat(v, grp, axis=1)
    s = np.asarray(slopes, dtype=np.float64)
    if s.ndim == 1:
        s = np.broadcast_to(s[None, :], (b, h)).copy()
    offset = k.shape[2] - sq
    out = np.zeros((b, h, sq, d), dtype=np.float64)
    for bi in range(b):
        scores = np.matmul(q[bi], kb[bi].transpose(0, 2, 1)) * scale  # [h,sq,sk]
        jabs = np.arange(k.shape[2])[None, :]                          # [1,sk]
        bias = s[bi][:, None, None] * jabs[None].astype(np.float64)    # [h,1,sk]
        scores = scores + bias
        valid = _causal_valid(sq, k.shape[2], offset)[None]            # [1,sq,sk]
        scores = np.where(valid, scores, -np.inf)
        mx = scores.max(-1, keepdims=True)
        e = np.exp(scores - mx)
        p = e / e.sum(-1, keepdims=True)
        out[bi] = np.matmul(p, vb[bi])
    return out


def chunked_causal_alibi(q, k, v, slopes, scale, chunk, abs_col):
    # online softmax over kv-chunks; causal; alibi bias abs (slope*j) or rel (slope*col).
    b, h, sq, d = q.shape
    sk = k.shape[2]; hk = k.shape[1]
    grp = h // hk
    kb = np.repeat(k, grp, axis=1)
    vb = np.repeat(v, grp, axis=1)
    s = np.asarray(slopes, dtype=np.float64)
    if s.ndim == 1:
        s = np.broadcast_to(s[None, :], (b, h)).copy()
    offset = sk - sq
    out = np.zeros((b, h, sq, d), dtype=np.float64)
    for bi in range(b):
        Qh = q[bi].astype(np.float64); Kh = kb[bi].astype(np.float64); Vh = vb[bi].astype(np.float64)
        m = np.full((h, sq), -np.inf)
        denom = np.zeros((h, sq))
        acc = np.zeros((h, sq, d))
        for kvStart in range(0, sk, chunk):
            kvEnd = min(kvStart + chunk, sk)
            col = np.arange(kvStart, kvEnd)               # absolute j for this chunk
            # scores[h,i,jj] = scale*(Q·K^T)
            sub = np.matmul(Qh, Kh[:, kvStart:kvEnd].transpose(0, 2, 1)) * scale  # [h,sq,clen]
            if abs_col:
                bias = s[bi][:, None, None] * col[None, None, :].astype(np.float64)   # slope*j
            else:
                bias = s[bi][:, None, None] * (col[None, None, :] - kvStart).astype(np.float64)  # slope*(j-kvStart)
            sub = sub + bias
            valid = (col[None, :] <= (np.arange(sq)[:, None] + offset))[None]  # [1,sq,clen]
            sub = np.where(valid, sub, -np.inf)
            cm = sub.max(-1)                                  # [h,sq]
            finite = np.isfinite(cm)
            e = np.where(finite[..., None], np.exp(sub - np.where(finite, cm, 0.0)[..., None]), 0.0)
            cl = e.sum(-1)                                    # [h,sq]
            cev = np.matmul(e, Vh[:, kvStart:kvEnd])          # [h,sq,d]
            m_new = np.maximum(m, np.where(finite, cm, -np.inf))
            so = np.exp(m - m_new); so[~np.isfinite(so)] = 0.0
            sc = np.exp(np.where(finite, cm, -np.inf) - np.where(np.isfinite(m_new), m_new, 0.0))
            acc = acc * so[..., None] + cev * sc[..., None]
            denom = denom * so + cl * sc
            m = m_new
        denom = np.where(denom == 0, 1.0, denom)
        out[bi] = acc / denom[..., None]
    return out


def _case(b, h, hk, sq, sk, d, chunk, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((b, h, sq, d)).astype(np.float32)
    k = rng.standard_normal((b, hk, sk, d)).astype(np.float32)
    v = rng.standard_normal((b, hk, sk, d)).astype(np.float32)
    slopes = (rng.standard_normal((b, h)) * 0.1 + 0.5).astype(np.float32)
    scale = 1.0 / np.sqrt(d)
    ref = ref_causal_alibi(q, k, v, slopes, scale)
    cabs = chunked_causal_alibi(q, k, v, slopes, scale, chunk, abs_col=True)
    crel = chunked_causal_alibi(q, k, v, slopes, scale, chunk, abs_col=False)
    denom = max(np.max(np.abs(ref)), 1e-6)
    e_abs = np.max(np.abs(cabs - ref)) / denom
    e_rel = np.max(np.abs(crel - ref)) / denom
    n_chunks = int(np.ceil(sk / chunk))
    return n_chunks, e_abs, e_rel


def main():
    print("CAUSAL ALiBi: chunked online-softmax, ABSOLUTE vs RELATIVE column bias")
    print("  (ref = full unchunked attention with slope*|abs j|)\n")
    # single chunk: abs == rel (bug dormant)
    nc, ea, er = _case(2, 4, 4, 16, 16, 16, chunk=16)
    print(f"[single-chunk ] kv chunks={nc}  abs_err={ea:.2e}  rel_err={er:.2e}"
          f"   -> expect BOTH small (abs==rel when 1 chunk)")
    ok_single = (ea < 1e-5) and (er < 1e-5)
    # multi chunk: abs correct, rel wrong
    nc, ea_m, er_m = _case(2, 4, 4, 32, 32, 16, chunk=16)
    print(f"[multi-chunk  ] kv chunks={nc}  abs_err={ea_m:.2e}  rel_err={er_m:.2e}"
          f"   -> expect abs SMALL, rel LARGE")
    nc, ea_g, er_g = _case(1, 8, 2, 48, 64, 16, chunk=16)   # GQA + unequal seq + multi chunk
    print(f"[GQA/uneven   ] kv chunks={nc}  abs_err={ea_g:.2e}  rel_err={er_g:.2e}"
          f"   -> expect abs SMALL, rel LARGE")

    print("\nverdict:")
    ok = ok_single and (ea_m < 1e-5) and (er_m > 1e-2) and (ea_g < 1e-5) and (er_g > 1e-2)
    print("  * ABSOLUTE column bias (slope*(kvSStartIdx+col)) matches the reference "
          "across tiles  => CORRECT")
    print("  * RELATIVE column bias (slope*col, design.md §3 as written) diverges when "
          "kvSeqlen > MAX_KV_STACK_LEN  => must use ABSOLUTE")
    print(f"\nPASS={ok}  (multi-chunk: abs_err={ea_m:.2e} < 1e-5, rel_err={er_m:.2e} > 1e-2)")
    assert ok, "absolute/relative verdict failed"


if __name__ == "__main__":
    main()
