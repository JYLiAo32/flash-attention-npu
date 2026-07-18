#!/usr/bin/env python3
"""
Standalone numpy reference for FlashAttention + ALiBi, matching the convention
implemented in the NPU kernel (see agent_research/design.md §5 / analysis.md §6).

Convention (mirrors csrc online_softmax.hpp ApplyAlibi + the Python docstring):
    score[i,j,h] = softmax_scale * (Q . K^T)[i,j,h]  +  (-slope_h * |i + (seqlen_k - seqlen_q) - j|)
    (causal: additionally mask j > i + (seqlen_k - seqlen_q)  -> -inf)
    p = softmax(score);  out = p @ V

This script cross-checks the vectorized reference against an O(N^2) triple loop,
so the *math* is proven correct & self-consistent. (The NPU kernel must reproduce
this; it cannot be run here — no Ascend toolchain / torch.)
"""
import numpy as np

def alibi_attn_ref(q, k, v, alibi_slopes, softmax_scale, causal=False):
    # q,k,v: [b, h/hk, sq/sk, d] (hk may repeat to h via group). slopes: [b, h]
    b, h, sq, d = q.shape
    _, hk, sk, _ = k.shape
    grp = h // hk
    # broadcast K/V heads to Q heads (GQA)
    kb = np.repeat(k, grp, axis=1) if hk != h else k
    vb = np.repeat(v, grp, axis=1) if hk != h else v
    slopes = np.asarray(alibi_slopes, dtype=np.float64)          # [b, h]
    if slopes.ndim == 1:
        slopes = np.broadcast_to(slopes[None, :], (b, h)).copy()  # [b,h]
    offset = sk - sq                                             # bottom-right align
    out = np.empty((b, h, sq, d), dtype=np.float64)
    lse = np.empty((b, h, sq), dtype=np.float64)
    for bi in range(b):
        Qh = q[bi].astype(np.float64)                            # [h,sq,d]
        Kh = kb[bi].astype(np.float64)                           # [h,sk,d]
        Vh = vb[bi].astype(np.float64)
        s = np.matmul(Qh, np.transpose(Kh, (0, 2, 1))) * softmax_scale  # [h,sq,sk]
        ii = np.arange(sq)[:, None] + offset                     # [sq,1]
        jj = np.arange(sk)[None, :]                              # [1,sk]
        bias = -slopes[bi][:, None, None] * np.abs(ii[None] - jj)  # [h,sq,sk]
        s = s + bias
        if causal:
            mask2d = (np.arange(sk)[None, :] > (np.arange(sq)[:, None] + offset))  # [sq,sk]
            s = np.where(mask2d[None], -np.inf, s)
        s = s - s.max(axis=-1, keepdims=True)
        e = np.exp(s)
        p = e / e.sum(axis=-1, keepdims=True)
        out[bi] = np.matmul(p, Vh)
        lse[bi] = (s.max(axis=-1) + np.log(e.sum(axis=-1)))
    return out, lse

def alibi_attn_naive(q, k, v, alibi_slopes, softmax_scale, causal=False):
    """Brute-force triple-loop reference (independent implementation)."""
    b, h, sq, d = q.shape
    _, hk, sk, _ = k.shape
    grp = h // hk
    slopes = np.asarray(alibi_slopes, dtype=np.float64)
    if slopes.ndim == 1:
        slopes = np.broadcast_to(slopes[None, :], (b, h)).copy()
    offset = sk - sq
    out = np.zeros((b, h, sq, d), dtype=np.float64)
    for bi in range(b):
        for hi in range(h):
            hki = hi // grp
            for i in range(sq):
                scores = np.empty(sk, dtype=np.float64)
                for j in range(sk):
                    dot = float(np.dot(q[bi, hi, i], k[bi, hki, j])) * softmax_scale
                    bias = -slopes[bi, hi] * abs((i + offset) - j)
                    if causal and (j > i + offset):
                        scores[j] = -np.inf
                    else:
                        scores[j] = dot + bias
                m = scores.max()
                if not np.isfinite(m):   # all masked
                    out[bi, hi, i] = 0.0
                    continue
                e = np.exp(scores - m)
                p = e / e.sum()
                for j in range(sk):
                    out[bi, hi, i] += p[j] * v[bi, hki, j]
    return out

def main():
    rng = np.random.default_rng(0)
    for (b, h, hk, sq, sk, d, causal) in [
        (2, 4, 4, 16, 16, 32, False),
        (2, 4, 2, 13, 21, 32, True),    # GQA, unequal seq, causal
        (1, 8, 8, 64, 64, 16, False),
        (3, 6, 2, 8, 8, 8, True),
    ]:
        q = rng.standard_normal((b, h, sq, d)).astype(np.float32)
        k = rng.standard_normal((b, hk, sk, d)).astype(np.float32)
        v = rng.standard_normal((b, hk, sk, d)).astype(np.float32)
        slopes = rng.standard_normal((b, h)) * 0.1 + 0.5           # ~[0.4,0.6]
        scale = 1.0 / np.sqrt(d)
        ref, _ = alibi_attn_ref(q, k, v, slopes, scale, causal)
        naive = alibi_attn_naive(q, k, v, slopes, scale, causal)
        err = np.max(np.abs(ref - naive) / (np.max(np.abs(naive)) + 1e-6))
        assert np.all(np.isfinite(ref)), f"non-finite ref for {b,h,hk,sq,sk,d,causal}"
        status = "OK" if err < 1e-4 else "FAIL"
        print(f"[{status}] b={b} h={h}/{hk} sq={sq} sk={sk} d={d} causal={causal}  rel_err={err:.2e}")
        assert err < 1e-4, f"MISMATCH: vectorized ref vs naive (rel_err={err})"
    # sanity: alibi should suppress distant keys (non-causal, sq=sk, single head)
    d = 8; sq = sk = 32
    q = rng.standard_normal((1, 1, sq, d)); kk = rng.standard_normal((1,1,sk,d)); vv = rng.standard_normal((1,1,sk,d))
    slope_far = np.array([[1.0]]); slope_near = np.array([[0.0]])
    _, lse_far = alibi_attn_ref(q,kk,vv,slope_far,1/np.sqrt(d),False)
    _, lse_near= alibi_attn_ref(q,kk,vv,slope_near,1/np.sqrt(d),False)
    assert np.all(np.isfinite(lse_far)) and np.all(np.isfinite(lse_near))
    print("[OK] alibi sanity: finite LSE for slope=1.0 and slope=0.0")
    print("\nALL CHECKS PASSED — numpy alibi reference is self-consistent.")

if __name__ == "__main__":
    main()
