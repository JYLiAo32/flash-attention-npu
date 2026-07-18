#!/usr/bin/env python
# Compare our ALiBi convention against the AUTHORITATIVE Tri Dao flash_attn 2.8.3
# (flash_attn_func with alibi_slopes), on CUDA. Resolves the slope-scaling /
# bottom-right-offset / |i-j| questions the NPU kernel must match.
import torch
from flash_attn import flash_attn_func

def ref_alibi(q, k, v, slopes, scale, causal):
    # q,k,v: [b, sq, h, d]; slopes: [h] or [b,h]
    b, sq, h, d = q.shape
    _, sk, hk, _ = k.shape
    grp = h // hk
    s = slopes
    if s.dim() == 1:
        s = s.unsqueeze(0).expand(b, h)
    kb = k.repeat_interleave(grp, dim=2) if hk != h else k
    vb = v.repeat_interleave(grp, dim=2) if hk != h else v
    # [b,h,sq,sk]
    qt = q.float().transpose(1, 2)                       # [b,h,sq,d]
    kt = kb.float().transpose(1, 2)                      # [b,h,sk,d]
    scores = torch.matmul(qt, kt.transpose(-1, -2)) * scale   # [b,h,sq,sk]
    offset = sk - sq
    i = torch.arange(sq, device=q.device).view(sq, 1) + offset
    j = torch.arange(sk, device=q.device).view(1, sk)
    bias = -s.float().view(b, h, 1, 1) * (i - j).abs().unsqueeze(0).unsqueeze(0)
    scores = scores + bias
    if causal:
        mask = torch.arange(sk, device=q.device).view(1, sk) > (torch.arange(sq, device=q.device).view(sq, 1) + offset)
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    scores = scores - scores.amax(dim=-1, keepdim=True)
    e = torch.exp(scores)
    p = e / e.sum(dim=-1, keepdim=True)
    out = torch.matmul(p, vb.float().transpose(1, 2)).transpose(1, 2)  # [b,sq,h,d]
    return out.to(q.dtype)

def main():
    torch.manual_seed(0)
    dev = "cuda"
    configs = [
        # (b, sq, sk, h, hk, d, causal)
        (2, 64, 64, 8, 8, 64, False),
        (2, 64, 64, 8, 8, 64, True),
        (2, 48, 64, 4, 4, 32, True),   # unequal seq (prefill-style bottom-right)
        (2, 48, 64, 4, 4, 32, False),
        (2, 64, 64, 8, 2, 64, True),   # GQA
    ]
    for (b, sq, sk, h, hk, d, causal) in configs:
        q = torch.randn(b, sq, h, d, device=dev, dtype=torch.float16)
        k = torch.randn(b, sk, hk, d, device=dev, dtype=torch.float16)
        v = torch.randn(b, sk, hk, d, device=dev, dtype=torch.float16)
        slopes = torch.rand(h, device=dev, dtype=torch.float32) * 0.3 + 0.1   # [0.1,0.4]
        scale = 1.0 / (d ** 0.5)
        # Tri Dao ground truth (flash_attn requires sq multiple of 128? no, it pads).
        try:
            fa_out = flash_attn_func(q, k, v, softmax_scale=scale,
                                     causal=causal, alibi_slopes=slopes)
        except Exception as e:
            print(f"[skip] cfg {b,sq,sk,h,hk,d,causal}: flash_attn raised {e}")
            continue
        ref = ref_alibi(q, k, v, slopes, scale, causal)
        # align shapes
        assert fa_out.shape == ref.shape, (fa_out.shape, ref.shape)
        denom = max(fa_out.abs().max().item(), 1.0)
        err = (fa_out.float() - ref.float()).abs().max().item() / denom
        status = "MATCH" if err < 1e-2 else "DIFF"
        print(f"[{status}] b={b} sq={sq} sk={sk} h={h}/{hk} d={d} causal={causal}  rel_err={err:.2e}")

if __name__ == "__main__":
    main()
