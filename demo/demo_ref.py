import json, math, torch, torch.nn as nn, torch.nn.functional as F
from types import SimpleNamespace

TEST_DEVICE = "cpu"

softcap_val=1.0

class LlamaAttentionLayer(nn.Module):
    """
    截取LlamaAttentionLayer的核心计算部分, 作为参考实现。
    """
    def __init__(self, config, layer_idx: int):
        super().__init__()
        # 1. 初始化模型参数
        self.config, self.layer_idx = config, layer_idx
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.scaling = self.head_dim ** -0.5
        self.causal = config.causal
        self.softcap = config.softcap

        # 2.实例化 Matmul 算子
        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        # 3.初始化 KV Cache
        max_cache_len = config.seqlen_k + config.seqlen_q
        self.k_cache = torch.zeros(config.batch_size, max_cache_len, config.num_key_value_heads, config.head_dim,
                                    device=TEST_DEVICE, dtype=torch.float16)
        self.v_cache = torch.zeros_like(self.k_cache)

        init_k = torch.randn(config.batch_size, config.seqlen_k, config.num_key_value_heads, config.head_dim,
                              device=TEST_DEVICE, dtype=torch.float16)
        init_v = torch.randn_like(init_k)
        self.k_cache[:, :config.seqlen_k].copy_(init_k)
        self.v_cache[:, :config.seqlen_k].copy_(init_v)
        self.cache_seqlens = config.seqlen_k

    def forward(self, hidden_states, position_embeddings):
        B, Sq, _ = hidden_states.shape
        Sk = self.cache_seqlens

        # 1. 计算 Q/K/V
        q = self.q_proj(hidden_states).view(B, Sq, self.num_attention_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(B, Sq, self.num_key_value_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B, Sq, self.num_key_value_heads, self.head_dim)

        # 2. 应用 RoPE 位置编码
        cos, sin = position_embeddings
        q, k = self.apply_rotary_pos_emb(q, k, cos, sin)

        # 3. 写入 KV Cache
        self.k_cache[:, Sk:Sk + Sq] = k
        self.v_cache[:, Sk:Sk + Sq] = v

        full_k = self.k_cache[:, :Sk + Sq]
        full_v = self.v_cache[:, :Sk + Sq]

        if self.num_key_value_heads != self.num_attention_heads:
            repeat = self.num_attention_heads // self.num_key_value_heads
            full_k = full_k.repeat_interleave(repeat, dim=2)
            full_v = full_v.repeat_interleave(repeat, dim=2)

        q = q.transpose(1, 2)          # [B, H, Sq, D]
        full_k = full_k.transpose(1, 2)
        full_v = full_v.transpose(1, 2)

        # 4. 使用 Attention 算子计算注意力输出
        out = self._eager_attn(q, full_k, full_v, Sk)
        self.cache_seqlens += Sq
        
        # 5. 通过 O_Proj 输出投影
        out = out.transpose(1, 2).contiguous()
        out = out.reshape(B, Sq, self.num_attention_heads * self.head_dim)
        out = self.o_proj(out)
        return out

    # 注意力算子的纯 PyTorch 实现, 用于生成参考输出
    def _eager_attn(self, q, k, v, Sk):
        qf, kf, vf = q.float(), k.float(), v.float()
        scores = torch.matmul(qf, kf.transpose(-1, -2))

        # 应用softcap
        print(self.softcap)
        if self.softcap > 0.0:
            # scores = self.softcap * torch.tanh(scores / self.softcap)
            scores = scores * (self.scaling / self.softcap)
            scores = scores * (-2.0)
            scores = torch.exp(scores)
            scores = scores + 1.0
            scores = torch.reciprocal(scores)
            scores = scores * (2.0 * self.softcap)
            scores = scores - self.softcap
        else:
            scores = scores * self.scaling

        if self.causal:
            B, H, Sq, Sk_total = scores.shape
            q_idx = torch.arange(Sq, device=q.device).view(1, 1, Sq, 1) + Sk
            k_idx = torch.arange(Sk_total, device=q.device).view(1, 1, 1, Sk_total)
            scores = scores.masked_fill(k_idx > q_idx, float("-inf"))

        attn = F.softmax(scores, dim=-1, dtype=torch.float32)
        return torch.matmul(attn, vf).to(q.dtype)

    # RoPE 位置编码实现
    def apply_rotary_pos_emb(self, q, k, cos, sin):
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed

    def rotate_half(self, x):
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)


def test_ref_implementation(seqlen_q, seqlen_k, d, causal, mha_type, dtype, softcap_val=0.0,
                            save_path="golden_data.pt"):
    device = TEST_DEVICE
    torch.manual_seed(0)

    batch_size, nheads = 2, 6
    nheads_k = nheads if mha_type == "mha" else (1 if mha_type == "mqa" else 3)

    config = SimpleNamespace(
        hidden_size=nheads * d,
        num_attention_heads=nheads,
        num_key_value_heads=nheads_k,
        head_dim=d,
        attention_bias=True,
        causal=causal,
        softcap=softcap_val,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        batch_size=batch_size
    )

    model = LlamaAttentionLayer(config, 0).to(device).to(dtype)

    hidden_states = torch.randn(batch_size, seqlen_q, config.hidden_size, device=device, dtype=dtype)

    # create position embeddings - shape [Sq, head_dim]
    cos = torch.randn(seqlen_q, d, device=device, dtype=dtype)
    sin = torch.randn(seqlen_q, d, device=device, dtype=dtype)
    position_embeddings = (cos, sin)

    init_k_cache, init_v_cache = model.k_cache.clone(), model.v_cache.clone()

    out_ref = model(hidden_states, position_embeddings)

    torch.save({
        "hidden_states": hidden_states.cpu(),
        "q_proj_weight": model.q_proj.weight.cpu(),
        "q_proj_bias": model.q_proj.bias.cpu(),
        "k_proj_weight": model.k_proj.weight.cpu(),
        "k_proj_bias": model.k_proj.bias.cpu(),
        "v_proj_weight": model.v_proj.weight.cpu(),
        "v_proj_bias": model.v_proj.bias.cpu(),
        "o_proj_weight": model.o_proj.weight.cpu(),
        "o_proj_bias": model.o_proj.bias.cpu(),
        "init_k_cache": init_k_cache.cpu(),
        "init_v_cache": init_v_cache.cpu(),
        "cache_seqlens": config.seqlen_k,
        "out_ref": out_ref.cpu(),
        "cos": cos.cpu(),
        "sin": sin.cpu()
    }, save_path)

    metrics = {
        "min": out_ref.min().item(),
        "max": out_ref.max().item(),
        "mean": out_ref.mean().item(),
        "std": out_ref.std().item(),
        "shape": list(out_ref.shape)
    }

    with open("metrics_ref.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[Ref] shape:{out_ref.shape} range:[{out_ref.min():.6e}, {out_ref.max():.6e}]")
    print(f"Saved: {save_path}, metrics_ref.json")


if __name__ == "__main__":
    test_ref_implementation(
        seqlen_q=16,
        seqlen_k=4096,
        d=128,
        causal=True,
        mha_type="mha",
        dtype=torch.float16,
        # save_path="golden_data_with_softcap.pt",
        # softcap_val=softcap_val,
    )