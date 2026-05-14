import torch
import torch.nn as nn

from types import SimpleNamespace

# from flash_attn import flash_attn_with_kvcache
# TEST_DEVICE = "cuda"
# PACKAGE_NAME = "flash-attn"

import torch_npu
from flash_attn_npu import flash_attn_with_kvcache
TEST_DEVICE = "npu"
PACKAGE_NAME = "flash-attn-npu"

class LlamaAttentionLayer(nn.Module):
    """
    截取LlamaAttentionLayer的核心计算部分, 作为FlashAttention算子的调用示例。
    """
    
    def __init__(self, config):
        super().__init__()
        
        # 步骤1：保存 config
        self.config = config


        # 步骤2：基础模型结构参数
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_attention_heads


        # 步骤3：注意力层功能选项
        self.causal = config.causal
        self.window_size = config.window_size
        self.num_splits = config.num_splits
        self.softcap = config.softcap


        # 步骤4：初始化 KV cache
        self.k_cache = None
        self.v_cache = None
        
        
        # 步骤5：实例化 Matmul 算子
        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        
        # 步骤6：初始化 FlashAttention 算子
        self.attn_layer = flash_attn_with_kvcache

    
    def forward(self, hidden_states, position_embeddings):
        
        # 步骤1：先通过 Matmul 算子计算 Q/K/V 投影
        B, Sq, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(B, Sq, self.num_attention_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(B, Sq, self.num_key_value_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B, Sq, self.num_key_value_heads, self.head_dim)
        
        
        # 步骤2：对 Q/K 应用 RoPE 位置编码
        cos, sin = position_embeddings
        q, k = self.apply_rotary_pos_emb(q, k, cos, sin)
        
        
        # 步骤3：将新生成的 K/V 与历史 KV Cache 拼接，记录 KV Cache 的当前长度
        self.k_cache[:, self.cache_seqlens:self.cache_seqlens + Sq] = k
        self.v_cache[:, self.cache_seqlens:self.cache_seqlens + Sq] = v
        self.cache_seqlens += Sq
        cache_seqlens = torch.full((B,), self.cache_seqlens, dtype=torch.int32, device=hidden_states.device)
        
        # 步骤4：调用 FlashAttention 算子计算注意力结果
        out = self.attn_layer(q, self.k_cache, self.v_cache, None, None, cache_seqlens=cache_seqlens, causal=self.causal, window_size=self.window_size, softcap=self.softcap, num_splits=self.num_splits)
        
        
        # 步骤5：通过 Matmul 算子计算出最终结果
        out = out.reshape(B, Sq, self.num_attention_heads * self.head_dim)
        out = self.o_proj(out)
        return out

    
    def apply_rotary_pos_emb(self, q, k, cos, sin):
        """
        应用 RoPE 位置编码
        """
        
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed
    
    def rotate_half(self, x):
        """
        辅助工具函数
        将输入张量的最后一个维度分成两半, 对后一半进行旋转并与前一半组合。
        """
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)


def load_data_and_model(device, dtype, model):
    """
    加载预先生成的输入数据和参考输出, 初始化 LlamaAttentionLayer 模型并加载权重。
    """
    
    data = torch.load("ref_data.pt", map_location=device)
    hidden_states, out_ref = data["hidden_states"], data["out_ref"]
    init_k_cache, init_v_cache = data["init_k_cache"], data["init_v_cache"]
    cache_seqlens = data["cache_seqlens"]
    cos, sin = data["cos"], data["sin"]
    
    for proj, p in [(model.q_proj,"q"),(model.k_proj,"k"),(model.v_proj,"v"),(model.o_proj,"o")]:
        proj.weight.data.copy_(data[f"{p}_proj_weight"].to(device).to(dtype))
        proj.bias.data.copy_(data[f"{p}_proj_bias"].to(device).to(dtype))
    B, max_cache_len, Hk, D = init_k_cache.shape
    model.k_cache = torch.zeros(B, max_cache_len, Hk, D, device=device, dtype=dtype)
    model.v_cache = torch.zeros_like(model.k_cache)
    model.k_cache.copy_(init_k_cache.to(device).to(dtype))
    model.v_cache.copy_(init_v_cache.to(device).to(dtype))
    model.cache_seqlens = cache_seqlens
    position_embeddings = (cos.to(device).to(dtype), sin.to(device).to(dtype))
    return model, hidden_states.to(device).to(dtype), out_ref.to(device).to(dtype), position_embeddings


def print_metrics(out_ref, out_flash):
    """
    打印性能指标, 输入为参考输出结果与 FlashAttention 输出结果。
    """
    ref_f, flash_f = out_ref.flatten().float(), out_flash.flatten().float()
    diff = (flash_f - ref_f).abs()
    max_d, mean_d = diff.max().item(), diff.mean().item()
    l1, l2 = torch.norm(flash_f - ref_f, p=1).item(), torch.norm(flash_f - ref_f, p=2).item()
    cos_sim = torch.nn.functional.cosine_similarity(ref_f.unsqueeze(0), flash_f.unsqueeze(0)).item()

    label_width = 24
    value_width = 12
    
    header_part1 = f"{'':<{label_width}}" 
    header_part2 = f"{'最大误差':>{7}}{'平均误差':>{8}}{'L1范数':>{9}}{'L2范数':>{9}}{'余弦相似度':>{9}}"
    
    separator = "=" * (label_width + value_width * 5)
    
    print(separator)
    print(header_part1 + header_part2)
    print(separator)

    line1_text = "Llama atten layer demo"
    line2_text = f"   ({PACKAGE_NAME})"
    # values_str = f"{max_d:>{11}.3e}{mean_d:>{12}.3e}{l1:>{12}.3e}{l2:>{11}.3e}{cos_sim:>{11}.5f}"
    values_str = f"{max_d:>{11}.6f}{mean_d:>{12}.6f}{l1:>{12}.6f}{l2:>{11}.6f}{cos_sim:>{11}.5f}"

    print(f"{line1_text:<{label_width}}")
    print(f"{line2_text:<{label_width}}{values_str}")
    print(separator)


if __name__ == "__main__":
    
    # 第一步，初始化参数
    # 1.1. 基础模型超参数与设备选择
    d = 128
    nheads = 6
    mha_type = "mha"   # ["mha", "mqa", "gqa"]
    device = TEST_DEVICE


    # 1.2. attention 层的功能配置
    causal = True
    num_splits = 1
    softcap_val = 0.0
    dtype = torch.float16


    # 1.3. 根据注意力类型确定 K/V 的注意力头数
    if mha_type == "mha":
        nheads_k = nheads
    elif mha_type == "mqa":
        nheads_k = 1
    elif mha_type == "gqa":
        nheads_k = 3
    else:
        raise ValueError(f"Unknown mha_type: {mha_type}")


    # 1.4. 模型配置构造
    config = SimpleNamespace(
        hidden_size=nheads * d,
        num_attention_heads=nheads,
        num_key_value_heads=nheads_k,
        head_dim=d,
        attention_bias=True,
        causal=causal,
        window_size=(-1, -1),
        num_splits=num_splits,
        softcap=softcap_val,
    )
    
    
    # 第二步，用配置实例化 Llama 注意力层
    model = LlamaAttentionLayer(config).to(device).to(dtype)
    
    
    # 第三步，加载数据和模型权重
    llama_attention_layer, input_data, out_ref, pos_emb = load_data_and_model(TEST_DEVICE, dtype, model)
    
    
    # 第四步，调用 Llama 注意力层计算结果
    out_flash = llama_attention_layer(input_data, pos_emb)
    
    
    # 第五步，打印性能指标
    print_metrics(out_ref, out_flash)