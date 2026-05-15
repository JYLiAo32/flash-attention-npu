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
        
        # 步骤1：初始化参数配置
        self.config = config
        self.start_time_event, self.end_time_event = self.init_time_event()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.causal = config.causal
        self.window_size = config.window_size
        self.num_splits = config.num_splits
        self.softcap = config.softcap
        self.k_cache = None
        self.v_cache = None
        
        # 步骤2：实例化 Matmul 算子
        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        
        # 步骤3：指定 FA 算子
        self.attn_layer = flash_attn_with_kvcache
    
    def forward(self, hidden_states, position_embeddings):
        # 步骤1：先通过 Matmul 算子计算 Q/K/V 张量
        B, Sq, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(B, Sq, self.num_attention_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(B, Sq, self.num_key_value_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B, Sq, self.num_key_value_heads, self.head_dim)
        
        # 步骤2：对 Q/K 应用位置编码
        cos, sin = position_embeddings
        q, k = self.apply_rotary_pos_emb(q, k, cos, sin)
        
        # 步骤3：组织 KV Cache
        self.k_cache[:, self.cache_seqlens:self.cache_seqlens + Sq] = k
        self.v_cache[:, self.cache_seqlens:self.cache_seqlens + Sq] = v
        self.cache_seqlens += Sq
        cache_seqlens = torch.full((B,), self.cache_seqlens, dtype=torch.int32, device=hidden_states.device)
        
        # 步骤4：调用 FA 算子计算注意力结果
        out = self.attn_layer(q, self.k_cache, self.v_cache, None, None, cache_seqlens=cache_seqlens, causal=self.causal, window_size=self.window_size, softcap=self.softcap, num_splits=self.num_splits)
                
        # 步骤5：通过 Matmul 算子计算出最终结果
        out = out.reshape(B, Sq, self.num_attention_heads * self.head_dim)
        out = self.o_proj(out)
        return out
    
    # 应用 RoPE 位置编码
    def apply_rotary_pos_emb(self, q, k, cos, sin):
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed

    # 辅助工具函数, 将输入张量的最后一个维度分成两半, 对后一半进行旋转并与前一半组合。
    def rotate_half(self, x):
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)
    
    # 辅助工具函数, 初始化时间事件对象。
    def init_time_event(self):
        start_event, end_event = None, None
        if TEST_DEVICE == "npu":
            start_event = torch_npu.npu.Event(enable_timing=True)
            end_event = torch_npu.npu.Event(enable_timing=True)
        elif TEST_DEVICE == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
        return start_event, end_event
    
    # 辅助工具函数, 在 Device 上开始记录时间
    def start_time_recording(self):
        if TEST_DEVICE == "npu":
            torch.npu.synchronize()
            self.start_time_event.record()
        elif TEST_DEVICE == "cuda":
            torch.cuda.synchronize()
            self.start_time_event.record()
        
    # 辅助工具函数, 在 Device 上结束时间记录。
    def end_time_recording(self):
        self.end_time_event.record()
        if TEST_DEVICE == "npu":
            torch.npu.synchronize()
        elif TEST_DEVICE == "cuda":
            torch.cuda.synchronize()

# 加载预先生成的输入数据和参考输出, 初始化网络并加载权重。
def load_data_and_model(device, dtype, model):
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

# 对比并打印性能指标, 输入为真值结果与 FA 输出结果。
def compare_metrics(out_ref, out_flash, model):
    ref_f, flash_f = out_ref.flatten().float(), out_flash.flatten().float()
    diff = (flash_f - ref_f).abs()
    max_d, mean_d = diff.max().item(), diff.mean().item()
    l1, l2 = torch.norm(flash_f - ref_f, p=1).item(), torch.norm(flash_f - ref_f, p=2).item()
    cos_sim = torch.nn.functional.cosine_similarity(ref_f.unsqueeze(0), flash_f.unsqueeze(0)).item()
    elapsed_time = model.start_time_event.elapsed_time(model.end_time_event)
    label_width = 22
    value_width = 10
    header_part1 = f"{'':<{label_width}}" 
    header_part2 = f"{'最大误差':>{6}}{'平均误差':>{6}}{'L1范数':>{6}}{'L2范数':>{8}}{'余弦相似度':>{9}}{'执行时间(秒)':>{9}}"
    separator = "=" * (label_width + value_width * 6 + 7)
    print(separator)
    print(header_part1 + header_part2)
    print(separator)
    line1_text = "Llama atten layer demo"
    line2_text = f"   ({PACKAGE_NAME})"
    values_str = f"{max_d:>{10}.6f}{mean_d:>{10}.6f}{l1:>{10}.6f}{l2:>{10}.6f}{cos_sim:>{10}.5f}{elapsed_time/1000:>{13}.6f}"
    print(f"{line1_text:<{label_width}}")
    print(f"{line2_text:<{label_width}}{values_str}")
    print(separator)


if __name__ == "__main__":
    # 步骤1：配置参数
    d, nheads, num_splits, softcap_val = 128, 6, 1, 0.0
    device, causal, dtype = TEST_DEVICE, True, torch.float16
    config = SimpleNamespace(hidden_size=nheads * d, num_attention_heads=nheads,
            num_key_value_heads=nheads, head_dim=d, attention_bias=True, causal=causal, 
            window_size=(-1, -1), num_splits=num_splits, softcap=softcap_val)
    
    # 步骤2：构建网络层实例
    model = LlamaAttentionLayer(config).to(device).to(dtype)
    
    # 步骤3：加载数据和模型权重
    llama_attention_layer, input_data, out_ref, pos_emb = load_data_and_model(TEST_DEVICE, dtype, model)
    
    # 步骤4：开始计算，并记录计算时间
    llama_attention_layer.start_time_recording()
    out_flash = llama_attention_layer(input_data, pos_emb)
    llama_attention_layer.end_time_recording()

    # 步骤5：对比并打印指标
    compare_metrics(out_ref, out_flash, llama_attention_layer)