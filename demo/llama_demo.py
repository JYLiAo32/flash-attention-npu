import torch
import torch.nn as nn
from utils import *

from types import SimpleNamespace

# from flash_attn import flash_attn_with_kvcache
from flash_attn_npu import flash_attn_with_kvcache

class LlamaAttentionLayer(nn.Module):
    """
    截取LlamaAttentionLayer的核心计算部分, 作为FlashAttention算子的调用示例。
    """
    def __init__(self, config):
        super().__init__()
        
        # 步骤1：初始化参数配置
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.causal = config.causal
        self.window_size = config.window_size
        self.num_splits = config.num_splits
        self.softcap = config.softcap if hasattr(config, 'softcap') else 0
        self.k_cache = None
        self.v_cache = None
        
        # 步骤2：实例化计算所需 Matmul 算子
        self.q_proj = nn.Linear(self.hidden_size, 
                                self.num_attention_heads * self.head_dim, 
                                bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, 
                                self.num_key_value_heads * self.head_dim, 
                                bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, 
                                self.num_key_value_heads * self.head_dim, 
                                bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, 
                                self.hidden_size, 
                                bias=config.attention_bias)
    
    def forward(self, hidden_states, position_embeddings):
        # 步骤1：先通过 Matmul 算子计算 Q/K/V 张量
        B, Sq, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(B, Sq, self.num_attention_heads, 
                                            self.head_dim)
        k = self.k_proj(hidden_states).view(B, Sq, self.num_key_value_heads, 
                                            self.head_dim)
        v = self.v_proj(hidden_states).view(B, Sq, self.num_key_value_heads, 
                                            self.head_dim)
        
        # 步骤2：对 Q/K 应用位置编码
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # 步骤3：组织 KV Cache
        self.k_cache[:, self.cache_seqlens:self.cache_seqlens + Sq] = k
        self.v_cache[:, self.cache_seqlens:self.cache_seqlens + Sq] = v
        self.cache_seqlens += Sq
        cache_seqlens = torch.full((B,), self.cache_seqlens, dtype=torch.int32, 
                                   device=hidden_states.device)
        
        # 步骤4：调用 FA 算子计算注意力结果
        out = flash_attn_with_kvcache(q, self.k_cache, self.v_cache, None, None, 
                              cache_seqlens=cache_seqlens, causal=self.causal, 
                              window_size=self.window_size, softcap=self.softcap, 
                              num_splits=self.num_splits)
                
        # 步骤5：通过 Matmul 算子计算出最终结果
        out = out.reshape(B, Sq, self.num_attention_heads * self.head_dim)
        out = self.o_proj(out)
        return out


if __name__ == "__main__":
    # 步骤1：配置参数
    file_path = "golden_data.pt"
    # file_path = "golden_data_with_softcap.pt"
    config = SimpleNamespace(
        hidden_size=768,
        num_attention_heads=6,
        num_key_value_heads=6,
        head_dim=128,
        attention_bias=True,
        causal=True,
        # softcap=softcap_val,  # 取消注释，算子使能softcap功能
        window_size=(-1, -1),
        num_splits=1,
    )
    
    # 步骤2：构建网络层实例
    model = LlamaAttentionLayer(config).to(TEST_DEVICE).to(torch.float16)
    
    # 步骤3：加载输入数据、模型权重和在CPU上计算得到的参考真值结果
    llama_layer, input_data, out_ref, pos_emb = load_data_and_model(file_path, 
                                                                    TEST_DEVICE, 
                                                                    torch.float16, 
                                                                    model)
    
    # 步骤4：开始计算
    out_flash = llama_layer(input_data, pos_emb)

    # 步骤5：对比并打印指标
    compare_metrics(out_ref, out_flash)