import torch

if torch.cuda.is_available():
    TEST_DEVICE = "cuda"
    PACKAGE_NAME = "flash-attn"
else:
    import torch_npu
    TEST_DEVICE = "npu"
    PACKAGE_NAME = "flash-attn-npu"

softcap_val = 1.0

# 应用 RoPE 位置编码
def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# 辅助工具函数, 将输入张量的最后一个维度分成两半, 对后一半进行旋转并与前一半组合。
def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

# 辅助工具函数, 初始化时间事件对象。
def init_time_event():
    start_event, end_event = None, None
    if TEST_DEVICE == "npu":
        start_event = torch_npu.npu.Event(enable_timing=True)
        end_event = torch_npu.npu.Event(enable_timing=True)
    elif TEST_DEVICE == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
    return start_event, end_event

# 辅助工具函数, 在 Device 上开始记录时间
def start_time_recording(start_time_event):
    if TEST_DEVICE == "npu":
        torch.npu.synchronize()
        start_time_event.record()
    elif TEST_DEVICE == "cuda":
        torch.cuda.synchronize()
        start_time_event.record()
    
# 辅助工具函数, 在 Device 上结束时间记录。
def end_time_recording(end_time_event):
    end_time_event.record()
    if TEST_DEVICE == "npu":
        torch.npu.synchronize()
    elif TEST_DEVICE == "cuda":
        torch.cuda.synchronize()

# 加载预先生成的输入数据和参考输出, 初始化网络并加载权重。
def load_data_and_model(file_path, device, dtype, model):
    data = torch.load(file_path, map_location=device)
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
def compare_metrics(out_ref, out_flash):
    ref_f, flash_f = out_ref.flatten().float(), out_flash.flatten().float()
    diff = (flash_f - ref_f).abs()
    max_d, mean_d = diff.max().item(), diff.mean().item()
    l1, l2 = torch.norm(flash_f - ref_f, p=1).item(), torch.norm(flash_f - ref_f, p=2).item()
    cos_sim = torch.nn.functional.cosine_similarity(ref_f.unsqueeze(0), flash_f.unsqueeze(0)).item()
    label_width = 22
    value_width = 10
    header_part1 = f"{'':<{label_width}}" 
    header_part2 = f"{'最大误差':>{6}}{'平均误差':>{6}}{'L1范数':>{6}}{'L2范数':>{8}}{'余弦相似度':>{9}}"
    separator = "=" * (label_width + value_width * 5 + 3)
    print(separator)
    print(header_part1 + header_part2)
    print(separator)
    line1_text = "Llama atten layer demo"
    line2_text = f"   ({PACKAGE_NAME})"
    values_str = f"{max_d:>{10}.6f}{mean_d:>{10}.6f}{l1:>{10}.6f}{l2:>{10}.6f}{cos_sim:>{10}.5f}"
    print(f"{line1_text:<{label_width}}")
    print(f"{line2_text:<{label_width}}{values_str}")
    print(separator)