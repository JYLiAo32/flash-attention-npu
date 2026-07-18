# impl_alibi —— FlashAttention-NPU Alibi 实现工作区

本目录是 Alibi（Attention with Linear Biases）功能的**设计与验证工作区**，自包含、可在新分支上独立推进实现。
当前分支（`agent_research`）下 `csrc/` 的 loop 改动会被丢弃，新 `alibi` 分支从 `main` 重建实现；**只迁移本目录**。

---

## 状态

| 项 | 状态 |
|---|---|
| 源码调用链 + 文件清单 | ✅（`notes/callchain_and_files.md`，源码验证） |
| Alibi 数学/布局理论 | ✅（`notes/alibi_math.md`，含 alibi.h 依据） |
| 设计评审（修正版 v2） | ✅（`design_review.md`） |
| 数学验证（numpy） | ✅（`verify/`，本机已跑通） |
| §3 绝对列修正证明 | ✅（`verify/verify_tiled_causal.py`，PASS） |
| **Kernel 实现** | ❌ 未做 —— 在新分支按 `design_review.md` + `callchain_and_files.md §5/§6` 从零实现 |
| NPU 编译/精度验证 | ❌ 无 Ascend 工具链 |

> 旧版（agent_research）的 `alibi_bias.hpp`（head-major 解码 + 全局 `|i-j|`）、`final_report.md`、`progress.md`、`findings.md`（含「v3 已有 alibi 脚手架/pseAlibiAddr」误判）**均作废，不迁移**。

---

## 目录

```
impl_alibi/
├── README.md                       # 本文件（导航 + 状态 + 实现顺序）
├── design.md                       # 用户原始设计（causal §3 / non-causal §7 / SWA §6）
├── design_review.md                # 修正评审 v2（撤回 v1 两处错误 + §3 绝对列修正）
├── refs/
│   └── alibi.h                     # TriDao CUDA alibi 参考（causal/non-causal 两分支，决定性依据）
├── notes/
│   ├── callchain_and_files.md      # 前向/反向调用链 + v2/v3 架构对比 + 改动文件地图 + 插入点
│   └── alibi_math.md               # BSN 布局 + 两套实现 + 绝对列 + FP32 向量子分块 + 增量更新
└── verify/
    ├── alibi_ref.py                # numpy 参考实现（向量化 vs 朴素，自洽）
    ├── verify_tiled_causal.py      # 证明 §3 绝对列（分块 online softmax，abs vs rel）
    └── vs_tridao.py                # 与官方 flash_attn 对拍（需 CUDA GPU，可选）
```

---

## 阅读顺序（实现前）
1. **`design_review.md` §0-§3** —— 结论先行 + 两套实现 + 绝对列修正。
2. **`notes/alibi_math.md`** —— 布局与数学的全部细节。
3. **`notes/callchain_and_files.md`** —— 改哪些文件、插入点在哪。
4. **`design.md`** —— 用户原始设计（causal §3 / non-causal §7 的算法描述）。
5. 跑 `verify/*.py` 确认数学 baseline。

---

## 三条核心结论（务必内化）
1. **两套实现，不是一套**（`refs/alibi.h`）：
   - **causal** = 平移不变性丢逐行常数 `-slope·i_q` → 只剩 `+slope·j_k`（**列偏置，所有行共享**）。
   - **non-causal** = 逐元素 `-slope·|i_q - j_k|`。
   - SWA 按是否带因果套其一，再叠既有 SWA `-inf` 窗口掩码。
2. **S 块布局 = BSN，head(N) 为内层**：`row = S·qNBlockSize + N`，连续行同 token 不同 head → slope 逐行变、`i_q` 逐 token 变。`i_q=f(S)` 与 head 无关 → `|i-j|` 跨 head 共享，换 head 只换 slope。
3. **causal 列偏置必须用绝对列** `slope·(kvSStartIdx+col)`，不能用相对 `[0,1,…]`（多 KV-tile 跨 tile 归一化会错，已用 numpy 证明）。

---

## 设计硬约束（用户给定）
- fwd 与 bwd **必须共用同一套 alibi 构造**（反向重算 P 时带同一 bias；bias 无额外梯度）。
- 不改 softmax-bwd 数学、dQ/dK/dV 推导、不加 alibi 梯度；保留现有优化与 pipeline。
- ALiBi 代码**无条件编译**（无 opt-in 宏），是否生效由**运行时**决定：不传 `alibi_slopes` 时 `GetAlibiSlopesRef` 返回 `{nullptr,0}`，kernel 全程 no-op，与原行为等价。
- 接口层与 TriDao 对齐：slopes 以原始 `data_ptr()` + `alibi_slopes_batch_stride` **零拷贝**透传（无 host 广播/拷贝），kernel 索引 `slopes[bidb*batch_stride + head]`；内部 SIMD/SIMT 差异属正常。
- 验证口径：fwd 与既有 alibi 参考一致；bwd dQ/dK/dV 与 PyTorch 一致；不传 slopes 时走原路径、位一致。

---

## 实现顺序（`callchain_and_files.md §6` 详）
1. 跑 `verify/` 确认数学 baseline。
2. 重写 `fag_common/alibi_bias.hpp`（BSN 解码 + causal/non-causal 两路）。
3. 前向：v2 `online_softmax.hpp`（`ApplyAlibi` + `setAlibiTileContext` + 三 `operator()` 插入），跑通非掩码+causal；同步到 v3（逐字节相同）。
4. 反向：`fag_epilogue_op.hpp`（spec1→spec2）复用同一构造。
5. dispatch/API/Python 透传 slope，移除占位 `TORCH_CHECK`。
6. NPU 对拍 + 不传 slopes 时与原路径位一致。
