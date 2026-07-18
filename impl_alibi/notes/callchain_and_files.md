# 调用链 + 文件清单（实现 Alibi 的代码地图）

> 来源：对 `main` 分支的**源码静态阅读**（file:line）。环境无 Ascend 工具链，运行期行为标 `[推断]`。
> 本文件只含**正确**结论；先前 agent_research 中「head-major 布局 / 统一 alibi 构造」等错误论断已剔除（见 `design_review.md` §0）。
> 配合：`design.md`（用户原始设计）、`design_review.md`（修正评审）、`alibi_math.md`（数学/布局）。

---

## 0. 仓库结构事实
- `csrc/flash_attn_npu/` —— v2（模块 `flash_attn_npu_2`，fwd+bwd 完整）。
- `csrc/flash_attn_npu_v3/` —— v3，**Ascend 910** 后端（模块 `flash_attn_npu_3`，fwd+bwd 完整）。
- `csrc_AscendC950/flash_attn_npu_v3/` —— v3 的 **950** 后端（**仅 fwd**，架构独立，不支持 SWA）。
- `csrc/catlass`、`csrc_AscendC950/catlass` —— Ascend 的 CUTLASS 等价物，**子模块未 checkout**，按用法推断 API。
- `flash_attn_npu/`、`flash_attn_npu_v3/` —— Python API；`tests/` —— 精度/反向测试。

### Alibi 现状（一致基线）
- v2 `flash_api.cpp` 的 `mha_fwd`/`mha_varlen_fwd`/`mha_bwd`/`mha_varlen_bwd` 用 `TORCH_CHECK(!alibi_slopes_.has_value(), ...)` **占位拦截** —— 这是待实现的临时检查，**实现时移除**（用户确认）。
- v3 已从 API 移除 alibi（仅 docstring 残留）。
- `FAGKernelParams.alibi_slopes`（`kernel_common_fag.hpp`）是**死字段**（恒 nullptr，从不读取）；`pseAlibiAddr`（`fag_kernel.cpp`）只是确定性路径的 workspace 分区，**名字误导，与 alibi 无关**。⚠️ 先前 findings.md「v3 已有 alibi 脚手架」系误判，作废。
- **结论：Alibi 在 v2/v3 运行期完全不存在，从干净基线实现。**

---

## 1. 前向调用链（v2 与 v3 同构）
```
flash_attn_func / flash_attn_varlen_func
→ _flash_attn_forward(custom_op) → flash_attn_npu_{2,3}.fwd (pybind)
→ mha_fwd (flash_api.cpp)                         # v2 alibi 占位拦截在此
→ FwdLaunchArgs + launch_fwd<IS_TND>(fwd_args)
→ fwd_dispatch.hpp launch_fwd → fwd_dispatch_impl.hpp
   按 (is_local/is_causal/none × paged × flashDecode × layout) 实例化
   SplitFuse::FAInfer<…, MaskType, …, IS_FD>      # MaskType 编译期
→ mha_fwd_kvcache.cpp 提供 FAInfer kernel + FAIKernelParams(含 GM_ADDR mask)
→ kernel 内部：
   blockMmadQK (qk_matmul.hpp) → S=QK^T 写 GM (**不施加 mask/bias**)
   → online_softmax.hpp BlockEpilogue:
       CopySGmToUb → ScaleS(Muls S,scale)
       → [ApplyMask: Muls(mask,-3e38) → Add(S)]   # 二值 mask 本就是加性
       → ★ALIBI 插入点: ScaleS 之后、CalcLocalRowMax 之前★
       → CalcLocalRowMax → UpdateGlobalRowMax → CalcExp
       → CalcLocalRowSum → UpdateGlobalRowSum → DownCastP → CopyPUbToGm
   → pv_matmul.hpp O=P·V → rescale_o*.hpp/CombineScale.hpp 在线 rescale → gO/gLse
```
- **autogen 桩**（`fwd_dispatch_<dtype>_<layout>.cpp`）只按 dtype×layout 实例化 `launch_fwd_impl`，**不按 mask 分** → 新增 MaskType/Alibi 分支只改 `fwd_dispatch_impl.hpp`，**无需重跑 generate_kernels.py**。
- scale 在 `epilogue.init(resource, scaleValue)` 传入；alibi slope 是 **per-head、随 tile 变化**，**不能走 init**，须走 GM `alibiSlopes` 指针 + tile 内 head 索引。
- varlen 前向：v2 有独立 `mha_varlen_fwd`（`launch_fwd<true>` TND）；v3 靠 `mha_fwd` 内 `is_varlen_q` 分流。

## 2. 反向调用链（v2 与 v3 **共用同一 kernel**）
```
mha_bwd / mha_varlen_bwd (flash_api.cpp)
→ launch_fag_general(...)                          # bwd 总入口
→ FagGeneralLaunchArgs (fag_general_dispatch.hpp)
→ fag_general_dispatch_impl.hpp:  #include "../flash_attn_npu_v3/fag_kernel.cpp"
   ↑ v2 反向直接复用 v3 的 ::FAGGeneral kernel
→ ::FAGGeneral<AlignedNNN, DType, Layout, IS_CAUSAL, 0, IS_DTM>
→ FlashAttentionScoreGrad::operator()<AIC>/<AIV> (fag_kernel.cpp)
   AIC : ComputeMM1(dy·V, Q·K^T) → ComputeMMDqkv/DTMComputeMMDqkv(dq,dk,dv)
   AIV : EpilogueFAGPre → EpilogueFAGSfmg(SoftmaxGrad)
         → EpilogueFAGSabVec(S=Mask(QK^T), **重算 P**, dS)   ★ALIBI 反向点★
         → EpilogueFAGPost → EpilogueFAGDtmAdd
```
- v3 反向的 AIV epilogue 类与 FAG mmad 块**逐字 include 自 v2**：`fag_kernel.cpp` `#include "../flash_attn_npu/{fag_block.h, kernel_common_fag.hpp, fag_epilogue_*.hpp}"` + `fag_common/`。
- bwd **重算 P**：`P = softmax(scale·S + alibi_bias)`，须带上与 fwd 相同的 bias。**bias 是常量（与 Q/K 无关）→ ∂bias/∂Q=∂bias/∂K=0 → 反向无额外梯度项，只需 P 重算施加 bias。dQ/dK/dV 公式不变（在 biased P 处求值）。** 这是 fwd/bwd **必须共用同一 alibi 构造**的根因。

---

## 3. v2 ↔ v3 架构对比（影响实现范围）

| 维度 | v2 | v3 (910) | 共享/差异 |
|---|---|---|---|
| 前向 kernel | `SplitFuse::FAInfer` | `SplitFuse::FAInfer`（重写） | 结构平行，文件各自一份 |
| flash-decode | 模板参 `IS_FD` | tiling 轴 `flashDecodeFlag` | 差异 |
| mask | 编译期 MaskType (NO/CAUSAL/SWA) | 同（+SWA，commit 3ccb96e） | 同构 |
| 反向 kernel | `launch_fag_general` → **include v3 fag_kernel.cpp** | `::FAGGeneral`(fag_kernel.cpp) | **同一 kernel，v2 反向=复用 v3** |
| 反向 epilogue | `fag_epilogue_*.hpp`(v2) | include v2 `fag_epilogue_*.hpp` | **v3 反向=复用 v2 epilogue** |

**逐字节相同（v2==v3，diff=0）的文件**：`online_softmax.hpp`、`online_softmax_low_prec.hpp`、`qk_matmul.hpp`。
→ **改 `online_softmax.hpp` 必须两树同步**（硬拷贝，非软链）。

---

## 4. 插入点（前向 + 反向，源码验证）

**前向**：`online_softmax.hpp` 的 softmax epilogue 内，`ScaleS`（`:516`）**之后**、`CalcLocalRowMax`/`ApplyMask` **之前**。三处 `operator()`：
- 无 mask `operator()`（`:930`）：`ScaleS → ApplyAlibi → SubCoreCompute<false>`。
- causal 单 mask `operator()`（`:1011`）：`ScaleS → ApplyAlibi → ApplyMask → SubCoreCompute<true>`（alibi 与 mask 皆加性，顺序对 P 无影响，但 alibi 须在 row-max 前）。
- SWA 双 mask `operator()`（`:1172`）：`ScaleS → ApplyAlibi → (pre/next mask pingpong) → SubCoreCompute<true>`。

**反向**：`EpilogueFAGSabVec`（`fag_kernel.cpp:793/820`）重算 `S=Mask(QK^T)` 之后、softmax 重算 P 之前施加同 bias。

**为何此点正确且最优**：bias 必须在 row-max/exp 之前生效，才能进入在线统计量 m/l（影响 rescale 后的 P 与 LSE）；exp 之后再加破坏归一化。数据已在 UB、已是 fp32、已过 ScaleS，`S += bias` 仅再叠向量指令，无额外 GM 往返、无 DMA，与既有加性 mask 同构。详见 `design_review.md` §3 + 旧 analysis.md §6.3。

---

## 5. 需要修改的文件清单（实现时的改动地图）

> 设计原则：**fwd/bwd 共用同一 alibi 构造**（`fag_common/` 下单一 helper，按 MaskType 分 causal/non-causal 分支）；ALiBi **无条件编译**、运行时控制（不传 slopes 即 no-op）；接口层与 TriDao 对齐——slopes 以 `data_ptr()`+`alibi_slopes_batch_stride` **零拷贝**透传，kernel 索引 `b*stride+head`。

| 文件 | 改动 | 职责 |
|---|---|---|
| `fag_common/alibi_bias.hpp`（**新增**） | **重写**（旧版 head-major + 全 \|i-j\| 作废） | BSN 解码 + causal(列偏置)/non-causal(\|i-j\|) 两路；fwd/bwd 共用 |
| `online_softmax.hpp`（v2+v3 各一份，**逐字节相同**） | `ApplyAlibi` 方法 + `setAlibiTileContext` + UB scratch + 三 `operator()` 调用插入 | 前向 alibi 施加点 |
| `fag_epilogue_op.hpp`（v2，spec1 + spec2 两处） | SubGrapA 重算 score 后施加同 bias；构造时绑 `alibiSlopesGm` | 反向 alibi 重算点 |
| `kernel_common.hpp` / `kernel_common_fag.hpp`（v2+v3） | `FAIKernelParams`+`alibiSlopes`；padding→`isAlibi`；`FAGKernelParams.alibi_slopes` 激活 | kernel 参数透传 |
| `tilingdata.h`（v2+v3） | `padding3→isAlibi`（复用槽，布局不变） | 运行期 gate |
| `mha_fwd_kvcache.cpp`（v2+v3） | `runMainLoop` 算 qSBlockSize/qNBlockSize/kvSStartIdx 后 `setAlibiTileContext(...)` | tile 上下文 |
| `fwd_dispatch{,_impl}.hpp`、`fag_general_dispatch{,_impl}.hpp`、`bwd_dispatch{,_common}.hpp`（v2/v3） | `*LaunchArgs`+`alibiSlopesDevice`；GEN/BWD_LAUNCH 宏透传 | dispatch 透传 |
| `flash_api.cpp`（v2+v3） | `mha_fwd/bwd` 接收 `alibi_slopes`、广播 [b,nheads]、传给 launch；**移除 TORCH_CHECK 占位** | C++ API |
| `flash_attn_interface.py`（v2+v3） | `_flash_attn_forward/_fake` 加 `alibi_slopes`，`FlashAttnFunc` autograd 透传 bwd | Python API |

> **不要**移植旧 agent_research 里的：`alibi_bias.hpp`（head-major 错）、`final_report.md`/`progress.md`/`findings.md`（基于错误假设）、`design.md`(agent_research 版，已被 impl_alibi/design_review.md 取代)。

---

## 6. 实现顺序建议
1. **先验证数学**：跑 `verify/alibi_ref.py`（向量化 vs 朴素）、`verify/verify_tiled_causal.py`（绝对列）—— 确认 baseline。
2. **写 `alibi_bias.hpp`**（BSN 解码 + causal/non-causal 两路），`[需 NPU 验证]` 标注 intrinsic/UB。
3. **前向**：v2 `online_softmax.hpp` + `setAlibiTileContext`，跑通非掩码 + causal；同步到 v3。
4. **反向**：`fag_epilogue_op.hpp` spec1，复用同一构造；再 spec2。
5. **dispatch/API/Python** 透传 slope，移除占位 TORCH_CHECK。
6. NPU 上对拍：fwd 与既有 alibi 参考一致；bwd dQ/dK/dV 与 PyTorch 一致；不传 slopes 时与原路径位一致。
