# ALiBi 实现状态 (v2: `csrc/flash_attn_npu/`)

实现范围：**前向 + 反向，代码完成（code-complete）**。覆盖 v2 树（`flash_attn_npu`）。NPU 精度验证待具备 Ascend 工具链后进行（本环境无工具链，无法编译/上板）。

## 0. 开关方式（运行时控制）

ALiBi 代码**无条件编译**（开发环境，无 opt-in 宏）。是否生效由**运行时**决定：host 侧
`GetAlibiSlopesRef`（**零拷贝**，返回 `{data_ptr(), batch_stride}`）在调用方未传 `alibi_slopes` 时
返回 `{nullptr, 0}` → `alibiSlopesDevice = nullptr` → kernel 侧 `alibiEnabled = (alibi_slopes != nullptr)`
为 false，`ApplyAlibi` 早退、`setAlibiTileContext` 以 `if (alibiEnabled)` 守卫，全部为 no-op。
传 `[nheads]`/`[b,nheads]` fp32 slopes 即启用。即：不传 slopes 与原行为等价。

**接口层与 TriDao 对齐**：slopes 以原始 `data_ptr()` + `alibi_slopes_batch_stride` 透传（无 host 端
`[nheads]→[b,nheads]` 广播/拷贝）；`batch_stride = (dim==2) ? stride(0) : 0`。kernel 索引
`slopes[bidb*batch_stride + head]`（与 TriDao `flash_fwd_kernel.h:287` 完全一致）。前向
`alibiSlopesGmOffset = BIdx*batch_stride + qHeadIdx`、反向 `slopeOffset = bIdx*batch_stride +
n2Idx*groupSize + gIdx`（= query-head 绝对索引）。

## 1. 文件清单与职责

| 文件 | 角色 | 状态 |
|---|---|---|
| `fag_common/alibi_bias.hpp` | **前向/反向共享**的偏置构造（`FaiAlibi::ApplyAlibiRows<IS_CAUSAL>`） | ✅ 新建 |
| `online_softmax.hpp` | 前向 online-softmax epilogue：`bindAlibi/setAlibiTileContext/ApplyAlibi` 成员、UB scratch、`operator()` 增 `kvSStartIdx` 形参、在 NO_MASK 与 CAUSAL 两条路径 ScaleS 之后插入 ApplyAlibi（SWA 路径不在范围内，跳过） | ✅ |
| `mha_fwd_kvcache.cpp` | 前向 call-site：`gAlibiSlopes` 绑定；`FAInfer` 增 `alibi_slopes` 形参；`bindAlibi`（VEC）；`setAlibiTileContext`（VEC-gated）；5 处 NO_MASK `operator()` 调用补 `kvSStartIdx` | ✅ |
| `kernel_common.hpp` | `FAIKernelParams` 增 `GM_ADDR alibi_slopes` + `int64_t alibi_slopes_batch_stride` | ✅ |
| `fwd_dispatch.hpp` / `fwd_dispatch_impl.hpp` | `FwdLaunchArgs.alibiSlopesDevice/alibiSlopesBatchStride` + 10 处 `FAInfer` launch 补传两者 | ✅ |
| `fag_epilogue_op.hpp` | 反向 SubGrapA：`BindAlibiSlopes(ptr, batch_stride)` setter（存 `alibiSlopesBatchStride` 成员）、`alibiSlopesGm/alibiEnabled`、UB 偏移；score-recompute 的 `Muls(scale)` 之后用 `ApplyAlibiRows<false>` 重加偏置 | ✅ |
| `flash_attn_npu_v3/fag_kernel.cpp` | 反向 `FAGGeneral` 增 `alibi_slopes_batch_stride` 形参；`epilogueFAGSabVec` 构造后调用 `BindAlibiSlopes(params.alibi_slopes, params.alibi_slopes_batch_stride)` | ✅ |
| `kernel_common_fag.hpp` | `FAGKernelParams` 含 `GM_ADDR alibi_slopes` + `int64_t alibi_slopes_batch_stride`（透传到 FAGGeneral） | ✅ |
| `fag_general_dispatch.hpp` / `_impl.hpp` | `FagGeneralLaunchArgs.alibiSlopesDevice/alibiSlopesBatchStride`；`GEN_LAUNCH` 透传两者 | ✅ |
| `flash_api.cpp` | host API：**零拷贝** `GetAlibiSlopesRef`（校验 fp32/同设备/末维连续/形状 `[nheads]`或`[b,nheads]`，返回 `{data_ptr(), batch_stride}`，不拷贝）；移除 4 处 `TORCH_CHECK(!alibi_slopes_)`；前向 3 函数设 `fwd_args.alibiSlopes{Device,BatchStride}`；反向 2 函数经 `launch_fag_general(ptr, batch_stride)` 透传 | ✅ |
| `fag_general_host.hpp` / `.cpp` | `launch_fag_general` 末两参改为 `(uint8_t *alibi_slopes_ptr, int64_t alibi_slopes_batch_stride)`；设 `gen_args.alibiSlopes{Device,BatchStride}` | ✅ |
| `flash_attn_npu/flash_attn_interface.py` | **无需改动**：6 个 autograd Function 与 `_flash_attn_(varlen_)forward/bwd`、fake wrapper 已端到端透传 `alibi_slopes`（沿用上游接口，原先仅被 C++ 侧 TORCH_CHECK 拦截） | ✅ 既有 |

## 2. 数学（前向=反向，同一构造）

- **causal**：`bias[i,j,h] = -slope·|i-j|`，causal 区 `i≥j` ⇒ `-slope·i + slope·j`。
  丢弃 per-row 常数 `-slope·i`（online softmax 行 max 吸收，等价），保留 `+slope·j`，
  即**仅依赖列** `j = kvSStartIdx + col`（绝对列）。缓存 `baseCol=[kvSStartIdx, kvSStartIdx+1,…]`，
  KV-tile 变化时重建。
- **non-causal**：`bias[i,j,h] = -slope·|i_q - j|`，`i_q = qPosBase + token`
  （`qPosBase` 已含右下对齐 `diffS=max(0,Sk-Sq)`），`j = kvSStartIdx + col`。
- **反向始终用 general `-slope·|i-j|`（`ApplyAlibiRows<false>`）**：causal 区内与 causal 形式差一个
  per-row 常数（softmax 吸收），故对 causal / non-causal 都正确；且反向 epilogue 以
  `IS_ATTEN_MASK` 模板参数为主、无 `IS_CAUSAL`，此举避免了在反向里额外引入 IS_CAUSAL。
  重算的 `P` 与前向在 fp32 舍入内一致（落在既有 fwd/bwd recompute 容差内）。

## 3. 关键设计点

- **绝对列是必须的**（`impl_alibi/verify/verify_tiled_causal.py` 已证）：online softmax 跨
  KV-tile 合并，不同 tile 的 `kvSStartIdx` 不同，per-tile 常数 `slope·kvSStartIdx` **不能**像
  per-row 量那样丢弃。
- **BSN 布局解码**（前向）：`row = token·qNBlockSize + head`；GQA（`qNBlockSize>1`）下逐行换斜率。
- **2-AIV 行切分**（前向）：`absRow = rowOffsetThisSubBlock + rowOffsetCurLoop + ri` 全局解码，
  每个 AIV 独立对自己负责的行施加偏置；`slopesUb` 在每个 AIV 上冗余载入全部 `qNBlockSize` 个斜率。
- **count-API**：`Adds/Muls/Add/Abs(dst, src, …, count)` 一条指令处理整行 `columnNumRound` 个元素，
  `isSetMask` 处理尾部，无内层 chunk 循环。索引沿用既有 `tensor[offset]`（LocalTensor 视图）写法，
  与 `ScaleS`/既有 `Muls(vecClc2Buffer, …, count)` 完全一致。
- **反向 varlen 路由**：`mha_varlen_bwd` 的优化 varlen-bwd 内核（`launch_varlen_bwd_impl`，独立内核、
  无 ALiBi）在 `alibi_slopes_.has_value()` 时**改走 FAGGeneral 反向**（有 ALiBi）。alibi 关闭时走原路径，
  保持位一致。

## 4. 已知不在范围内（未来工作）

- **v3 前向**（`csrc/flash_attn_npu_v3/`，`flash_attn_npu_3`）：用户明确「先关注 v2」。
  v3 与 v2 的 `online_softmax.hpp` 在本次改动前为**逐字节镜像**；但移植需连同 v3 的
  `mha_fwd_kvcache.cpp`/`fwd_dispatch`/`kernel_common` 全链路同步（否则 `operator()` 的
  `kvSStartIdx` 签名变更会编译报错）。两树为独立构建、各自副本，故 v2 不依赖 v3 副本，当前不影响 v2 正确性。
- **`online_softmax_low_prec.hpp`**：前向低精度 LSE 路径；当前 ALiBi 仅接入 `OUT_ONLY`（`online_softmax.hpp`）。低精度路径是否被启用待确认。
- **反向 varlen 优化内核**（`launch_varlen_bwd_impl`）本身未加 ALiBi（见上「路由」绕过）。

## 5. 待 NPU 验证项（`[需 NPU 验证]`，无工具链无法确认）

1. `CreateVecIndex<float>(tensor, startValue, count)` 的确切签名（用户确认为官方 ramp API）。
2. `Muls/Add/Abs` 的 count 形（4 参、`isSetMask` 内置）——与用户确认的 `Adds` count 形同族。
3. `LocalTensor::GetValue/SetValue`（UB↔scalar）在 AIV 上的可用性。
4. UB 偏移无碰撞：前向 `12*UB_UINT8_BLOCK_SIZE`（slope+work+baseCol）；反向 `110K/111K`。
5. `PipeBarrier<PIPE_V>` 在同张量上串行 count-op 间的同步语义。
6. `params.alibi_slopes != nullptr` 判空（`GM_ADDR` 为 `__gm__ uint8_t*` 类，可 `!= nullptr`）。

## 6. 自测路径（具备工具链后）

- 编译：`python setup.py bdist_wheel`（ALiBi 已无条件编入）。
- 数值对拍：`impl_alibi/verify/verify_tiled_causal.py` 的因果/绝对列结论 + 与 PyTorch 参考
  `softmax(scale·S + bias)` 的端到端前向/反向对比（causal & 无掩码，含 GQA、`Sk≠Sq`）。
