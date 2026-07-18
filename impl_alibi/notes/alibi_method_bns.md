# ALiBi 构造方法设计文档（BNS 布局）

> 状态：**前向已实现并 review；反向已实现（§11，仿 softcap 编译期模板穿透），待编译/测试验证**。
> 关系：取代 `design.md` 中已声明"过时"的下半部分；布局定论见 §1。
> 适用：前向 `online_softmax.hpp` 三个 operator() 与反向 `fag_epilogue_op.hpp` 重算，共用 `alibi.hpp` 里的 `ApplyAlibiRows`。

---

## 0. 目标

在线 softmax epilogue（前向）与 score 重算（反向）中，对 score 块 `S` 叠加 ALiBi 偏置：

- 非因果 / SWA：`bias[i,j,h] = -slope_h · |i_q - j_abs|`
- 因果：`bias[i,j,h] = slope_h · j_abs`（`j_abs = kvSStartIdx + col`）

本文给出**适配 S 真实 BNS 行排布**的向量化构造方法与跨行复用策略。

---

## 1. 布局定论与行解码

- **Q 输入张量**：`[B, S_q, N_q, D]`（BSHD）。
- **S = QK^T 输出（alibi 作用对象）**：`[B, N_q, S_q, S_k]`（**BNS，head-major**）。

  - 三条独立证据互洽：
    1. mask 复制（`CopyMaskGmToUb`）按 `tokenNumPerHead=qSBlockSize` 行/头块整块复制 `integralHeadNum` 次；
    2. 2-AIV 切分 `rowSplitSubBlock = qSBlockSize·qNSplitSubBlock` 按 head 连续切块；
    3. Nd2Nz 分形（`ndNum=qSBlockSize` 矩阵 × `[qNBlockSize,embed]` + `dstNzMatrixStride`）使 M 轴 head-major 落地。
- 单个 S tile 的 M 轴（行）解码（**head 在外、sequence 在内**）：

  ```
  head  = absRow / qSBlockSize      // 慢/外层索引 ∈ [0, qNBlockSize)
  token = absRow % qSBlockSize      // 快/内层索引 ∈ [0, qSBlockSize)
  i_q   = qPosBase + token          // 该行 Q 的绝对位置（qPosBase 已含 diffS）
  ```
- **核心推论**：sequence 维连续 ⇒ **同一 head 内相邻行 slope 相同、token 每行 +1**。这是 §4/§5 所有优化的基石。
- 常见情形 `qNBlockSize == 1`：整 tile 单 head，BSN 与 BNS 解码等价（旧 bug 潜伏）；本方案在 `qNBlockSize>1`（短序列 + GQA）时才显现差异与收益。

> 旧 `alibi.hpp` 头注释与三处特化用的是 **BSN 解码**（`head=absRow%qNBlockSize, token=absRow/qNBlockSize`），对 BNS 是**错的**，须按上式改正。

---

## 2. 影响范围（改 / 不改）

**需要改：**

- 行 → (head, token) 解码（`/` 与 `%` 互换，除数由 `qNBlockSize` 改为 `qSBlockSize`）。
- 跨行 slope 连续性的利用（同 head 内批量 Add / 增量更新 bias）。
- `ApplyAlibiRows` 增加 `qSBlockSize` 入参（解码除数 / head 块大小）。

**无需改（与行内排布无关，基于绝对位置，已正确）：**

- 列向绝对位置 `j_abs = kvSStartIdx + col`（causal bias 仍是 `slope·j_abs`，只是不再单独缓存 ramp、改 in-place 比值——见 §4）。
- `qPosBase = alibiDiffS + qSBlockIdx·curQSBlockTile`。
- slopes 的 GM→UB 加载：`slopesUb[h] = slopesGm[BIdx·stride + qNStartIdx + h]`，`h∈[0,qNBlockSize)`（`headInTile` 范围正好 `[0,qNBlockSize)`，索引自洽）。

---

## 3. 行段（head segment）切分（causal / NO_MASK 共用）

chunk 行范围 `[absRowStart, absRowStart + rowNumCurLoop)`。head `h` 占行 `[h·qSBlockSize, (h+1)·qSBlockSize)`。chunk 可能跨 head 边界，切成：

- **prologue 段**：从 `absRowStart` 到下一个 `qSBlockSize` 边界；
- **若干完整 head 段**（各 `qSBlockSize` 行）；
- **epilogue 段**：剩余。

每段内 head / slope 恒定。段头 head 索引 = `segStartRow / qSBlockSize`，slope = `slopesUb[head]`。

> `qNBlockSize == 1` 时整 chunk 即单段、无边界——最常见、最简。
> 该结构与 mask 代码的 pro/integral/epi 同构，可参照。

---

## 4. 因果掩码（MASK_CAUSAL）

`bias[i,j,h] = slope_h · (kvSStartIdx + j)` —— **与行 `i` 无关**。同一 head 的所有行偏置**完全相同**。

**单 buffer + 比值增量**（§8.2 决议：UB 紧张、不另存 baseCol；每次 `ApplyAlibiRows` 调用重建裸 ramp，用 `workUb` in-place 维护 "ramp·slope"）：

- 调用入口（每个 row-chunk / 每次 `ApplyAlibiRows`）：`CreateVecIndex(workUb, kvSStartIdx)` 写入裸 ramp `[kvSStartIdx, kvSStartIdx+1, …]`；`preSlope = 1`（占位符：表示 workUb 当前是裸 ramp，`Muls(workUb, slope/1)` 即得 `ramp·slope`；且 `slope==1` 时裸 ramp 本就等于 bias，无需 Muls）。
- 逐行：取该行 head 的 `slope`；若 `slope != preSlope`：`Muls(workUb, workUb, slope/preSlope)`（in-place 比值）；`Add(score[row], score[row], workUb)`；`preSlope = slope`。

```
# 全局部变量，无需跨调用持久化
CreateVecIndex(workUb, kvSStartIdx, count=N)     # 裸 ramp
preSlope = 1.0f
for ri in [0, rowNumCurLoop):
    absRow = absRowStart + ri
    head   = absRow / qSBlockSize
    slope  = slopesUb.GetValue(head)
    if (slope != preSlope):
        Muls(workUb, workUb, slope / preSlope, count=N)   # in-place 比值（首 head: slope/1）
    Add(score[row], score[row], workUb, count=N)          # 逐行（§8.1：不做广播）
    preSlope = slope
```

**收益**：每调用 1 次 ramp 重建、每 head 边界 1 次 Muls、逐行 1 次 Add。`qNBlockSize==1` 时每调用仅 1 次 Muls（vs 旧实现每行 1 次 Muls）。`baseColUb` 与**跨调用持久化状态都不需要** → 省 UB、零状态。
**决议**：§8.1——AscendC 的 Add 不支持单行→多行广播，故逐行循环；§8.2——in-place 比值（待跑通后再研究用空余 UB 改"缓存 baseCol、跨 KV-tile 复用 ramp"，届时再加回 `lastKvSStart` 跟踪）。

---

## 5. 无掩码 / SWA（NO_MASK / MASK_SWA）

`bias[i,j,h] = -slope_h · |i_q - (kvSStartIdx + j)|`。令 `baseColIdx = i_q - kvSStartIdx`，则
`bias[j] = -slope · |j - baseColIdx|`。
同 head 内 slope 恒定、token 每行 +1 ⇒ `baseColIdx` 每行 +1（`delta=1`）。

> SWA 暂与 NO_MASK 同形（窗口方向无关的 `|i-j|`，安全），保留独立特化以便将来发散。

### 5.1 段首行：从零构造 bias2（≤4 op）

```
baseColIdx = i_q - kvSStartIdx            # i_q = qPosBase + token
CreateVecIndex(bias0, -baseColIdx, N)     # bias0[j] = j - baseColIdx,  N = columnNumRound
if baseColIdx >= 0:
    Abs(bias0, bias0, count=min(baseColIdx, N))   # 仅翻转负前缀 → bias1（case1 时为 no-op）
Muls(bias2, bias1, -slope)                # bias2[j] = -slope·|j-baseColIdx|
Add(score[row], score[row], bias2)
```

三 case 图示（N=8）：

```
baseColIdx=-3: bias0=[3 4 5 6 7 8 9 10]   Abs(no-op)  → 同           *-s
baseColIdx= 3: bias0=[-3 -2 -1 0 1 2 3 4] Abs(cnt=3)  → [3 2 1 0 1 2 3 4] *-s
baseColIdx=12: bias0=[-12 .. -5]          Abs(cnt=8)  → [12 11 10 9 8 7 6 5] *-s
```

### 5.2 段内跨行：增量更新 bias2（delta=1，同 head，不换 slope）

新零点 `baseColIdx' = baseColIdx + 1`。`Δ[j] = -slope·(|j-baseColIdx'| - |j-baseColIdx|)`：

- **case1 零点在块左**（`baseColIdx' ≤ 0`）：全列 `bias2 += +delta·slope`。
  `Adds(bias2, +delta·slope, 全 N)`。**1 op**。
  例：`(3 4 5 6 7 8 9 10)*-s → (2 3 4 5 6 7 8 9)*-s`（bias1 减 1 ⇒ bias2 加 s）。
- **case3 零点在块右**（`baseColIdx ≥ N-1`）：全列 `bias2 += -delta·slope`。
  `Adds(bias2, -delta·slope, 全 N)`。**1 op**。
- **case2 零点在块内**（`0 < baseColIdx' < N`）：左半 `[0, baseColIdx')` 减、右半 `[baseColIdx', N)` 加。按 64 元向量分段 + `SetVectorMask` 处理零点所在的不完整向量（`N=columnNumRound`）：

  - `num1 = baseColIdx' / 64`，`x = baseColIdx' % 64`；
  - `[0, num1·64)`：`Adds(-delta·slope)`，repeat = `num1`（不设 mask）；
  - 零点向量 `[num1·64, (num1+1)·64)`：前 `x` 位 `Adds(-delta·slope)`（mask 前 x）、后 `64-x` 位 `Adds(+delta·slope)`（mask 后 64−x）；
  - `[(num1+1)·64, N)`：`Adds(+delta·slope)`，repeat = `N/64 - num1 - 1`。
  - **2~4 op**。

  例（`baseColIdx=3 → 4`）：`(3 2 1 0 1 2 3)*-s → (4 3 2 1 0 1 2)*-s`，左 `[0,4)` 减 s、右 `[4,7)` 加 s。

更新后 `Add(score[row], score[row], bias2)`（bias2 与 workUb 同址，原地更新）。

> `delta` 一般为 1（token 每行 +1）。若将来支持跨多行跳转，`delta>1` 公式不变，仅 case2 分段点用 `baseColIdx'`。

### 5.3 head 边界（段首）

token 回绕到 0 ⇒ `baseColIdx` **后跳**（非 +1），**不能**用 §5.2 增量。段首行一律按 §5.1 从零构造（同时换 head 则 slope 取新值）。即 **"换 head ⇒ 重算"**。

> **退化选项**：若 §5.2 的 case 分支实现易错，可统一"每行从零构造（≤4 op）"，**仍正确**、仅略慢。建议先按增量实现，验证遇阻再退。

---

## 6. 数据结构与参数

- `ApplyAlibiRows` 入参 **新增 `uint32_t qSBlockSize`**（解码除数 / head 块大小）；**移除 `baseColUb` 与 `lastKvSStartInOut`**（causal 改每次调用重建 ramp，NO_MASK 本就不用列缓存）。`qNBlockSize` 不再传入本函数（仅 `operator()` 装 `slopesUb` 时用）。
- UB（沿用现有 `ALIBI_*_UB_OFFSET`）：

  - `slopesUb`[qNBlockSize] —— 每 head 斜率；
  - `workUb`（`columnNumRound` floats）—— 单行偏置（bias2），比值/增量 in-place 更新同址；
  - **`baseColUb` 移除**（§8.2）→ 释放 `ALIBI_BASECOL_UB_OFFSET`。
- 跨行状态（`preSlope` / `prevHead` / `lastBaseColIdx`）均为 `ApplyAlibiRows` 内**局部变量**，**无持久化成员、无跨调用状态**。
- 函数签名（提案）：

  ```cpp
  template <AlibiMaskType MASK_TYPE>
  void ApplyAlibiRows(scoreUb, scoreOffset, rowStride, columnNumRound,
                      absRowStart, rowNumCurLoop,
                      qSBlockSize,            // ← 新增
                      qPosBase,
                      slopesUb, workUb,
                      kvSStartIdx);           // 无 baseColUb、无 lastKvSStartInOut
  ```

---

## 7. 边界情况与实现备注

- `columnNumRound` 可达 512（`MAX_KV_STACK_LEN`，多向量）；§5.2 case2 分段按 64 元向量处理，参考 `ApplyMask` 分支 B。§4 causal 逐行 `Add(..., count=N)` 由 `isSetMask` 处理尾部，无需内层向量循环。
- 仅 `[0, columnNum)` 列有效；对齐尾柱加有限偏置无害（与现状一致，下游 rowmax/rowsum 只在 `columnNum` 上归约）。
- `baseColIdx` 可为负（零点块左）/ 块内 / ≥ N（块右）：§5.1 Abs 步与 §5.2 case 分类已全覆盖。
- head 边界仅 `qNBlockSize>1` 且 chunk 跨边界时出现；`qNBlockSize==1` 全程单段。
- **反向**（`fag_epilogue_op.hpp`）：单 head/tile，两处调用点传 `qSBlockSize = tile 行数`（class1 `SameAbVec`: `s1ExtendSubGraph`；class2 `FAGOp`: `s1Extend`），解码退化为 `head=0, token=row`。反向的完整集成（编译期 `HAS_ALIBI` 穿透 + slopes 指针独立 arg）见 **§11**。

---

## 8. 决议记录（已拍板）

1. **causal 跨行广播 Add**：`src1 repeatStride=0`（bias 行广播到多行）AscendC 是否支持？
   - 支持 ⇒ §4 批量 Add 有加速；
   - 不支持 ⇒ 退逐行 Add（仍正确，causal 无加速，但 BNS 解码修正照常生效）。
     答复：我研究了这点，目前似乎AscencC不支持。AscendC的Add（https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/API/ascendcopapi/atlasascendc_api_07_0035.html）暂时不支持一次性完成多行向量与单行向量的逐行相加。那么我们还是套用for循环吧：for each row in Q_til: Add(row, bias）
2. **causal 每 head 的 bias**：推荐"每段从 `baseColUb` 现 `Muls`"（无误差累积）；是否改用"维护单 bias 行 + 比值 `Muls` 增量"？
   答复：你的建议很好，比值 slope/pre_slope 增量可能累积误差，但是实际上，现在的UB的空间已经很紧张，我们可能没有空间来存这个初始bias，而是只能在待斜率信息的bias中in-place地更新——我建议先保留后者这种做法，待所有代码跑通后，我们再尝试改用你说的方法（到时候再研究是否有空余空间UB可以用）
3. **NO_MASK 增量更新（§5.2）**：是否值得 case 分支复杂度？或统一"每行从零构造（≤4 op）"更稳？倾向先增量、遇阻再退。
   答复：我建议是保留case 分支的增量更新，这样可以减少计算量
4. **SWA**：是否需要独立形（如因果列式）？暂同 NO_MASK。
   答复： SWA暂时同NO_MASK，至少保证计算结果正确，后续再优化

---

## 9. 代码改动清单（本文确认后执行）

1. `csrc/flash_attn_npu/alibi.hpp`
   - 头注释（BSN→BNS）与三处特化的解码公式改正：`head=absRow/qSBlockSize, token=absRow%qSBlockSize`；
   - 签名加 `qSBlockSize`、**去掉 `baseColUb` 与 `lastKvSStartInOut`**；
   - causal（§4）：入口 `CreateVecIndex` 重建 ramp + 局部 `preSlope=1` + in-place 比值 + 逐行 Add；
   - NO_MASK/SWA（§5）：段首行从零构造、段内 case 分支增量、head 边界重算，逐行 Add。
2. `csrc/flash_attn_npu/online_softmax.hpp`
   - `init`：移除 `alibiBaseColUb` 分配（释放 `ALIBI_BASECOL_UB_OFFSET`）；移除 `alibiLastKvSStart` 成员；
   - `ApplyAlibi`：透传 `qSBlockSize`，去掉 `baseColUb` / `alibiLastKvSStart`；
   - operator() ALiBi 初始化块：仅装载 slopesUb（无 state 重置）。
3. `csrc/flash_attn_npu/fag_epilogue_op.hpp`
   - class1 / class2 两处 bwd 调用点：传 `qSBlockSize = tile 行数`（class1 `s1ExtendSubGraph`、class2 `s1Extend`），并适配新签名（去 baseColUb / lastKvSStart）。
   > **注**：反向的实际实现远超此早期提案 —— 编译期 `HAS_ALIBI` 模板穿透、slopes 指针走独立 kernel arg、batchStride 走 tiling、运行时 `alibiEnabled` 改编译期。详见 **§11**。
4. `impl_alibi/design.md`
   - 按 BNS 更新过时段落，链接本文。

## 10. 补充优化
1. slopeValues的读取
  - 原先会存放在GM，形状为 num_heads 或者 b x num_heads
  - 在OnlineSoftmax阶段需要使用它们（至多需要当前batch的num_heads个fp32）
  - 怎么在AIV的计算中获取它们？
    - 方法1：UB上预留num_heads 个 fp32 空间存放，在每个SM计算开始时，将所需的数据从GM加载到UB。
      如何加载，两者做法：由于数据量不大，不确定哪种方法更合适
        1. 批量加载（占用MTE2，需要同步事件
        2. 以标量形式逐个加载（目前实现方法）：通过GetValue、SetValue实现。
      > 文档提示：不要大量使用SetValue对LocalTensor进行赋值，会使性能下降。
    ```c++
      if constexpr (HAS_ALIBI_) {
      for (uint32_t h = 0; h < qNBlockSize; ++h) {  
          alibiSlopesUb.SetValue(h, alibiSlopesGm.GetValue(alibiSlopesGmOffset + h));
      }
      AscendC::PipeBarrier<AscendC::PIPE_V>();
    ```
    - 方法2：不显式加载到UB，而是在需要时直接用GetValue从GM获取，参照代码中参数gBlockTable的做法（要求）
    ``` C++
        // csrc/flash_attn_npu/qk_matmul.hpp line 177~182
      __aicore__ inline
      void getKVOffset(AscendC::GlobalTensor<int32_t> &gBlockTable, uint32_t &kOffset, uint32_t nowNIdx, 
          uint32_t startOffset, uint32_t strideKV, uint32_t blockSize)
      {
          uint32_t blockTableId = gBlockTable.GetValue(nowNIdx);
          kOffset = blockTableId * blockSize * strideKV + startOffset * strideKV;
      }

    ```
  > **已采纳方法2**：`ApplyAlibiRows` 直接收 `GlobalTensor<float> &slopesGm + slopesGmOffset`，需要某 head 的 slope 时 `slopesGm.GetValue(slopesGmOffset + head)` 现读，不在 UB 预存。前向 `online_softmax.hpp` 已移除 `alibiSlopesUb` 成员与三处 `for(h) SetValue` 加载块。

---

## 11. 反向传播（FAG）ALiBi 集成（已实现）

> 本节记录反向的实际实现，**取代 §7/§9.3 中关于反向的旧描述**（旧描述是"运行时 alibiEnabled + 仅传 qSBlockSize"的早期思路，现已演进为编译期模板穿透）。
> 参考样板：`impl_alibi/flash-attention-npu-smh/csrc/arch22/flash_attn_npu_v2` 下 **softcap 反向**的实现方式（"参数传递方式是一样的"）。

### 11.1 设计原则（仿 softcap，与前向统一）

| 项 | 做法 | 说明 |
|---|---|---|
| **门控** | 编译期模板参数 `HAS_ALIBI` | 从 `flash_api.cpp`（是否传了 slopes）一路穿透到 dispatch policy；epilogue 内 `if constexpr (HAS_ALIBI)`，不启用时整体（含 UB scratch）编译消除 |
| **slopes 指针 `alibi.ptr`** | **独立 kernel `GM_ADDR` 参数**（像 q/k/v） | 地址指针不入 tiling；前向 `flash_api.cpp` 里 `set_alibiSlopesAddr` 本就是注释掉的 |
| **`alibi.batchStride`** | tilingdata | 标量，走 tiling（`alibiSlopesBatchStride` 字段） |

> **关键纠错**：早期曾把 slopes 指针放进 tiling（`tilingData->alibiSlopesAddr`）+ 运行时 `alibiEnabled` flag，这与前向不一致、且 `FAGKernelParams` 构造 mismatch 导致反复编译失败。**现已全部改为：指针走独立 arg、batchStride 走 tiling、门控走编译期 `HAS_ALIBI`**。tiling 结构与 FAGInfo 中的 `alibiSlopesAddr` 字段已全部移除（仅保留 `alibiSlopesBatchStride`）。

### 11.2 两条反向内核路径与 dispatch 拓扑

```
mha_varlen_bwd / mha_bwd  (flash_api.cpp)
   │
   ├─ 分支A: !same_seqlen || is_local || headdim!=128   ──► launch_fag_general
   │     → v3 FAGGeneral 内核 (EpilogueAtlasA2SameAbVec = class1)
   │       fag_general_host → fag_general_dispatch_impl (8路: has_alibi×is_causal×deterministic ×4 headdim)
   │
   └─ 分支B: 否则 (自注意, headdim=128)   ──► varlen_bwd_dispatch
         → v2 FAGVarlenOpt 内核 (EpilogueAtlasA2FAGOp = class2)
           varlen_bwd_dispatch_impl (4路: has_alibi×is_causal)
```

- **class1 `EpilogueAtlasA2SameAbVec`**（v3 FAGGeneral 路径）：单 head/tile，`qSBlockSize = s1ExtendSubGraph`。
- **class2 `EpilogueAtlasA2FAGOp`**（v2 FAGVarlenOpt 路径）：单 head/tile，`qSBlockSize = s1Extend`。
- 两 class 解码均退化为 `head=0, token=absRow`（BNS 下 qNBlockSize=1），都用通用 `-slope·|i_q - j|` 形式（对因果/非因果都正确：因果区 `|i-j|=i-j`，与列式 `slope·j` 仅差一个 softmax 不变的行常数）。

### 11.3 编译期 `HAS_ALIBI` 穿透链

```
flash_api.cpp  has_alibi = (alibi.ptr != nullptr)
   │  vb_args.has_alibi / gen_args.has_alibi
   ▼
dispatch_impl  按 has_alibi 二分实例化 kernel 模板
   │  FAGVarlenOpt<..., HAS_ALIBI=true/false>  /  FAGGeneral<..., HAS_ALIBI=...>
   ▼
kernel 模板第N参 bool HAS_ALIBI  →  Epilogue policy 实例化为 <HAS_ALIBI>
   ▼
BlockEpilogue 特化  static constexpr bool HAS_ALIBI  →  if constexpr (HAS_ALIBI) { ApplyAlibiRows(...) }
```

涉及文件：
1. `fag_block.h` — `EpilogueAtlasA2FAGOp<HAS_ALIBI_>`、`EpilogueAtlasA2SameAbVec<...,HAS_ALIBI_>`（均加 `static constexpr bool HAS_ALIBI`）。
2. `mha_varlen_bwd.cpp` — v2 `FAGVarlenOpt` 模板加 `bool HAS_ALIBI=false`；`EpilogueAtlasA2FAGOp<HAS_ALIBI>`。
3. `flash_attn_npu_v3/fag_kernel.cpp` — v3 `FAGGeneral` 模板加 `const bool HAS_ALIBI=0`；`SameAbVec<...,HAS_ALIBI>`。
4. `varlen_bwd_dispatch_impl.hpp` — 4 路实例化。
5. `fag_general_dispatch_impl.hpp` — `GEN_LAUNCH` 宏加 `IS_ALIBI` 第4参、`FAGGeneral` 第7模板参；8 路派发。

### 11.4 slopes 指针独立 arg 路径（两内核）

**v3（FAGGeneral）**——kernel signature 本就有 `GM_ADDR alibi_slopes_` 槽位（dv 之后、workspace 之前）：
```
FAGGeneral(..., dk_, dv_, alibi_slopes_, workspace, tiling)
   → FAGKernelParams{..., dv_, alibi_slopes_, workspace, tiling}   // FAGKernelParams 加了 alibiSlopes 字段
   → epilogue 构造传 params.alibiSlopes
   → class1 构造函数 __gm__ uint8_t *alibi_slopes → alibiSlopesGm.SetGlobalBuffer(...)
   GEN_LAUNCH 第15位 = a.alibiSlopesDevice（原是 nullptr）
   FagGeneralLaunchArgs 加 alibiSlopesDevice 字段；fag_general_host.cpp 设 gen_args.alibiSlopesDevice = alibi_slopes_ptr
```

**v2（FAGVarlenOpt）**——新增 `GM_ADDR alibiSlopes`（dv 之后）：
```
FAGVarlenOpt(..., dq, dk, dv, alibiSlopes, workspace, tiling_data, ptrDump)
   → FAGKernel::Params 加 alibiSlopes 字段+构造参；params{..., dv, alibiSlopes, workspace,...}
   → epilogue 构造传 params.alibiSlopes
   → class2 构造函数 __gm__ uint8_t *alibi_slopes → alibiSlopesGm.SetGlobalBuffer(...)
   VarlenBwdLaunchArgs 加 alibiSlopesDevice 字段；dispatch launch 传 a.alibiSlopesDevice（所有5处含dump）
   flash_api.cpp 设 vb_args.alibiSlopesDevice = alibi.ptr
```

> batchStride 仍走 tiling：两 class 的 `alibiSlopesBatchStride = tilingData->alibiSlopesBatchStride`（`FAGTilingData`/`FAGv2TilingData` 保留该字段；`fag_tiling.cpp` v2、`fag_tiling.cpp` v3 的 `GetFAGTilingParam` 拷贝）。
> `HAS_ALIBI=false` 时 `alibiSlopesDevice=nullptr`，epilogue 绑定 nullptr 但 `if constexpr` 保证不解引用，安全（与前向 `fwd_args.alibiSlopesDevice` 可为 nullptr 一致）。

### 11.5 epilogue 改造要点（运行时 → 编译期）

- 两 class 模板加 `bool HAS_ALIBI_`，`DispatchPolicy` 同步带 `<HAS_ALIBI_>`，加 `static constexpr bool HAS_ALIBI`。
- 调用点 `if (alibiEnabled) {` → `if constexpr (HAS_ALIBI) {`（class1 ~L690、class2 ~L1358）。
- **删除** `bool alibiEnabled` 成员及其 `= (tilingData->alibiSlopesAddr != 0)` 赋值。
- `alibiSlopesGm` 改在**构造函数体**从 `alibi_slopes` 参数绑定（不再从 tiling 的 `alibiSlopesAddr`）。
- `ApplyAlibiRows` 调用签名不变（已是 §10 方法2 的 `slopesGm + slopesGmOffset`）。

### 11.6 已知点

- v2 dump（`ENABLE_ASCENDC_DUMP`）路径用 `<DType>` 默认（HAS_ALIBI=false），与 softcap 参考一致；dump 为调试模式，不影响正常路径。
- 反向不启用 ALiBi 时，`ApplyAlibiRows` 整体编译消除，行为与无 ALiBi 完全一致（回归安全）。
