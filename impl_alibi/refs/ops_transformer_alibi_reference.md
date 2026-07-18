# ops-transformer 中 ALiBi 的实现参考

> 本文档解读 `impl_alibi/ops-transformer/attention/flash_attention_score` 这套（华为 ops-transformer / aclnnFlashAttentionScore）算子里 **ALiBi** 的工程实现，作为我们为 `flash-attention-npu` 实现 ALiBi 的参考。
>
> 所有结论均来自源码，文末标注了 `文件:行号`。引用路径均相对 `impl_alibi/ops-transformer/`。

---

## 0. 一句话总结

ops-transformer **没有把 ALiBi 写成一个独立的、与掩码耦合的特例**，而是把它做成了一套通用「**位置敏感偏置 / pse (position shift embedding)**」框架的**一个特例**：

- 用户把 `alibi_slopes` 作为可选输入 `pse` 传入；
- host 侧根据 `pseType = INNER_MUL_ADD[_SQRT]` 判定「这是 ALiBi 的斜率模式」，并据此决定 `pseShapeType ∈ {PSE_B_N2_G_SLOPE, PSE_1_N2_G_SLOPE}`；
- kernel 侧在 softmax 主循环里，于 **ScaleS 之后、ApplyAttenMask 之前** 插入一段「构造偏置 + 加到 score 上」的逻辑；
- 偏置本身 = `−slope·|i−j|`，通过对**预生成的「列−行」斜坡表**做 `DMA → Cast → Adds(posShift) → Abs → Muls(−slope)` 得到。

关键点：它**不依赖** TriDao CUDA 那套「causal 下用 softmax 平移不变性把 `-slope·row` 丢掉、退化成纯列偏置」的技巧（见 [refs/alibi.h](alibi.h) 的 `Is_causal` 分支），而是**无条件走完整 `-slope·|i−j|`**（用 `Abs` 消号），因此对 causal / 非 causal / 滑动窗口都通用；causal 的下三角优化只在另一条「外部预编码表」路径里做。

---

## 1. 总体数据流与调用链

```
op_host (tiling)
  └─ 据 pse 形状/类型 → 设 hasPse、pseType、pseShapeType、pseEncodeType、pseAlibiBaseS1/S2
        （flash_attention_score_tiling_regbase.cpp:746-780）

op_kernel (arch22 / arch35)
  init 阶段:
  ├─ PseInnerAlibiCreate()   一次性在 GM 工作区生成「col−row」斜坡表 pseAlibiGm   (pse.h:474, s1s2_bn2gs1.h:358)
  └─ pseSlope = pse;         记下斜率 GM 指针                                       (s1s2_bn2gs1.h:397)

  主循环（每个 Q×KV tile）:
  ├─ CopyInAttenMask(...)                    搬 attention mask                       (s1s2_bn2gs1.h:1307)
  ├─ if pseType != OUTER_ADD_MUL:  Muls(scale)   ← ScaleS                          (s1s2_bn2gs1.h:1308-1312)
  ├─ if hasPse:
  │     ├─ 填充 pseInfo（当前 tile 的 s1oIdx/s2StartIdx/boIdx/n2oIdx/goIdx ...）   (s1s2_bn2gs1.h:1314-1329)
  │     ├─ pseType∈{INNER_MUL_ADD,INNER_MUL_ADD_SQRT} → PseSlopeCopyIn()           (s1s2_bn2gs1.h:1336-1343)
  │     │     否则（外部偏置表）                       → PseCopyIn()                (s1s2_bn2gs1.h:1344-1352)
  │     └─ PseCompute()   把偏置 Add 到（已 scale 的）score 上                       (s1s2_bn2gs1.h:1354)
  ├─ if pseType == OUTER_ADD_MUL:  Muls(scale)                                    (s1s2_bn2gs1.h:1356-1360)
  └─ if hasAtten:  SelectWithBytesMask(...)   ← ApplyAttenMask（-inf 掩码）         (s1s2_bn2gs1.h:1361+)
```

**插入顺序 = ScaleS → (构造并叠加 ALiBi 偏置) → ApplyAttenMask。** 这正是 ALiBi 应当出现的位置：偏置是实数加性项，必须先于 softmax 的行最大值统计加上去；而 attention mask（−inf）在它之后施加，掩码位置被压成 −inf，与实数偏置互不干扰。

> pseType 同时控制「scale 与 bias 的先后」：
> - `INNER_MUL_ADD`（ALiBi 用）= 先 `*scale` 再 `+bias`，即 `score = QKᵀ·scale + alibi`；
> - `OUTER_ADD_MUL` = 先 `+bias` 再 `*scale`；
> - `OUTER_MUL_ADD`（默认）= 先 `*scale` 再 `+bias`。
> ALiBi 取 INNER，保证 `softmax(scale·QKᵀ + bias)` 的标准语义。

---

## 2. 核心数据结构

### 2.1 `PseInfo`（每 tile 的上下文，pse.h:42-79）

它是把「当前 tile 在 (B,N,G,S1,S2) 中的坐标」打包成的一个结构体，等价于我们 `setAlibiTileContext` 要传递的那批参数。ALiBi 相关字段：

| 字段 | 含义 |
|---|---|
| `boIdx / n2oIdx / goIdx` | 当前 batch / head-group / group 内偏移，用于在**斜率表**里定位 `slope` |
| `s1oIdx, s1BaseSize, vec1S1BaseSize, loopIdx, vecCoreOffset` | 拼「当前 tile 的 Q 行全局起点 `s1Offset`」 |
| `s2StartIdx, s2LoopCount, s2BaseNratioSize` | 拼「当前 tile 的 KV 列全局起点 `s2Offset`」 |
| `qStartIdx, kvStartIdx` | Q/K 序列在全局（含 padding/前缀）中的起点，用于 `posShift` |
| `pseShapeType` | `pseSlopeBn=2` / `pseSlopeN=3`（ALiBi 斜率表形状，见 §3） |
| `pseType` | `INNER_MUL_ADD[_SQRT]` 触发 ALiBi 路径 |
| `pseEncodeType` | `pseEncodeALibiS2Full=0x11` 触发「外部下三角预编码表」路径 |
| `pseAlibiBaseS1, pseAlibiBaseS2` | 核内生成的斜坡表尺寸（行数×列数） |
| `vec1S1RealSize, s2RealSize, s2AlignedSize, pseS2ComputeSize, readS2Size` | 当前 tile 实际要算的行列数与对齐量 |

### 2.2 `PseTypeEnum`（pse.h:34-40）

```cpp
PSE_OUTER_MUL_ADD_TYPE = 0,   // 默认：外层先 mul 再 add
PSE_OUTER_ADD_MUL_TYPE,       // 外层先 add 再 mul
PSE_INNER_MUL_ADD_TYPE,       // ALiBi：内层（已 scale 之后）add 偏置
PSE_INNER_MUL_ADD_SQRT_TYPE,  // ALiBi 的 sqrt 变体
```

### 2.3 ALiBi 斜率表形状（pse.h:27-30）

```cpp
pseSlopeBn = 2;   // PSE_B_N2_G_SLOPE：形状 (B, N2)，每个 batch/head 一个 slope
pseSlopeN  = 3;   // PSE_1_N2_G_SLOPE：形状 (N2,)，跨 batch 共享 slope
```

host 侧据此判定（`tiling_regbase.cpp:753-766`）：当 `pseType` 为 INNER 且 `pse` 输入是 2D → `PSE_B_N2_G_SLOPE`；1D → `PSE_1_N2_G_SLOPE`，且要求维度**只能是 (B,N2) 或 (N2,)**，否则报错。这正好覆盖了「`alibi_slopes: (nheads,) 或 (batch_size, nheads)`」的两种用法。

---

## 3. 三条实现路径

`PseCopyIn` / `PseCompute` 是分发入口，内部按 `pseType` 与 `pseEncodeType` 走三条路：

### 路径 A：核内斜率模式（ALiBi 主路径）

触发：`pseType ∈ {INNER_MUL_ADD, INNER_MUL_ADD_SQRT}`。
分两步——**init 时生成基础表** + **每 tile 做位置对齐与乘斜率**。

**(A1) 一次性生成「col−row」斜坡表** —— `PseInnerAlibiCreate` (pse.h:474-498)

```cpp
float tmpValue = -1.0;
for (int64_t i = 0; i < pseInfo.pseAlibiBaseS1; i++) {
    CreateVecIndex(helpTensor, (half)(i * tmpValue), pseInfo.pseAlibiBaseS2);
    // 第 i 行 = [-i, -i+1, -i+2, ..., -i+(S2-1)]  ==  (col - row)
    DataCopy(dstTensor[i * pseAlibiBaseS2], helpTensor, pseInfo.pseAlibiBaseS2);  // 写回 GM
}
```

- 用 **`CreateVecIndex(tensor, firstValue, count)`** 一条指令生成等差向量（首项 `firstValue`、步长 1）。第 `i` 行首项 = `−i`，于是整行就是 `(col − row)`。
- 整张表写到 GM 工作区 `pseAlibiGm`（`half` 精度，省带宽），尺寸 `pseAlibiBaseS1 × pseAlibiBaseS2`，按 512B 对齐分配（`s1s2_bn2gs1.h:438-441, 459-467`），每个 block 一份。
- **关键结论**：表里存的是**带符号的 `(j − i)`**（不是 `|j−i|`，也不是最终偏置），不含 slope。消号（Abs）和乘斜率放到每 tile 做。

**(A2) 每 tile 构造偏置并叠加** —— `PseSlopeCopyIn` (pse.h:326-370)

```cpp
// 1) 从 GM 表搬当前 tile 的一块 [vec1S1RealSize × s2RealSize] 进 UB（helpTensor）
DataCopyIn(helpTensor, alibiGm, 0, vec1S1RealSize, s2RealSize, pseAlibiBaseS2, ...);
// 2) Cast 到计算精度（通常 fp32）
Cast(dstTensor, helpTensor, ...);

// 3) 位置对齐 + 取绝对值
int64_t s1Offset = s1oIdx*s1BaseSize + vecCoreOffset + loopIdx*vec1S1BaseSize;   // Q 全局行起点
int64_t s2Offset = s2StartIdx + s2LoopCount*s2BaseNratioSize;                   // KV 全局列起点
float posShift = float(s2Offset + kvStartIdx - s1Offset - qStartIdx);           // ★核心
Adds(dstTensor, dstTensor, posShift, computeSize);   // (col−row) + posShift
Abs (dstTensor, dstTensor, computeSize);             // |global_j − global_i|
if (SQRT 变体) Sqrt(dstTensor, dstTensor, computeSize);

// 4) 乘以 -slope（每个 head 一个标量，从 GM 读）
float slopes = ((__gm__ T *)pseSlope)[offset] * -1;  // offset = bOffset + n2oIdx*gSize + goIdx
Muls(dstTensor, dstTensor, slopes, computeSize);
```

随后 `PseCompute` → `PseBroadcastAdd` → `Add(score, score, pseUb)` 把这块偏置加到 score 上（pse.h:451 / 179-180）。

**`posShift` 的含义**（pse.h:355, 393）：
```
posShift = (s2Offset + kvStartIdx) − (s1Offset + qStartIdx)
         =  K 序列全局起点 − Q 序列全局起点
```
- 表里 `dst[row][col] = col − row` 是「局部坐标」下的 `(j−i)`。
- 加 `posShift` 后变成「全局坐标」下的 `(global_j − global_i)`，`Abs` 即得 `|global_j − global_i|`。
- 这正是我们 design.md §7 里的 `qOffset = seqlen_k − seqlen_q` / `baseColIdx` 概念，**用「一条标量 Adds」就把局部斜坡搬到了全局位置**——非常干净，值得照搬。

> 注：`posShift` 是整 tile 共享的标量，所以它隐含「该 tile 内所有行的 `i_q` 用同一个基准」——与 §4 的「按行细究」并不矛盾，因为 `col−row` 表已经把行间差异编码进去了，`posShift` 只补一个 tile 级常数。

### 路径 B：外部「下三角预编码表」模式

触发：`pseEncodeType == pseEncodeALibiS2Full (0x11)`。表 `pseGm` 由外部提供，内容是**已经乘好 slope 的完整 `−slope·|i−j|`**（下三角）。

- `NeedPseAlibiCompute` (pse.h:278-291)：**「ALiBi 编码只计算下三角」**——若整块 Q 行完全在 causal 对角线之上（`q_end ≤ kv_start`），直接 `return false` 跳过本 tile（这些位置反正会被 mask 成 −inf）。
- `PseAlibiComputeOffset` (pse.h:230-275)：在预编码表里定位当前 tile 的偏移，**显式处理 causal 对角线**（`threshold = s1Size − pseS1Size`，分 `row≥threshold` 与回绕两支；TND 布局另有一套 `posVal` 计算）。并据此算出本 tile 实际要读的列数 `readS2Size`、对齐后的 `pseS2ComputeSize`。
- `PseAlibiCopyIn` (pse.h:294-323)：按上面算出的 offset/size 把表搬进 UB（必要时 Cast）。
- `PseAlibiCompute` (pse.h:445-454)：`Add(score, score, pseTensor)`，仅做加法（表里已是最终值，不再 Abs/Muls）。

这条路的**全部「causal 智能」都在 host 预编码 + offset 计算 + 下三角跳过**里；kernel 内只搬+加。

### 路径 C：通用外部偏置（S1S2 / 1S2）

`pseEncodeType != 0x11` 且非斜率模式：走 `PseComputeOffset` + `PseCopyIn` + `PseCompute→PseBroadcastAdd`。
- `pseShapeType ∈ {pseS1S2, pseSlopeBn, pseSlopeN}`：偏置已是 `[s1,s2]` 全尺寸，直接逐元素 `Add`。
- 否则（如 `pse1S2`，偏置是 `[1,s2]`）：用 `BroadcastAdd` 把一行偏置广播加到多行 score 上（pse.h:142-196），利用 `src1RepStride=0` 做行广播。

ALiBi 一般不走这条（除非用户直接喂了一张预算好的偏置表）。

---

## 4. 与掩码/布局的关系（回应「不同掩码下实现是否不同」）

这是与我们此前争论直接相关的点，源码给的答案是**分层的**：

1. **ALiBi 偏置本身（路径 A）对掩码无关**：它无条件算 `−slope·|i−j|`，causal / 非 causal / SWA 都走同一段 `Adds→Abs→Muls`。这是 ops-transformer 的选择——**用 Abs 的通用性换取「不必为每种掩码写一套偏置逻辑」**。
   - 代价：相比 TriDog CUDA 的 causal 技巧（`−slope·|i−j| = −slope·row + slope·col`，丢掉行常数 `-slope·row`，退化成纯列偏置 `slope·col`，可省掉 Abs、省掉逐行），ops-transformer 的 causal 没有这层简化，**多了 Abs + 斜坡表**。
   - 收益：实现统一、不易错；且非 causal/SWA 天然支持。

2. **掩码的差异体现在「attention mask 那一段」**（`SelectWithBytesMask`，s1s2_bn2gs1.h:1361+），由 host 根据 `sparseMode`（`NO_MASK / LEFT_UP_CAUSAL / RIGHT_DOWN_CAUSAL / BAND / ...`）决定，与 ALiBi 偏置段解耦。

3. **唯一与掩码耦合的 ALiBi 优化在路径 B**（外部下三角预编码表）：`NeedPseAlibiCompute` 的「只算下三角」和 `PseAlibiComputeOffset` 的对角线处理，是专门给 causal 用的。路径 A 不走这套。

4. **布局（BNSD vs TND）**：`PseAlibiComputeOffset` 对 `LAYOUT_TND` 单独写了一套 offset（pse.h:250-265），处理变长序列；kernel 里还有个 `innerAlibiFlag`（s1s2_bn2gs1.h:1330-1342）：当 `LAYOUT_TND + BAND_LEFT_UP_CAUSAL + boIdx!=0` 时把 `kvStartIdx/qStartIdx` 置 0（因为 TND 下后续 batch 的绝对位置基准不同，需要重新对齐）。

> **对我们设计的启示**：我们之前的争论可以调和了——
> - 「causal 用平移不变性退化成列偏置」（design.md §3 / refs/alibi.h）是**数学上更优**的 causal 特化，省 Abs、省表；
> - 「统一用 Abs 算 `−slope·|i−j|`」（ops-transformer 路径 A）是**工程上更通用**的做法。
> 两者并不矛盾，是**同一目标的两条路**。我们要做的选择是：causal 路径是否值得为性能做 §3 的特化，还是统一走 Abs。ops-transformer 选择了后者（更稳）。

---

## 5. 性能与资源特征

| 维度 | ops-transformer 路径 A 的做法 | 备注 |
|---|---|---|
| GM 表 | `pseAlibiBaseS1 × pseAlibiBaseS2 × sizeof(half)`，每 block 一份，512B 对齐 | half 省一半带宽；表是 `(col−row)`，不含 slope |
| 表生成 | init 一次：`pseAlibiBaseS1` 次 `CreateVecIndex` + `DataCopy`（V→MTE3 流水） | 用了 V_MTE3/MTE3_V/MTE3_S 三种 flag 严格同步 |
| 每 tile 向量算 | `Cast` + `Adds(标量)` + `Abs` + 可选 `Sqrt` + `Muls(标量)`，各 1 次（覆盖整 tile） | 没有逐行循环，整 `[s1×s2]` 块一次性算 |
| slope 读取 | 每 tile 1 次 GM 标量读 `pseSlope[offset]` | 不是逐行读，开销可接受 |
| 位置对齐 | 1 个标量 `posShift` 的 `Adds` | 比 design.md §7 的「分三段 case1/2/3」简单得多 |
| UB 占用 | `helpTensor`（搬表）+ `pseUb`（结果）+ `commonTBuf` | 走的是「搬整块表」而非「逐行增量」 |

**与我们 design.md 的对比**：
- design.md §7（非 causal）的「逐行分三段 case1/2/3、按 delta=±1 增量更新斜坡」**更省 UB、不占 GM 表**，但实现复杂、易错。
- ops-transformer 路径 A **更简单、更通用**（一个 `posShift` 标量 + Abs 搞定所有行），代价是要在 GM 里存一张斜坡表 + init 时生成它。
- 二者**位置对齐的内核是一致的**：都是「把局部 `(col−row)` 斜坡用一个标量平移到全局坐标，再 Abs」。ops-transformer 用 `posShift`，design.md 用 `baseColIdx`，本质相同。

---

## 6. 对我们实现的借鉴清单（建议照搬 / 可选）

**建议照搬：**
1. **`posShift` 标量平移 + `Abs`** 的位置对齐写法（pse.h:355-359）——比 design.md §7 的三段分段更简洁，且对 causal/非 causal 统一。`posShift = (KV 全局起点) − (Q 全局起点)`。
2. **每 head 一个 `slope` 标量、每 tile 读 1 次 GM**（pse.h:361）——避免逐行 GM 标量读；可进一步把本 tile 涉及的若干 head 的 slope 预 DMA 进 UB（ops-transformer 没做这步优化，我们可以加）。
3. **`PseInfo` 式的「tile 上下文打包」**——对应我们的 `setAlibiTileContext`，字段定义可参考 pse.h:42-79。
4. **插入点 = ScaleS 之后、ApplyAttenMask 之前**（s1s2_bn2gs1.h:1308-1361）——与我们既定方案一致，这里得到工业级佐证。
5. **`CreateVecIndex` 是 AscendC 真实接口**（`kernel_vec_intf.h`，`ASC_DEVKIT_MAJOR>=9`，pse.h:19-24）。我们若 devkit 版本够新可直接用；否则用「预填 `unitRampUb` + `Adds(标量)`」等价实现（我们 alibi_bias.hpp 已是这种）。

**可选（按需权衡）：**
6. **是否预生成 GM 斜坡表**（路径 A 的 `pseAlibiGm`）：表换带宽。若 GM 工作区紧张，可改用 design.md §7 的「逐行 UB 内增量」。
7. **causal 是否做 §3 平移不变性特化**：若 causal 是热路径且追求极致性能，值得做（省 Abs、省表）；否则统一走 Abs 更稳。ops-transformer 选了「不特化」。
8. **下三角跳过**（`NeedPseAlibiCompute`）：若我们走「外部预编码表」或 causal 且想省算力，可借鉴；若统一走 Abs 则不需要。

**需要核验（我们没有 NPU 工具，标注 `[需 NPU 验证]`）：**
- `CreateVecIndex` 在我们 devkit 版本是否存在/签名一致；
- `Adds/Abs/Muls` 在 fp32 下的 `RepeatParams`（块步进/重复步进）是否与 pse.h 用法一致（pse.h 这里用的是简化重载 `Adds(dst,src,scalar,count)`，我们 alibi_bias.hpp 用的是显式 `UnaryRepeatParams`，需统一）；
- GM 工作区偏移、UB scratch 偏移不与现有布局冲突。

---

## 7. 关键源码索引

| 内容 | 位置 |
|---|---|
| pse 框架总头（所有 ALiBi 函数） | `common/include/op_kernel/pse.h` |
| `PseInfo` 结构体 | `pse.h:42-79` |
| `PseTypeEnum` / 斜率形状常量 | `pse.h:27-40` |
| 核内斜坡表生成 `PseInnerAlibiCreate` | `pse.h:474-498` |
| 每 tile 偏置构造 `PseSlopeCopyIn`（含 `posShift`） | `pse.h:326-370` |
| `posShift` 公式 | `pse.h:355, 393` |
| 下三角跳过 `NeedPseAlibiCompute` | `pse.h:278-291` |
| causal offset 计算 `PseAlibiComputeOffset` | `pse.h:230-275` |
| 外部表搬入 `PseAlibiCopyIn` / 叠加 `PseAlibiCompute` | `pse.h:294-323, 444-454` |
| 广播加 `PseBroadcastAdd` / `BroadcastAdd` | `pse.h:142-196` |
| 主循环调用链（ScaleS→Pse→Mask） | `op_kernel/arch22/flash_attention_score_s1s2_bn2gs1.h:1307-1369` |
| init 生成表 / `pseSlope` / `pseAlibiGm` 布局 | `s1s2_bn2gs1.h:358, 397, 438-467` |
| `innerAlibiFlag`（TND+BAND_LEFT_UP_CAUSAL） | `s1s2_bn2gs1.h:1330-1342` |
| host：pseType/pseShapeType 判定 | `op_host/arch35/flash_attention_score_tiling_regbase.cpp:746-780` |
| host：pseAlibiBaseS1/S2 默认 0 | `op_host/arch22/flash_attention_score_tiling_general.cpp:1009-1010` |
| TriDao CUDA ALiBi（causal 平移不变性对照） | `refs/alibi.h` |
