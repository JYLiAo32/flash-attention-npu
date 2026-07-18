# impl_alibi/design.md 评审与修正设计 (v2)

> 本版 **推翻 v1**（v1 的「统一构造 / mask 正交」「head-major 布局」两处核心论断均错误，已在用户指正下改正）。
> 依据：`impl_alibi/refs/alibi.h`（TriDao CUDA 参考）、`online_softmax.hpp` 三个 `operator()`、用户对 S 块布局的确认。
> 标注 `[需 NPU 验证]` 的项需 Ascend 工具链确认。

---

## 0. 结论先行

1. **design.md 的主体是对的**：causal（§3）与 non-causal（§7）是**两套不同实现**，分别对应 `alibi.h` 的 `Is_causal` 两分支。我 v1 攻击 §3「错误/多余」是我错，撤回。
2. **S 块布局 = BSN，head(N) 为内层**：`row = S·qNBlockSize + N`，连续行是**同一 token 的不同 head**。因此 **slope 几乎逐行变**（§3 注释「基本都会进入该分支」即此），`i_q` 只在 **token 边界**（每 qNBlockSize 行）才变。这使 §3 的「逐行 slope 缩放」、§7 的「逐行变动斜率 + 逐 token 变动位置」都成立。我 v1 推「head-major」是误读，撤回。
3. **一处必须修正**：§3 的 `Bias = [0,1,…,columnNumRound-1]`（相对列）对多 KV-tile **不正确**，应改为**绝对列** `Bias[j] = slope·(kvSStartIdx + j)`（即 `CreateVecIndex(firstValue = kvSStartIdx)`）。理由见 §3.3。
4. `CreateVecIndex` 是真实官方 API（用户确认），应采用，替换我旧实现的 `unitRamp` 斜坡技巧。

---

## 1. 基础事实（与代码 / 参考对齐）

### 1.1 alibi 的两种实现（来自 `alibi.h`，决定性依据）

| 分支 | 数学（绝对位置） | 为何不同 |
|---|---|---|
| **causal**（`alibi.h:34-52`） | 恒有 `row≥col` → `-slope·\|row-col\| = -slope·row + slope·col`。`-slope·row` 是**逐行常数**，softmax 平移不变 → **丢弃**。只剩 `+slope·col`：**只依赖列、所有行共享**。 | 不需要逐行 `\|i-j\|`；只需一个列偏置向量 |
| **non-causal**（`alibi.h:54-84`） | 不能丢，逐元素 `-slope·\|row-col\|`，`row` 需加 `(max_seqlen_k - max_seqlen_q)` 处理 Q/K 不等长 | 必须逐元素算 `\|i-j\|` |

→ **三个 `operator()` 各自对应一种 alibi**：no-mask→non-causal；causal→列偏置；SWA→随其是否带因果选其一，再叠 SWA 的 `-inf` 窗口。**不存在「一套统一构造」**。

### 1.2 S 块布局 = BSN（用户确认 + design.md §3 注释佐证）

- `rowNum = qSBlockSize·qNBlockSize`，把 BSN 的 (S,N) 融进 M 轴。
- **行序：同一 token 的多个 head 连续，再到下一 token** → `row = S·qNBlockSize + N`：
  - `token(S) = row / qNBlockSize`，`head(N) = row % qNBlockSize`；
  - **连续行 = 同一 token、不同 head** → `i_q` 不变、`slope` 变；
  - **token 边界（每 qNBlockSize 行）** → `i_q + 1`（delta=1），slope 回到 head0。
- 佐证：design.md §3 注释「由于 NHead 之间斜率不同，在 BSND 布局下，基本都会进入该分支」——只有「连续行换 head」才会逐行触发 `slope != pre_slope`，与 BSN 内层一致；若 head-major 则连续行同 head、同 slope，该分支几乎不触发，注释即错。
- `i_q = f(S)` 只依赖 token，**与 head 无关** → `\|i_q - j_k\|` 结构跨 head 共享，换 head 只换 slope 标量。这是 §3/§7 能用「slope 缩放」复用的根因。

> 我 v1 从 `CopyMaskGmToUb`（`online_softmax.hpp:478-513`）推断 head-major，与用户确认的 BSN 冲突——系我误读，作废。alibi 行→(S,N) 解码以 BSN 为准。

---

## 2. 对 design.md 的（修正后）批判

### §3 causal —— ✅ 结构正确，⚠️ 一处必须改（相对列→绝对列）
- 「缓存列 Bias、逐行 `Muls(slope/pre_slope)` + `Add`、`bias/pre_slope` 跨子任务复用」完全正确（BSN 下逐行换 head → 逐行 rescale）。
- ⚠️ **修正**：`Bias` 必须用**绝对列** `slope·(kvSStartIdx + j)`，不可用相对 `[0,1,…]`。见 §3.3。

### §7 non-causal —— ✅ 正确
- 数学用绝对位置（`j_k = kvSStartIdx + col`）✓；三情况（零点在块左/内/右）+ delta=1 分段 Adds 我已独立验算通过。
- 「变动斜率（Muls s'/s）/ 变动位置（重建或 delta Adds）/ 两者兼有（先斜率后位置）」覆盖了 BSN 下的事件序列：逐行变动斜率、逐 token 变动位置。**与 BSN 自洽**。

### §6 SWA —— ⚠️ 需补：选 alibi 风格
- SWA 不是第三种 alibi 数学；按 SWA 是否带因果，套用 causal（列偏置）或 non-causal（\|i-j\|）alibi，再叠既有 SWA `-inf` 窗口掩码。

### §1-§2 / §4 —— ✅ 正确
- alibi_slopes 形状、diff=Sk-Sq、行方向拆子任务、双缓冲、两 AIV 均分，与代码吻合。

---

## 3. 修正后的实现设计

### 3.1 调用点（已就绪，无需改）
- 无掩码：`ScaleS → ApplyAlibi(non-causal) → SubCoreCompute<false>`（`online_softmax.hpp:988-993`）。
- causal：`ScaleS → ApplyAlibi(causal) → ApplyMask → SubCoreCompute<true>`（`:1119-1127`）。
- SWA：`ScaleS → ApplyAlibi(按因果性) → (pre/next mask) → SubCoreCompute<true>`。
- 三路调用**同一份 `ApplyAlibi`**，内部按编译期 `MaskType` 选 causal / non-causal 分支。

### 3.2 状态（`BlockEpilogue` 成员，跨调用持久；fwd/bwd 各一份）
- `alibiBiasUb`：columnNumRound 个 float，= 当前行 bias。
- `alibiPreSlope`：上一行的 slope（rescale 增量用）。
- `alibiCurToken` / `alibiCurZero`（non-causal 位置状态）：当前 token 的 i_q / 零点列，用于 token 边界的 delta=1 判断。
- 每 tile（`setAlibiTileContext`）：记录 `batch/numHeads/headIdxBase/qNBlockSize/qPosBase/qOffset/kvSStartIdx`；复位状态。

### 3.3 CAUSAL 分支（列偏置 + 逐行 slope 缩放）
```
// 首次或 kv 块切换：构造绝对列结构（注意 firstValue = kvSStartIdx，不是 0！）
CreateVecIndex(alibiBiasUb, (float)kvSStartIdx, columnNumRound);   // [kvSStartIdx, +1, ...]
// （causal 无需 Abs：bias=slope·col 恒正结构，col 单调）
Muls(alibiBiasUb, alibiBiasUb, slope_head0, …);                    // 缩放到 head0 的 slope
// 逐行（BSN：每行换 head）：
for each row:
    slope_h = alibiSlopes[headIdxBase + row % qNBlockSize];
    if (slope_h != alibiPreSlope) Muls(alibiBiasUb, alibiBiasUb, slope_h / alibiPreSlope, …);
    Add(score_row, score_row, alibiBiasUb, …);
    alibiPreSlope = slope_h;
```
- **绝对列是关键修正**：`-slope·i_q`（逐行、跨所有 KV-tile 的常数）可丢；但 `slope·kvSStartIdx` 是**逐 tile** 常数（kvSStartIdx 随 KV stack 变），跨 tile 在线 softmax 归一化时不能丢，否则 `P` 错。`alibi.h` 用绝对列佐证。
- 性能：稳态每行 1 Muls + 1 Add（≈2 向量重复 × colVecs）；列结构跨行/跨 token 全程复用。

### 3.4 NON-CAUSAL 分支（逐行变动斜率 + 逐 token 变动位置，即 design.md §7）
```
for each row:
    token = row / qNBlockSize;  head = row % qNBlockSize;
    slope_h = alibiSlopes[headIdxBase + head];
    if (head == 0 && row != 0) {            // token 边界：i_q + 1（位置变动）
        // 先把 slope 归到 head0
        Muls(alibiBiasUb, alibiBiasUb, slope_h / alibiPreSlope, …);
        // 再做位置 delta=1（case1 +s / case3 -s / case2 分段 Adds）
        PositionDeltaAdds(alibiBiasUb, columnNumRound, slope_h, newZero);
    } else if (slope_h != alibiPreSlope) {  // 同 token 内换 head：只变动斜率
        Muls(alibiBiasUb, alibiBiasUb, slope_h / alibiPreSlope, …);
    } else if (首行) {                       // 全量构造
        CreateVecIndex(work, -baseCol, columnNumRound); Abs(work, …); Muls(work, -slope_h, …); 拷到 alibiBiasUb;
    }
    Add(score_row, score_row, alibiBiasUb, …);
    alibiPreSlope = slope_h;
```
- token 边界才变动位置（每 qNBlockSize 行一次），同 token 内只变动斜率 → 比 head-major（逐行变动位置）更省。

### 3.5 共享与 UB
- causal / non-causal 共用 `alibiSlopesUb`、`alibiBiasUb`、`alibiWorkUb`；按 `MaskType` 编译期分支。
- 删 `alibiUnitRampUb`（改 `CreateVecIndex` 后不需要）。
- UB 预算 `[需 NPU 验证]`：alibiBiasUb/workUb 各 ≤ columnNumRound·4B（≤2KB），需避让 S 双缓冲与 lm/hm/gm/ll/gl/dm 区。

---

## 4. 我旧 `alibi_bias.hpp` 的问题（待重写）
1. **行→(head,token) 解码用了 head-major**（`head = row/tokenNumPerHead`）——与 BSN 不符，会拿错 slope 到错行。**必须改 BSN**：`head = row % qNBlockSize, token = row / qNBlockSize`。
2. **数学一律 full `|i-j|`**：对 non-causal 正确但对 causal 没用列偏置优化；且每行从零重建、无跨行/跨 head 复用。**应拆 causal（§3）/ non-causal（§7）两分支**。
3. 应采用 `CreateVecIndex`，删 `unitRamp`。

---

## 5. 性能小结
- causal：稳态每行 ≈ Muls+Add（2 向量重复 × colVecs），列结构全程复用。
- non-causal：稳态每行 ≈ Muls+Add（同 token 内），token 边界多 1 次位置 delta。比 v1「每行 Adds+Abs+Muls+Add（4 组）」显著更优。
- 两者都在 online-softmax 关键路径（`ScaleS` 与 `CalcLocalRowMax` 之间），省下的向量算子直接缩短 kernel 时间。

---

## 6. 待 NPU 验证 / 待确认
- [ ] `Adds`/`Abs`/`CreateVecIndex`（count 形式，count>64 跨 repeat）签名与本仓库 AscendC 一致；
- [ ] **§3 绝对列修正**的正确性（多 KV-tile 跨 tile 归一化）——用 PyTorch 参考对拍 `kvSeqlen > MAX_KV_STACK_LEN` 的 causal+alibi；
- [ ] BSN 行→(S,N) 解码与现网 mask 路径一致（我误读 CopyMaskGmToUb，需在 NPU 上用打印/单测确认 head/token 归属）；
- [ ] case2 分段 Adds 的 mask/repeat 写法；UB 偏移布局；
- [ ] SWA 的 alibi 风格归属（causal-SWA vs 双向窗口）；
- [ ] bwd 端 `fag_epilogue_op.hpp` 复用同一构造（spec2 待接入）；不传 slopes 时与原路径位一致。
