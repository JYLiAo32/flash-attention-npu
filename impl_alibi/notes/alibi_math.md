# Alibi 数学与 S 块布局（实现的核心理论）

> 本文件是 alibi 实现的**理论记忆材料**：布局、两套实现（causal/non-causal）、绝对列、FP32 向量子分块、增量更新。
> 据依据：`refs/alibi.h`（TriDao CUDA 参考，决定性）、用户对布局的确认、`design.md`（用户原始设计）、`design_review.md`（修正评审）。
> 凡标 `[需 NPU 验证]` 的为 Ascend 工具链待确认项。

---

## 1. alibi 基本定义
对 query 位置 i、key 位置 j、head h（每头一个正斜率 slope_h>0）：

```
score[i,j,h] = softmax_scale · (Q·K^T)[i,j,h]  +  bias[i,j,h]
bias[i,j,h]  = -slope_h · |i - j|               # 论文公式（非因果，逐元素）
```

- bias **与 Q/K 无关**（常量加性偏置），故前向加、反向重算 P 时加；无额外梯度。
- Q/K 不等长（如 prefill、KV cache）：bottom-right 对齐，`i` 用 `i + (seqlen_k - seqlen_q)`。
- slope 形状 `[nheads]` 或 `[b, nheads]`，fp32。

---

## 2. S 块布局 = BSN（head N 为内层）★核心★

`rowNum = qSBlockSize · qNBlockSize`，把 BSN 的 (S,N) 融进 GEMM 的 M 轴。
**行序：同一 token 的多个 head 连续，再到下一个 token**（用户确认；design.md §3 注释「基本都会进入该分支」佐证——只有连续行换 head 才会逐行触发 slope 变化）。

```
row = S · qNBlockSize + N
  token(S) = row / qNBlockSize     # 查询位置 i_q 只依赖它
  head (N) = row % qNBlockSize     # slope 只依赖它
```

推论（决定增量策略）：
- **连续行 = 同一 token、不同 head** → `i_q` 不变、`slope` 变 → **逐行变动斜率**（`Muls(s'/s)`）。
- **token 边界（每 qNBlockSize 行）** → `i_q + 1`（delta=1）、slope 回到 head0 → **变动位置**（重建或 delta Adds）。
- `i_q = f(S)` 与 head 无关 → `|i_q - j_k|` 结构**跨 head 共享**，换 head 只换 slope 标量。这是「列偏置（causal）+ slope 缩放」与「non-causal 跨行复用」的根因。

> ⚠️ 撤回旧结论：先前从 `CopyMaskGmToUb` 推「head-major（N 外层）」系误读，作废。alibi 行→(S,N) 解码一律按 **BSN（N 内层）**。

---

## 3. 两套实现（来自 `refs/alibi.h`，**不存在统一构造**）

### 3.1 causal（`alibi.h:34-52`）—— 列偏置
因果掩码保证 `row ≥ col`（即 `i_q ≥ j_k`），故 `|i_q - j_k| = i_q - j_k`：
```
bias = -slope·(i_q - j_k) = -slope·i_q + slope·j_k
```
- `-slope·i_q`：**逐行常数，且跨所有 KV-tile 相同** → softmax 平移不变 → **丢弃**。
- 余 `+slope·j_k`：**只依赖列（key 绝对位置）、所有行共享**。→ 只需一个列偏置向量。
- **绝对位置**：`j_k = kvSStartIdx + col`（col 为 tile 内列号）。**必须用绝对**（见 §4）。

→ causal 实现思路（design.md §3）：构造列向量 `Bias[j] = slope·(kvSStartIdx + j)`，逐行（BSN 下每行换 head）`Muls(slope_h/pre_slope)` 缩放 + `Add` 到该行 score。`Bias`/`pre_slope` 跨子任务复用。

### 3.2 non-causal（`alibi.h:54-84`）—— 逐元素 |i-j|
不能利用平移不变性，逐元素：
```
bias[i,j,h] = -slope_h · |i_q - j_k|
  i_q = row + (seqlen_k - seqlen_q)   # 含 bottom-right 偏移
  j_k = kvSStartIdx + col             # 绝对
```
→ non-causal 实现思路（design.md §7）：
- 逐行 **变动斜率**（同 token 内）：`Muls(s'/s)`。
- 逐 token **变动位置**（delta=1）：三情况（零点在块左/内/右）+ 分段 `Adds`，或重建。
- 全量首构造：`CreateVecIndex(firstValue=-baseCol)` → `Abs`（仅零点左侧负值段）→ `Muls(-slope)` → `Add`。

---

## 4. ★绝对列修正（§3 原设计 bug，已被证明）★

design.md §3 原 `Bias = [0,1,…,columnNumRound-1]`（**相对列**，即 `slope·col = slope·(j_k - kvSStartIdx)`），会丢 `slope·kvSStartIdx`。但 `kvSStartIdx` **逐 KV-tile 变化**（`kvSStartIdx = kvSIdx · MAX_KV_STACK_LEN`）：

- `-slope·i_q` 可丢：它跨**所有** KV-tile 相同（同一 query 行）→ 整行分布平移 → softmax 不变。
- `slope·kvSStartIdx` **不可丢**：它是**逐 tile** 常数。online softmax 跨 tile 归一化时，不同 tile 的不同常数会扭曲跨 tile 权重 → 最终 `P` 错（只要 `kvSeqlen > MAX_KV_STACK_LEN`）。

**修正**：`CreateVecIndex(firstValue = kvSStartIdx)`（绝对列），与 §7/alibi.h 一致。cost 不变。

**证明**：`verify/verify_tiled_causal.py`（numpy，因果 + 分块 online softmax 对拍）：
- 单 tile：abs 与 rel 都对（bug 潜伏）。
- 多 tile（sk=32, chunk=16；GQA sk=64, 4 chunks）：abs_err ≈ 8e-8（正确），rel_err ≈ 1.0~1.2（错约 100%）。

---

## 5. FP32 向量子分块（Vector 一次只处理 64 个 fp32）

- 常量：`FLOAT_BLOCK_SIZE = 8`（对齐块），`FLOAT_VECTOR_SIZE = 64`（单条向量）。
- S 块一行列数 `columnNumRound = RoundUp(columnNum, BLOCK_SIZE_IN_BYTE)`，最多 512（`MAX_KV_STACK_LEN`）。
- 一行按 64 分段：`colVecs = CeilDiv(columnNumRound, 64)`。每条向量一条向量指令（含 `repeat = colVecs`，或循环）。
- **尾部不完整向量**：`columnNum % 64 != 0` 时，最后一条向量只有 `elems = columnNum % 64` 个有效元素，须 `SetVectorMask` 部分掩码，算完复位全开（避免越界 UB 脏数据污染 softmax）。掩码语义见 `online_softmax.hpp` `BlockEpilogue::SetVecMask` / `AlibiSetVecMask`。

### 5.1 斜坡技巧（ramp trick）—— 用一条标量加法替代逐元素下标
`j_k(col) = kvSStartIdx + col`，col 单调递增。预填 `ramp = [0,1,…,63]`（或 `CreateVecIndex`），则对一条向量：
```
(j_k - i_q) = ramp + (kvSStartIdx + base - i_q)    // base = v·64
```
一条 `Adds(ramp, scalar)` 即得整条 `(j_k - i_q)`，无需逐元素算下标。再 `Abs` → `Muls(-slope)` → `Add(score)`。把 N 个元素下标算术压成 4 条向量指令。`[需 NPU 验证]`：`Adds/Abs/Muls/Add` 的 `UnaryRepeatParams(1,1,8,8)` / `BinaryRepeatParams(1,1,1,8,8,8)` 步进语义。

### 5.2 non-causal 增量更新（design.md §7，三情况 + delta=1）
零点列 `baseCol = i_q - kvSStartIdx`（即 score 中 `|i-j|=0` 的列）。位置变动 delta=1（token 边界）时，对新偏置的更新量 `Δ(idx) = -slope·( |idx-(baseCol+1)| - |idx-baseCol| )`：
- **case1** 零点始终在块左（`baseCol ≤ 0`）：整条 `Adds(+slope)`（每元素 |i-j| 减 1 → bias 增 slope）。
- **case3** 零点始终在块右（`baseCol ≥ columnNumRound-1`）：整条 `Adds(-slope)`。
- **case2** 零点在块内（`0 < baseCol < columnNumRound-1`）：零点左侧 `Adds(+slope)`、右侧 `Adds(-slope)`，按 64 对齐分段（`num1 = baseCol'/64`，零头 x=`baseCol'%64` 用 `SetVectorMask` 处理切分向量）。
> BSN 下 delta=1 只在 token 边界（每 qNBlockSize 行）发生；同 token 内仅变动斜率。

---

## 6. 反向梯度正确性
bias 与 Q/K 无关 → `∂bias/∂Q = ∂bias/∂K = 0`。反向重算 `P = softmax(scale·S + bias)` 时**带上同一 bias** 即可，dQ/dK/dV 与 dP 公式不变（均在 biased P 处求值）。→ **反向无额外梯度项**，关键是 P 重算施加 bias，且 fwd/bwd 共用同一 alibi 构造（避免不一致）。

---

## 7. 待 NPU 验证项
- [ ] `Adds/Abs/Muls/Add/CreateVecIndex`（count 跨 repeat）签名与本仓库 AscendC 一致；
- [ ] BSN 行→(S,N) 解码与现网 mask 路径一致（我误读 CopyMaskGmToUb，NPU 上打印/单测确认 head·token 归属）；
- [ ] case2 分段 Adds 的 mask/repeat 写法；UB scratch 偏移布局；
- [ ] SWA 的 alibi 风格归属（causal-SWA 套列偏置 vs 双向窗口套 |i-j|）；
- [ ] 不传 `alibi_slopes` 时与原（无 alibi）路径位一致（运行时 no-op，无条件编译）。
