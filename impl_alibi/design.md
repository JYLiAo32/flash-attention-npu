实际情况支持
1. 参数说明
q: (batch_size, seqlen, nheads, headdim)
k: (batch_size, seqlen, nheads_k, headdim)
v: (batch_size, seqlen, nheads_k, headdim)
        
alibi_slopes: (nheads,) or (batch_size, nheads), fp32.
2. alibi_slopes: (nheads,) or (batch_size, nheads)，float32
  不同的head、batch的斜率不同
3. Q、K序列长度不一致（diff=Sk-Sq）
4. S块的形状: [batch, nheads, seqlen_q, seqlen_k]
  - GemmCoord actualBlockShapeQK{rowNum, stackSeqTile, embed};
  - 列数stackSeqTile:最多512（MAX_KV_STACK_LEN）
  - 行数rowNum= qSBlockSize * qNBlockSize; （多个head的Q矩阵在行方向拼接起来，qSBlockSize为实际行数，qNBlockSize为当前合并的head数
    - 每个Q分块对应的mask的行数是qSBlockSize，而非rowNum。
    - 根据Head进行分组，每组包含qSBlockSize个token，每组各自有一个Alibi斜率。
    - S块中的排列顺序是先sequence后head（参考Tiling 策略）
      - rowNum 最大是128，优先在sequence维度切分
      - 什么情况下会跨heads？满足两个条件
        - q_seqlen小于128，即 128 / q_seqlen > 1
        - GQA，即 groupsize = nheads_k/ nheads > 1。并且只会包含完整的Sequence。
    __aicore__ inline uint32_t GetQNBlockTile(uint32_t qSeqlen, uint32_t groupSize)
    {
        uint32_t qNBlockTile = (qSeqlen != 0) ? (Q_TILE_CEIL / qSeqlen) / N_SPLIT_HELPER * N_SPLIT_HELPER : Q_TILE_CEIL;// N_SPLIT_HELPER = 2
            //  Q_TILE_CEIL / qSeqlen 得到能够完整包含多少行，然后再将结果下取整到偶数，从而确保在为两个AIV任务切分时，确保负载均匀
        qNBlockTile = qNBlockTile < groupSize ? qNBlockTile : groupSize;
        qNBlockTile = qNBlockTile < 1 ? 1 : qNBlockTile;
        return qNBlockTile;
    }
5. 实际任务拆分方式：如何拆解S块，从GM搬运到UB，并分配到两个AIV计算？
  1. 行方向拆分子任务：类似softmax的计算，每次计算需要完整包含所有列，所以从行方向拆分形成多个子任务，两个AIV分别负责一半的子任务
uint32_t columnNumRound = RoundUp(columnNum, BLOCK_SIZE);  // BLOCK_SIZE = 16
uint32_t maxRowNumPerLoop = MAX_UB_S_ELEM_NUM / columnNumRound; 
uint32_t rowNumTile = RoundDown(maxRowNumPerLoop, FLOAT_BLOCK_SIZE);  // 每个子任务包含的Q行数
// 当前AIV负责的子任务
uint32_t rowActualThisSubBlock = (subBlockIdx == 1) ? (rowNum - rowSplitSubBlock) : rowSplitSubBlock;
uint32_t rowOffsetThisSubBlock = subBlockIdx * rowSplitSubBlock;

for (uint32_t rowLoopIdx = 0; rowLoopIdx < rowLoopNum + preLoad; rowLoopIdx++){
    ...
}
    - 拆分时需要注意一点，属于同Head的行的斜率相同，不同Head的行的斜率不同
  2. flashDecoding 模式下在task层对KVSeq划分，但是应该不影响后续做法
  3. 在online softmax阶段，S块会切分出两部分，分派到两个AIV计算
    1. 如果S块包含多个qHead（qNSplitSubBlock > 1），从Head维度切分
      1. #AIV0：（qNBlockSize / subBlockNum）* qSBlockSize 
      2. #AIV1：rowNum - #AIV0
    2. 否则：
      1. #AIV0：qSBlockSize / 2
      2. #AIV1：rowNum - #AIV0
uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
uint32_t subBlockNum = AscendC::GetSubBlockNum();  // AIV数量默认为2

uint32_t qNSplitSubBlock = qNBlockSize / subBlockNum;  // 如果S块包含多个qHead，则从Head维度切分
uint32_t rowSplitSubBlock = (qNBlockSize == 1) ?
    (qSBlockSize / 2) : (qSBlockSize * qNSplitSubBlock); // 计算AIV0包含的row数量，
uint32_t rowActualThisSubBlock = (subBlockIdx == 1) ?
    (rowNum - rowSplitSubBlock) : rowSplitSubBlock;
uint32_t rowOffsetThisSubBlock = subBlockIdx * rowSplitSubBlock;

uint32_t tokenNumPerHeadThisSubBlock = Min(qSBlockSize, rowActualThisSubBlock);  // 关注这个变量
uint32_t maskOffsetThisSubBlock = (qNBlockSize == 1) ?
    rowOffsetThisSubBlock : 0;
  4. 在每个AIV内，沿row维度进一步切分，迭代处理，每次迭代处理rowLoopNum行
这样切分会跨head吗？
uint32_t maxRowNumPerLoop = MAX_UB_S_ELEM_NUM / columnNumRound;
uint32_t rowNumTile = RoundDown(maxRowNumPerLoop, FLOAT_BLOCK_SIZE);
rowNumTile = AscendC::Std::min(rowNumTile, FLOAT_VECTOR_SIZE);
uint32_t rowLoopNum = CeilDiv(rowActualThisSubBlock, rowNumTile);

for (uint32_t rowLoopIdx = 0; rowLoopIdx < rowLoopNum + preLoad; rowLoopIdx++) {
    ... 
}


6. 在每个AIV的循环迭代中，跟踪每个rowTile的qSeqIdx偏移
TODO：待补充

> **以下 §5–§7 已被 [`notes/alibi_method_bns.md`](notes/alibi_method_bns.md) 取代**。
> 关键认知更正：S 块布局是 **BNS `[B, N_q, S_q, S_k]`（head-major）**，行解码为
> `head = absRow / qSBlockSize, token = absRow % qSBlockSize`（不是早先以为的 BSN）。
> 下面的算法骨架（causal 比值、NO_MASK case 分支）仍适用，但实现细节（每次调用重建、
> delta=1 统一切分、单 workUb in-place、无 baseColUb/持久状态）以方法文档为准。

下面的设计方案过时，需要基于BNS的布局顺序重新修正！

5. 因果掩码

> TODO: 还需要讨论是否doTriUMask，参考现有onlinesoftmax中mask的处理！

- 行方向拆分子任务，AIV均分任务数，每个任务形状为（maxRowNumPerLoop，columnNumRound）
- 参考Softmax，使用双缓冲实现子任务的Score分块GM2UB 和 计算的并行
- 使用 CreateVecIndex( **firstValue = kvSStartIdx** ，count=columnNumRound) 创建向量，记为Bias
- 记录斜率pre_slope=1（占位符）
- 对于每个子任务，for循环处理每一行
  - 根据rowIdx判断batch、head归属，然后从入参alibi_slopes中获取其斜率slope
  - 如果 slope != pre_slope

    > BNS 布局下 sequence 维连续，同一 head 内相邻行斜率相同：只在**跨 head 边界**才进入此分支（qNBlockSize==1 时整 tile 单 head，仅 1 次）。早先"BSND 布局下基本都会进入"的判断是错的。
    >

    - 使用Muls接口，将Bias向量与浮点数 slope / pre_slope 相乘，结果写回到Bias向量
  - 使用 Add 接口，将当前行与 Bias 向量相加
  - 更新pre_slope = slope
- 注意，bias和pre_slope可以跨子任务复用

6. 滑动窗口
   TODO：研究TriDao FA怎么做
7. 无掩码

- 不以位置编码为0的位置切分出两部分，而是直接计算出当前行位置编码bias（512个fp32），即 [(baseColIdx*s), ..., 3s, 2s, 1s, 0, 1s, 2s, ..., (KvSeqLen-1-qSeqOffset)*s]
- 根据rowIdx计算出当前行在Q序列的位置 qSIdx，则对应在K序列的0点编码位置为 qKvSIdx = diff + qSIdx。当前S块在K序列的范围为[kvSStartIdx,kvSEndIdx-1]位置编码分类：
  - qKvSIdx < kvSStartIdx，位置编码 bias = [kvSStartIdx - qKvSIdx, (kvSStartIdx + 1) - qKvSIdx, ..., (kvSEndIdx -1) - qKvSIdx] * (-slope)
  - kvSStartIdx <= qKvSIdx <= kvSEndIdx，位置编码 bias = [qKvSIdx - kvSStartIdx, qKvSIdx - (kvSStartIdx + 1), ... 2, 1, 0, 1, 2, ..., (kvSEndIdx - 1) - qKvSIdx] * (-slope)
  - qKvSIdx > kvSEndIdx，位置编码 bias = [qKvSIdx - kvSStartIdx, qKvSIdx - (kvSEndIdx+1), ..., qKvSIdx - (kvSEndIdx-1)] * (-slope)
- 定义：
  baseColIdx = qKvSIdx - kvSStartIdx

那么它可以取任意整数：
baseColIdx <= 0                      // 0点在Block左侧
0 < baseColIdx < columnNumRound     // 0点在Block内部
baseColIdx >= columnNumRound         // 0点在Block右侧

有：
bias[j] = -slope * abs(j - baseColIdx)， j in 0 ... N-1

- 单行内计算实现方法
  - 三种情况的统一的mask初始化做法
    1. 调用 CreateVecIndex（firstValue=-baseColIdx，count=columnNumRound)，创建 bias0 = [-baseColIdx, -baseColIdx + 1, ...,columnNumRound-baseColIdx-1]
    2. 若 baseColIdx >=0，还需要调用 Abs 接口将存在的负数转为正值，得到bias1，其中 count = min(baseColIdx, columnNumRound)
       举例子理解：假设columnNumRound = 8
       // case1: baseColIdx=-3
       CreateVecIndex(firstValue=3) // 3 4 5 6 7 8 9 10
       Abs()  // 3 4 5 6 7 8 9 10  // 实际上这步不需要

// case2: baseColIdx=3
CreateVecIndex(firstValue=-3) // -3 -2 -1 0 1 2 3 4
Abs(count=min(baseColIdx, columnNumRound)=3)  // 3 2 1 0 1 2 3 4

// case3: baseColIdx=12
CreateVecIndex(firstValue=-12) // -12 -11 -10 -9 -8 -7 -6 -5
Abs(count=min(baseColIdx, columnNumRound)=8)  // 12 11 10 9 8 7 6 5

    - 调用 Muls，将bias1 乘以斜率-s，得到bias2 = [-s*baseColIdx, -s*baseColIdx + 1, ..., -m*(columnNumRound-baseColIdx-1)]
    - 调用Add接口，将Score行与bias2相加

- 跨行更新bias2，快速复用：主要讨论如何更新Bias2向量（注意，上下文提及到Bias1、Bias2等都对应同一个内存地址）
  - 若变动斜率
    - 调用 Muls ，将Bias2 乘以 s‘ / s
  - 若变动位置
    - 做法1：（3个API调用）不复用，重新创建
    - 做法2：（1个API调用，需要分类处理）
      - 假设位置偏移为delta = qKvSIdx‘ - qKvSIdx，一般是1。默认从左往右遍历，每次最多移动1位
      - 每个位置要更新的量：Δ(i) = -slope * ( abs(i - (baseColIdx+delta)) - abs(i - baseColIdx))
      - case1：0点始终在Block左侧（baseColIdx <=0
        - 直接Adds(+delta*slope)
          (3 4 5 6 7 8 9 10)*-s -> (2 3 4 5 6 7 8 9)*-s
          (0 1 2 ...) *-s -> (-1 0 1 ...) * -s
      - Case3：0点始终在Block右侧（baseColIdx>=columnNumRound-1
        - 直接Adds(-delta*slope)
          (12 11 10 9 8 7 6 5)*-s -> (13 12 11 10 9 8 7 6)*-s
          ( ... 2 1 0) * -s -> (... 3 2 1) * -s
      - Case2：0点在Block内部（0<baseColIdx <columnNumRound-1）
        - 新0点的左侧（-delta * s)，新0点的右侧（+delta*s)
          (3 2 1 0 1 2 3) * -s -> (4 3 2 1 0 1 2) *-s
        - 分段处理，假设新0点的偏移量是baseColIdx'=baseColIdx+delta， num1 = baseColIdx' / 64
          - 对于前num1 * 64个位置，调用接口 Adds(srcoffset=0, scalarValue=-delta*s, repeat=num1），不用mask
          - 如果 x = baseColIdx' % 64 != 0，
            - 首先SetVectorMask，前x位设置为1，然后调用Adds(srcoffset=num1*64, scalarValue=-delta*s, repeat=1）
            - 首先SetVectorMask，后64-x位设置为1，然后调用 Adds(srcoffset=num1*64, scalarValue=delta*s, repeat=1）
          - 对于 (num1+1)*64 （剩下部分是对齐的）之后的位置，调用接口Adds(srcoffset=(num1+1)*64, scalarValue=delta*s, repeat=columnNumRound/64-num1-1)，不用mask
  - 若变动斜率+位置
    - 做法1：不复用，重新创建（3个API调用）
    - 做法2：先变动斜率，然后变动位置（1+1个API调用）
