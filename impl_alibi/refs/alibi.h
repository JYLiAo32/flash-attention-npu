// csrc/flash_attn/src/alibi.h
template <bool Is_causal>
struct Alibi {

    // 当前 head 的 ALiBi slope，以及 Q/K 的最大序列长度
    // （用于非 causal 场景计算全局位置偏移）。
    const float alibi_slope;
    const int max_seqlen_k, max_seqlen_q;

    __forceinline__ __device__ Alibi(const float alibi_slope, const int max_seqlen_k, const int max_seqlen_q)
        : alibi_slope(alibi_slope)
        , max_seqlen_k(max_seqlen_k)
        , max_seqlen_q(max_seqlen_q) {
    };

    // 为当前 Score Tile 添加 ALiBi Bias。
    //
    // tensor          : 当前线程持有的 Score Fragment（QK^T Tile）
    // col_idx_offset_ : 当前 Tile 在 K 方向的全局起始列
    // row_idx_offset  : 当前 Tile 在 Q 方向的全局起始行
    // warp_row_stride : Warp 内相邻 MMA Row 的跨度
    template <typename Engine, typename Layout>
    __forceinline__ __device__ void apply_alibi(Tensor<Engine, Layout> &tensor,
                     const int col_idx_offset_,
                     const int row_idx_offset,
                     const int warp_row_stride) {

        static_assert(Layout::rank == 2, "Only support 2D Tensor");

        // 每个线程负责两列，因此根据 lane_id 得到当前线程对应的列起始位置。
        const int lane_id = threadIdx.x % 32;
        const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;

        if constexpr (Is_causal) {
            // Causal Attention:
            // 原始 ALiBi 为 -m(row-col)，由于 row>=col，可展开为
            //     -m*row + m*col
            // 其中 -m*row 为整行常数，对 Softmax 无影响，因此仅保留 m*col，
            // 所有行共享同一组列 Bias。
            #pragma unroll
            for (int nj = 0; nj < size<1,1>(tensor); ++nj) {
                const int col_idx_base = col_idx_offset + nj * 8;
                #pragma unroll
                for (int j = 0; j < size<1,0>(tensor); ++j) {
                    const int col_idx = col_idx_base + j;
                    #pragma unroll
                    for (int mi = 0; mi < size<0>(tensor); ++mi) {
                        tensor(mi, make_coord(j, nj))
                            += alibi_slope * col_idx;
                    }
                }
            }

        } else {
            // Non-causal Attention:
            // 无法利用 Softmax 平移不变性，需要按论文公式
            //     -m * |row - col|
            // 逐元素计算 Bias。
            // row_idx 需加上 (max_seqlen_k - max_seqlen_q)，
            // 用于支持 KV Cache 等 Q/K 长度不一致的情况。
            #pragma unroll
            for (int mi = 0; mi < size<0,1>(tensor); ++mi) {
                const int row_idx_base = row_idx_offset + mi * warp_row_stride;

                #pragma unroll
                for (int i = 0; i < size<0,0>(tensor); ++i) {
                    const int row_idx = row_idx_base + i * 8;

                    #pragma unroll
                    for (int nj = 0; nj < size<1,1>(tensor); ++nj) {
                        const int col_idx_base = col_idx_offset + nj * 8;

                        #pragma unroll
                        for (int j = 0; j < size<1,0>(tensor); ++j) {
                            const int col_idx = col_idx_base + j;

                            tensor(make_coord(i, mi), make_coord(j, nj))
                                -= alibi_slope *
                                   abs(row_idx + max_seqlen_k - max_seqlen_q - col_idx);
                        }
                    }
                }
            }
        }
    }
};