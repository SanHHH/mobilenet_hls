/*
 * mobilenet_kernels.cpp -- Combined With Buffer (DWConv only)
 *
 * 8x8 output tiling, per-tile input buffering with stride-aware halo,
 * per-channel 3x3 weight buffering, KJ unrolling factor 3 and KI pipelining.
 * Input buffer columns are cyclically banked by 3 and weights partitioned
 * to allow parallel local reads. Each tile is loaded once from the m_axi
 * input, then reused for its DWConv outputs. Other kernels are unchanged.
 *
 * LOOP_TRIPCOUNT only tells the HLS report how many iterations to assume for
 * variable-bound loops (it does not change the hardware).
 */
#include "mobilenet.h"

/* ------------------------------------------------------------------------
 * Standard 3x3 convolution (only used by layer Conv2d_0: 3 -> 16 channels,
 * 160x160 -> 80x80)
 *   in  [N][H][H]   out [M][Ho][Ho]   w [M][N][3][3]   b [M]
 * ------------------------------------------------------------------------ */
void conv3x3(const data_t *in, data_t *out, const wgt_t *w, const bias_t *b,
             int N, int M, int H, int S, const int8_t *shift)
{
    const int Ho  = out_size(H, S);
    const int pad = same_pad_begin(H, S, 3);

    // Local partitioned register/LUTRAM buffers for 16 output channels (M = 16)
    // Complete partitioning allows all 16 accumulators to update in 1 cycle!
    acc_t acc_buf[16];
#pragma HLS ARRAY_PARTITION variable=acc_buf complete dim=1

    wgt_t w_local[16][3][3][3];
#pragma HLS ARRAY_PARTITION variable=w_local complete dim=1

    // 1. Pre-load all 16 x 3 x 3 x 3 = 432 weights from URAM into registers once
    LOAD_W_TO: for (int to = 0; to < M; to++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
        LOAD_W_TI: for (int ti = 0; ti < N; ti++) {
#pragma HLS LOOP_TRIPCOUNT min=3 max=3
            LOAD_W_KI: for (int i = 0; i < 3; i++) {
                LOAD_W_KJ: for (int j = 0; j < 3; j++) {
#pragma HLS PIPELINE II=1
                    w_local[to][ti][i][j] = w[((to * N + ti) * 3 + i) * 3 + j];
                }
            }
        }
    }

ROW: for (int row = 0; row < Ho; row++) {
#pragma HLS LOOP_TRIPCOUNT min=80 max=80
    COL: for (int col = 0; col < Ho; col++) {
#pragma HLS LOOP_TRIPCOUNT min=80 max=80

        // 2. Initialize all 16 output accumulators with biases in 1 cycle
        INIT_ACC: for (int to = 0; to < 16; to++) {
#pragma HLS UNROLL
            if (to < M) acc_buf[to] = b[to];
        }

        // 3. LOOP INTERCHANGE: Dependent reduction loops (TI, KI, KJ) OUTSIDE,
        //    independent output-channel loop (TO) INSIDE with full UNROLL!
        TI: for (int ti = 0; ti < N; ti++) {
#pragma HLS LOOP_TRIPCOUNT min=3 max=3
            KI: for (int i = 0; i < 3; i++) {
                KJ: for (int j = 0; j < 3; j++) {
#pragma HLS PIPELINE II=1
                    int y = S * row + i - pad;
                    int x = S * col + j - pad;
                    if (y >= 0 && y < H && x >= 0 && x < H) {
                        // Read DDR input pixel ONCE and broadcast to all 16 filters!
                        const acc_t in_val = (acc_t)in[(ti * H + y) * H + x];

                        TO: for (int to = 0; to < 16; to++) {
#pragma HLS UNROLL
                            if (to < M) {
                                acc_buf[to] += in_val * (acc_t)w_local[to][ti][i][j];
                            }
                        }
                    }
                }
            }
        }

        // 4. Requantize & write 16 output channels to URAM
        STORE_OUT: for (int to = 0; to < M; to++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
#pragma HLS PIPELINE II=1
            out[(to * Ho + row) * Ho + col] = requant(acc_buf[to], shift[to], 1);
        }
    }
}
}


/* ------------------------------------------------------------------------
 * Depthwise 3x3 convolution: channel c of the output only looks at channel c
 * of the input (slim.separable_conv2d with num_outputs=None).
 *   in [C][H][H]   out [C][Ho][Ho]   w [C][3][3]   b [C]
 * ------------------------------------------------------------------------ */
void dwconv3x3(const data_t *in, data_t *out, const wgt_t *w, const bias_t *b,
               int C, int H, int S, const int8_t *shift)
{
    const int Ho  = out_size(H, S);
    const int pad = same_pad_begin(H, S, 3);
    const int TILE_H = 8;
    const int TILE_W = 8;
    const int MAX_PATCH = 17;

    // Partition dim=2 completely (17 cols -> LUTRAM/registers) so 4 parallel
    // horizontal reads (lx + 0..3) never suffer bank conflicts!
    data_t input_buffer[MAX_PATCH][MAX_PATCH];
#pragma HLS ARRAY_PARTITION variable=input_buffer complete dim=2

    wgt_t weight_buffer[3][3];
#pragma HLS ARRAY_PARTITION variable=weight_buffer complete dim=0

    // Local 8x8 tile accumulators partitioned by factor=4 for 4 MACs/cycle at II=1
    acc_t acc_tile[TILE_H][TILE_W];
#pragma HLS ARRAY_PARTITION variable=acc_tile cyclic factor=4 dim=2

CH: for (int c = 0; c < C; c++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=512
    const bias_t bias_c   = b[c];
    const int8_t shift_c  = shift[c];
    const int    w_c_off  = c * 9;
    const int    in_c_off = c * H * H;
    const int    out_c_off = c * Ho * Ho;

    // 1. Load 3x3 depthwise weights into registers (pipelined II=1)
    LOAD_WEIGHT_ROW: for (int i = 0; i < 3; i++) {
        LOAD_WEIGHT_COL: for (int j = 0; j < 3; j++) {
#pragma HLS PIPELINE II=1
            weight_buffer[i][j] = w[w_c_off + i * 3 + j];
        }
    }

    TILE_ROW: for (int rb = 0; rb < Ho; rb += TILE_H) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=10
        TILE_COL: for (int cb = 0; cb < Ho; cb += TILE_W) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=10
            const int patch_h = (TILE_H - 1) * S + 3;
            const int patch_w = (TILE_W - 1) * S + 3;

            // 2. Load input patch + halo into local buffer (pipelined II=1)
            LOAD_INPUT_ROW: for (int py = 0; py < patch_h; py++) {
#pragma HLS LOOP_TRIPCOUNT min=10 max=17
                LOAD_INPUT_COL: for (int px = 0; px < patch_w; px++) {
#pragma HLS LOOP_TRIPCOUNT min=10 max=17
#pragma HLS PIPELINE II=1
                    const int y = rb * S + py - pad;
                    const int x = cb * S + px - pad;
                    input_buffer[py][px] =
                        (y >= 0 && y < H && x >= 0 && x < H)
                        ? in[in_c_off + y * H + x] : (data_t)0;
                }
            }

            // 3. Initialize 8x8 tile accumulators with channel bias (4/cycle at II=1)
            INIT_Y: for (int ty = 0; ty < TILE_H; ty++) {
                INIT_X: for (int tx = 0; tx < TILE_W; tx++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
                    acc_tile[ty][tx] = bias_c;
                }
            }

            // 4. LOOP INTERCHANGE: Dependent 3x3 kernel reduction (KI, KJ) OUTSIDE,
            //    independent 8x8 spatial tile (TILE_Y, TILE_X) INSIDE with UNROLL=4!
            KI: for (int i = 0; i < 3; i++) {
                KJ: for (int j = 0; j < 3; j++) {
                    // Read weight ONCE and broadcast across all 64 tile pixels!
                    const acc_t w_val = (acc_t)weight_buffer[i][j];

                    TILE_Y: for (int ty = 0; ty < TILE_H; ty++) {
                        TILE_X: for (int tx = 0; tx < TILE_W; tx++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
                            const int ly = ty * S + i;
                            const int lx = tx * S + j;
                            acc_tile[ty][tx] += (acc_t)input_buffer[ly][lx] * w_val;
                        }
                    }
                }
            }

            // 5. Requantize & store valid in-bounds tile pixels (pipelined II=1)
            STORE_Y: for (int ty = 0; ty < TILE_H; ty++) {
                const int row = rb + ty;
                STORE_X: for (int tx = 0; tx < TILE_W; tx++) {
#pragma HLS PIPELINE II=1
                    const int col = cb + tx;
                    if (row < Ho && col < Ho) {
                        out[out_c_off + row * Ho + col] =
                            requant(acc_tile[ty][tx], shift_c, 1);
                    }
                }
            }
        }
    }
}
}

/* ------------------------------------------------------------------------
 * Pointwise 1x1 convolution: at every pixel, out[to] = sum_ti w[to][ti]*in[ti]
 *   in [N][H][H]   out [M][H][H]   w [M][N]   b [M]
 * ------------------------------------------------------------------------ */
// In mobilenet_kernels.cpp:
void conv1x1(const data_t *in, data_t *out, const wgt_t *w, const bias_t *b,
             int N, int M, int H, const int8_t *shift)
{
    const int HW = H * H;
#pragma HLS DEPENDENCE variable=w inter false

    data_t in_pix_buf[512];
#pragma HLS BIND_STORAGE variable=in_pix_buf type=ram_2p impl=bram

    acc_t acc_buf[512];
#pragma HLS BIND_STORAGE variable=acc_buf type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=acc_buf cyclic factor=4 dim=1
#pragma HLS DEPENDENCE variable=acc_buf inter false

ROW: for (int row = 0; row < H; row++) {
#pragma HLS LOOP_TRIPCOUNT min=5 max=80
    COL: for (int col = 0; col < H; col++) {
#pragma HLS LOOP_TRIPCOUNT min=5 max=80
        const int pix_off = row * H + col;

        LOAD_PIX: for (int ti = 0; ti < N; ti++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=512
#pragma HLS PIPELINE II=1
            in_pix_buf[ti] = in[ti * HW + pix_off];
        }

        INIT_ACC: for (int to = 0; to < M; to++) {
#pragma HLS LOOP_TRIPCOUNT min=32 max=512
#pragma HLS PIPELINE II=1
            acc_buf[to] = b[to];
        }

        TI: for (int ti = 0; ti < N; ti++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=512
            const acc_t in_val = (acc_t)in_pix_buf[ti];

            TO: for (int to = 0; to < M; to++) {
#pragma HLS LOOP_TRIPCOUNT min=32 max=512
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=2
                acc_buf[to] += in_val * (acc_t)w[to * N + ti];
            }
        }

        STORE_OUT: for (int to = 0; to < M; to++) {
#pragma HLS LOOP_TRIPCOUNT min=32 max=512
#pragma HLS PIPELINE II=1
            out[to * HW + pix_off] = requant(acc_buf[to], shift[to], 1);
        }
    }
}
}

/* ------------------------------------------------------------------------
 * Global average pool: [C][H][H] -> [C]   (AvgPool_1a: 5x5 at 160 input,
 * 7x7 at 224 input; slim shrinks the 7x7 kernel to the map size)
 * Inputs are >= 0 (after ReLU6), so (sum + HW/2) / HW rounds to nearest.
 * ------------------------------------------------------------------------ */
void avgpool(const data_t *in, data_t *out, int C, int H)
{
    const int HW = H * H;
    const acc_t half_hw = HW / 2;

    // Local BRAM accumulator buffer for all C = 512 channels
    // Cyclic factor=4 gives 8 independent BRAM ports (4 read + 4 write) for II=1
    acc_t sum_buf[512];
#pragma HLS BIND_STORAGE variable=sum_buf type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=sum_buf cyclic factor=4 dim=1
#pragma HLS DEPENDENCE variable=sum_buf inter false

    // 1. Initialize all 512 channel sums to zero (II=1)
    INIT_SUM: for (int c = 0; c < C; c++) {
#pragma HLS LOOP_TRIPCOUNT min=512 max=512
#pragma HLS PIPELINE II=1
        sum_buf[c] = 0;
    }

    // 2. LOOP INTERCHANGE: Dependent spatial reduction loop (PIX) OUTSIDE,
    //    independent channel loop (CH) INSIDE with UNROLL factor=2 at II=1!
    PIX: for (int p = 0; p < HW; p++) {
#pragma HLS LOOP_TRIPCOUNT min=25 max=25
        CH: for (int c = 0; c < C; c++) {
#pragma HLS LOOP_TRIPCOUNT min=512 max=512
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=2
            sum_buf[c] += (acc_t)in[c * HW + p];
        }
    }

    // 3. Round-to-nearest division and write 512 pooled outputs to BRAM (II=1)
    STORE_AVG: for (int c = 0; c < C; c++) {
#pragma HLS LOOP_TRIPCOUNT min=512 max=512
#pragma HLS PIPELINE II=1
        out[c] = (data_t)((sum_buf[c] + half_hw) / HW);
    }
}

/* ------------------------------------------------------------------------
 * Classifier (Conv2d_1c_1x1): mathematically a 1x1 conv on a 1x1 map.
 * Kept separate here only because it outputs raw int32 scores (no ReLU6,
 * no clamp), so argmax is exact.   in [N]  logits [M]  w [M][N]  b [M]
 * ------------------------------------------------------------------------ */
void fc(const data_t *in, acc_t *logits, const wgt_t *w, const bias_t *b,
        int N, int M)
{
#pragma HLS DEPENDENCE variable=w inter false

    // Local BRAM accumulator buffer for all M = 1001 output classes
    // Cyclic factor=4 provides 8 independent BRAM ports (4 read + 4 write) for II=1
    acc_t acc_buf[1001];
#pragma HLS BIND_STORAGE variable=acc_buf type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=acc_buf cyclic factor=4 dim=1
#pragma HLS DEPENDENCE variable=acc_buf inter false

    // 1. Initialize all 1001 class accumulators with biases (II=1)
    INIT_ACC: for (int to = 0; to < M; to++) {
#pragma HLS LOOP_TRIPCOUNT min=1001 max=1001
#pragma HLS PIPELINE II=1
        acc_buf[to] = b[to];
    }

    // 2. LOOP INTERCHANGE: Dependent reduction loop (TI) OUTSIDE,
    //    independent output-class loop (TO) INSIDE with UNROLL factor=2 at II=1!
    TI: for (int ti = 0; ti < N; ti++) {
#pragma HLS LOOP_TRIPCOUNT min=512 max=512
        const acc_t in_val = (acc_t)in[ti]; // Read once, broadcast 1001 times!

        TO: for (int to = 0; to < M; to++) {
#pragma HLS LOOP_TRIPCOUNT min=1001 max=1001
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=2
            acc_buf[to] += in_val * (acc_t)w[to * N + ti];
        }
    }

    // 3. Burst-write final 1001 raw logits out to DDR m_axi (II=1)
    STORE_LOGITS: for (int to = 0; to < M; to++) {
#pragma HLS LOOP_TRIPCOUNT min=1001 max=1001
#pragma HLS PIPELINE II=1
        logits[to] = acc_buf[to];
    }
}

