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
    const int TC = 2; // Process 2 channels in parallel (16 MACs/cycle total)

    // Partition dim=1 (TC) and dim=3 (cols) completely for conflict-free parallel reads
    data_t input_buffer[TC][MAX_PATCH][MAX_PATCH];
#pragma HLS ARRAY_PARTITION variable=input_buffer complete dim=1
#pragma HLS ARRAY_PARTITION variable=input_buffer complete dim=3

    wgt_t weight_buffer[TC][3][3];
#pragma HLS ARRAY_PARTITION variable=weight_buffer complete dim=0

    // Local 2 x 8 x 8 tile accumulators partitioned completely on TC and TILE_W
    acc_t acc_tile[TC][TILE_H][TILE_W];
#pragma HLS ARRAY_PARTITION variable=acc_tile complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc_tile complete dim=3

CH_TILE: for (int cb_ch = 0; cb_ch < C; cb_ch += TC) {
#pragma HLS LOOP_TRIPCOUNT min=8 max=256

    // 1. Load 2 x 3 x 3 = 18 weights into registers (unrolled by 8 to match URAM factor=8)
    LOAD_W_TC: for (int tc = 0; tc < TC; tc++) {
        const int c = cb_ch + tc;
        const int w_c_off = c * 9;
        LOAD_WEIGHT: for (int k = 0; k < 9; k++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
            weight_buffer[tc][k / 3][k % 3] = (c < C) ? w[w_c_off + k] : (wgt_t)0;
        }
    }

    TILE_ROW: for (int rb = 0; rb < Ho; rb += TILE_H) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=10
        TILE_COL: for (int cb = 0; cb < Ho; cb += TILE_W) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=10
            const int patch_h = (TILE_H - 1) * S + 3;
            const int patch_w = (TILE_W - 1) * S + 3;

            // 2. Load input patch + halo for TC=2 channels (4 pixels/cycle at II=1)
            LOAD_IN_TC: for (int tc = 0; tc < TC; tc++) {
                const int c = cb_ch + tc;
                const int in_c_off = c * H * H;
                LOAD_INPUT_ROW: for (int py = 0; py < patch_h; py++) {
#pragma HLS LOOP_TRIPCOUNT min=10 max=17
                    LOAD_INPUT_COL: for (int px = 0; px < patch_w; px++) {
#pragma HLS LOOP_TRIPCOUNT min=10 max=17
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
                        const int y = rb * S + py - pad;
                        const int x = cb * S + px - pad;
                        input_buffer[tc][py][px] =
                            (c < C && y >= 0 && y < H && x >= 0 && x < H)
                            ? in[in_c_off + y * H + x] : (data_t)0;
                    }
                }
            }

            // 3. Initialize 2 x 8 x 8 accumulators with channel biases (16/cycle at II=1)
            INIT_Y: for (int ty = 0; ty < TILE_H; ty++) {
#pragma HLS PIPELINE II=1
                INIT_TC: for (int tc = 0; tc < TC; tc++) {
#pragma HLS UNROLL
                    const int c = cb_ch + tc;
                    const bias_t bias_c = (c < C) ? b[c] : (bias_t)0;
                    INIT_X: for (int tx = 0; tx < TILE_W; tx++) {
#pragma HLS UNROLL
                        acc_tile[tc][ty][tx] = bias_c;
                    }
                }
            }

            // 4. Compute 2 channels x 8 horizontal pixels = 16 MACs/cycle at II=1
            KI: for (int i = 0; i < 3; i++) {
                KJ: for (int j = 0; j < 3; j++) {
                    TILE_Y: for (int ty = 0; ty < TILE_H; ty++) {
#pragma HLS PIPELINE II=1
                        TILE_TC: for (int tc = 0; tc < TC; tc++) {
#pragma HLS UNROLL
                            const acc_t w_val = (acc_t)weight_buffer[tc][i][j];
                            TILE_X: for (int tx = 0; tx < TILE_W; tx++) {
#pragma HLS UNROLL
                                const int ly = ty * S + i;
                                const int lx = tx * S + j;
                                acc_tile[tc][ty][tx] +=
                                    (acc_t)input_buffer[tc][ly][lx] * w_val;
                            }
                        }
                    }
                }
            }

            // 5. Requantize & store valid in-bounds pixels for TC=2 channels
            STORE_TC: for (int tc = 0; tc < TC; tc++) {
                const int c = cb_ch + tc;
                if (c < C) {
                    const int8_t shift_c = shift[c];
                    const int out_c_off = c * Ho * Ho;
                    STORE_Y: for (int ty = 0; ty < TILE_H; ty++) {
                        const int row = rb + ty;
                        STORE_X: for (int tx = 0; tx < TILE_W; tx++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
                            const int col = cb + tx;
                            if (row < Ho && col < Ho) {
                                out[out_c_off + row * Ho + col] =
                                    requant(acc_tile[tc][ty][tx], shift_c, 1);
                            }
                        }
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
    const int TM = 16;
    const int TP = 8;

    // Buffer 8 horizontal pixels across up to 512 input channels
    data_t in_pix_buf[512][TP];
#pragma HLS BIND_STORAGE variable=in_pix_buf type=ram_2p impl=bram latency=1
#pragma HLS ARRAY_PARTITION variable=in_pix_buf complete dim=2

    // 16 output channels x 8 pixels accumulator tile (completely partitioned on TM)
    acc_t acc_buf[TM][TP];
#pragma HLS ARRAY_PARTITION variable=acc_buf complete dim=1

    // Local 16-weight slice fetched once per input channel and reused across 8 pixels
    wgt_t w_buf[TM];
#pragma HLS ARRAY_PARTITION variable=w_buf complete dim=1

    ROW: for (int row = 0; row < H; row++) {
#pragma HLS LOOP_TRIPCOUNT min=5 max=80

        TILE_COL: for (int cb = 0; cb < H; cb += TP) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=10

            // 1. Load 8 horizontal pixels across N channels into local BRAM
            LOAD_PIX_TI: for (int ti = 0; ti < N; ti++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=512
                LOAD_PIX_P: for (int p = 0; p < TP; p++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
                    int col = cb + p;
                    in_pix_buf[ti][p] = (col < H) ? in[ti * HW + row * H + col] : (data_t)0;
                }
            }
            
            TILE_M: for (int tm = 0; tm < M; tm += TM) {
#pragma HLS LOOP_TRIPCOUNT min=2 max=32

            // 2. Initialize 16 x 8 accumulators with channel biases
            INIT_ACC_P: for (int p = 0; p < TP; p++) {
#pragma HLS PIPELINE II=1
                INIT_ACC_M: for (int m = 0; m < TM; m++) {
#pragma HLS UNROLL
                    acc_buf[m][p] = (tm + m < M) ? b[tm + m] : (bias_t)0;
                }
            }

            // 3. Compute: fetch 16 weights once per ti, reuse across 8 pixels
            TI: for (int ti = 0; ti < N; ti++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=512

                // Fetch 16 weights from URAM (2 cycles @ 8 weights/cycle)
                LOAD_W: for (int m = 0; m < TM; m++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=16
                    w_buf[m] = (tm + m < M) ? w[(tm + m) * N + ti] : (wgt_t)0;
                }

                // 16 parallel MACs/cycle across 8 pixels (8 cycles @ II=1)
                COMPUTE_P: for (int p = 0; p < TP; p++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=2
#pragma HLS DEPENDENCE variable=acc_buf inter false
                    const acc_t in_val = (acc_t)in_pix_buf[ti][p];

                    COMPUTE_M: for (int m = 0; m < TM; m++) {
#pragma HLS UNROLL
                        acc_buf[m][p] += in_val * (acc_t)w_buf[m];
                    }
                }
            }

            // 4. Requantize & store valid in-bounds pixels back to URAM
            STORE_M: for (int m = 0; m < TM; m++) {
                const int out_ch = tm + m;
                if (out_ch < M) {
                    const int8_t sh = shift[out_ch];
                    STORE_P: for (int p = 0; p < TP; p++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
                        int col = cb + p;
                        if (col < H) {
                            out[out_ch * HW + row * H + col] = requant(acc_buf[m][p], sh, 1);
                        }
                    }
                }
            }
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
#pragma HLS BIND_STORAGE variable=sum_buf type=ram_2p impl=bram latency=2
#pragma HLS ARRAY_PARTITION variable=sum_buf cyclic factor=4 dim=1

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
#pragma HLS DEPENDENCE variable=sum_buf inter false
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
#pragma HLS BIND_STORAGE variable=acc_buf type=ram_2p impl=bram latency=2
#pragma HLS ARRAY_PARTITION variable=acc_buf cyclic factor=4 dim=1

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
#pragma HLS DEPENDENCE variable=acc_buf inter false
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