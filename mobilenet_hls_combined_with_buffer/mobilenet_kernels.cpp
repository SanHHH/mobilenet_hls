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

ROW: for (int row = 0; row < Ho; row++) {
#pragma HLS LOOP_TRIPCOUNT min=80 max=80
    COL: for (int col = 0; col < Ho; col++) {
#pragma HLS LOOP_TRIPCOUNT min=80 max=80
        TO: for (int to = 0; to < M; to++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=16
            acc_t acc = b[to];
            TI: for (int ti = 0; ti < N; ti++) {
#pragma HLS LOOP_TRIPCOUNT min=3 max=3
                KI: for (int i = 0; i < 3; i++) {
                    KJ: for (int j = 0; j < 3; j++) {
                        int y = S * row + i - pad;
                        int x = S * col + j - pad;
                        if (y >= 0 && y < H && x >= 0 && x < H)
                            acc += (acc_t)in[(ti * H + y) * H + x] *
                                   (acc_t)w[((to * N + ti) * 3 + i) * 3 + j];
                    }
                }
            }
            out[(to * Ho + row) * Ho + col] = requant(acc, shift[to], 1);
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
    // For stride 2, a tile needs (8-1)*2 + 3 = 17 input rows/cols.
    const int MAX_PATCH = 17;
    data_t input_buffer[MAX_PATCH][MAX_PATCH];
#pragma HLS ARRAY_PARTITION variable=input_buffer cyclic factor=3 dim=2
    wgt_t weight_buffer[3][3];
#pragma HLS ARRAY_PARTITION variable=weight_buffer complete dim=0

CH: for (int c = 0; c < C; c++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=512
    LOAD_WEIGHT_ROW: for (int i = 0; i < 3; i++) {
        LOAD_WEIGHT_COL: for (int j = 0; j < 3; j++) {
            weight_buffer[i][j] = w[(c * 3 + i) * 3 + j];
        }
    }
    TILE_ROW: for (int rb = 0; rb < Ho; rb += TILE_H) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=10
        TILE_COL: for (int cb = 0; cb < Ho; cb += TILE_W) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=10
            // Load an input patch including 3x3 halo; out-of-bounds = zero.
            const int patch_h = (TILE_H - 1) * S + 3;
            const int patch_w = (TILE_W - 1) * S + 3;
            LOAD_INPUT_ROW: for (int py = 0; py < patch_h; py++) {
#pragma HLS LOOP_TRIPCOUNT min=10 max=17
                LOAD_INPUT_COL: for (int px = 0; px < patch_w; px++) {
#pragma HLS LOOP_TRIPCOUNT min=10 max=17
                    const int y = rb * S + py - pad;
                    const int x = cb * S + px - pad;
                    input_buffer[py][px] =
                        (y >= 0 && y < H && x >= 0 && x < H)
                        ? in[(c * H + y) * H + x] : (data_t)0;
                }
            }
            ROW: for (int row = rb; row < rb + TILE_H && row < Ho; row++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=8
                COL: for (int col = cb; col < cb + TILE_W && col < Ho; col++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=8
                    acc_t acc = b[c];
                    KI: for (int i = 0; i < 3; i++) {
#pragma HLS PIPELINE II=1
                        KJ: for (int j = 0; j < 3; j++) {
#pragma HLS UNROLL factor=3
                            const int ly = (row - rb) * S + i;
                            const int lx = (col - cb) * S + j;
                            acc += (acc_t)input_buffer[ly][lx] *
                                   (acc_t)weight_buffer[i][j];
                        }
                    }
                    out[(c * Ho + row) * Ho + col] = requant(acc, shift[c], 1);
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
void conv1x1(const data_t *in, data_t *out, const wgt_t *w, const bias_t *b,
             int N, int M, int H, const int8_t *shift)
{
    const int HW = H * H;

ROW: for (int row = 0; row < H; row++) {
#pragma HLS LOOP_TRIPCOUNT min=5 max=80
    COL: for (int col = 0; col < H; col++) {
#pragma HLS LOOP_TRIPCOUNT min=5 max=80
        TO: for (int to = 0; to < M; to++) {
#pragma HLS LOOP_TRIPCOUNT min=32 max=512
            acc_t acc = b[to];
            TI: for (int ti = 0; ti < N; ti++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=512
                acc += (acc_t)in[ti * HW + row * H + col] *
                       (acc_t)w[to * N + ti];
            }
            out[to * HW + row * H + col] = requant(acc, shift[to], 1);
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
CH: for (int c = 0; c < C; c++) {
#pragma HLS LOOP_TRIPCOUNT min=512 max=512
        acc_t sum = 0;
    PIX: for (int p = 0; p < HW; p++) {
#pragma HLS LOOP_TRIPCOUNT min=25 max=25
            sum += in[c * HW + p];
        }
        out[c] = (data_t)((sum + HW / 2) / HW);
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
TO: for (int to = 0; to < M; to++) {
#pragma HLS LOOP_TRIPCOUNT min=1001 max=1001
        acc_t acc = b[to];
    TI: for (int ti = 0; ti < N; ti++) {
#pragma HLS LOOP_TRIPCOUNT min=512 max=512
            acc += (acc_t)in[ti] * (acc_t)w[to * N + ti];
        }
        logits[to] = acc;
    }
}
