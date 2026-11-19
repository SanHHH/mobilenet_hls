
/*
 * EECS 298 - Deep Learning Accelerator Design
 * MobileNetV1 HLS - Local Buffer Optimization
 *
 * Version: Three-row Line Buffer and Sliding Window
 *
 * Optimized:
 *   1. Standard Conv3x3: Local Input and Weight Buffers
 *   2. Depthwise Conv3x3: 3-row circular line buffer and 3x3 window
 *
 * Unchanged:
 *   3. Pointwise Conv1x1
 *   4. Average Pooling
 *   5. Fully Connected
 *
 * Target:
 *   MobileNetV1 alpha=0.5, 160x160
 *   Kria K26 FPGA
 *   Vitis HLS 2023.1
 *
 * Debugging strategy:
 *   - Avoid automatic pipelining of large loop nests.
 *   - Keep separate load and compute functions.
 *   - Preserve the original MAC accumulation order.
 */

#include "mobilenet.h"

// ============================================================
// 1. STANDARD CONVOLUTION 3x3 - LOCAL BUFFER
// ============================================================

void conv3x3(
    const data_t *in,
    data_t *out,
    const wgt_t *w,
    const bias_t *b,
    int N,
    int M,
    int H,
    int S,
    const int8_t *shift
)
{
    const int Ho = out_size(H, S);
    const int pad = same_pad_begin(H, S, 3);

    data_t input_buffer[3][160][160];
    wgt_t weight_buffer[16][3][3][3];

#pragma HLS BIND_STORAGE variable=input_buffer type=ram_2p impl=bram

LOAD_INPUT_CHANNEL:
    for (int n = 0; n < N; n++) {
#pragma HLS PIPELINE off
#pragma HLS LOOP_FLATTEN off

LOAD_INPUT_ROW:
        for (int y = 0; y < H; y++) {
#pragma HLS PIPELINE off

LOAD_INPUT_COL:
            for (int x = 0; x < H; x++) {
#pragma HLS PIPELINE off
                input_buffer[n][y][x] =
                    in[(n * H + y) * H + x];
            }
        }
    }

LOAD_WEIGHT_OUTPUT:
    for (int m = 0; m < M; m++) {
#pragma HLS PIPELINE off
#pragma HLS LOOP_FLATTEN off

LOAD_WEIGHT_INPUT:
        for (int n = 0; n < N; n++) {
#pragma HLS PIPELINE off

LOAD_WEIGHT_ROW:
            for (int i = 0; i < 3; i++) {
#pragma HLS PIPELINE off

LOAD_WEIGHT_COL:
                for (int j = 0; j < 3; j++) {
#pragma HLS PIPELINE off
                    weight_buffer[m][n][i][j] =
                        w[((m * N + n) * 3 + i) * 3 + j];
                }
            }
        }
    }

ROW:
    for (int row = 0; row < Ho; row++) {
#pragma HLS PIPELINE off
#pragma HLS LOOP_FLATTEN off
#pragma HLS LOOP_TRIPCOUNT min=80 max=80

COL:
        for (int col = 0; col < Ho; col++) {
#pragma HLS PIPELINE off
#pragma HLS LOOP_TRIPCOUNT min=80 max=80

TO:
            for (int to = 0; to < M; to++) {
#pragma HLS PIPELINE off
#pragma HLS LOOP_TRIPCOUNT min=16 max=16

                acc_t acc = b[to];

TI:
                for (int ti = 0; ti < N; ti++) {
#pragma HLS LOOP_TRIPCOUNT min=3 max=3

KI:
                    for (int i = 0; i < 3; i++) {

KJ:
                        for (int j = 0; j < 3; j++) {

                            int y = S * row + i - pad;
                            int x = S * col + j - pad;

                            if (y >= 0 && y < H &&
                                x >= 0 && x < H) {

                                acc +=
                                    (acc_t)input_buffer[ti][y][x] *
                                    (acc_t)weight_buffer[to][ti][i][j];
                            }
                        }
                    }
                }

                out[(to * Ho + row) * Ho + col] =
                    requant(acc, shift[to], 1);
            }
        }
    }
}

// ============================================================
// 2. DEPTHWISE CONVOLUTION - CIRCULAR LINE BUFFER
// ============================================================

// ------------------------------------------------------------
// 2.1 Load only the rows needed for the current output row.
// ------------------------------------------------------------

static void dw_prepare_rows(
    const data_t *in,
    data_t line_buffer[3][80],
    int row_tags[3],
    int c,
    int H,
    int S,
    int output_row
)
{
#pragma HLS INLINE off
    const int pad = same_pad_begin(H, S, 3);
DW_PREPARE_ROW:
    for (int i = 0; i < 3; ++i) {
#pragma HLS PIPELINE off
#pragma HLS LOOP_TRIPCOUNT min=3 max=3
        const int y = S * output_row + i - pad;
        if (y >= 0 && y < H) {
            const int bank = y % 3;
            if (row_tags[bank] != y) {
DW_PREPARE_COL:
                for (int x = 0; x < H; ++x) {
#pragma HLS PIPELINE off
#pragma HLS LOOP_TRIPCOUNT min=5 max=80
                    line_buffer[bank][x] = in[(c * H + y) * H + x];
                }
                row_tags[bank] = y;
            }
        }
    }
}

// ------------------------------------------------------------
// 2.2 Load one channel's 3x3 weights
// ------------------------------------------------------------

static void dw_load_weights(
    const wgt_t *w,
    wgt_t weight_buffer[3][3],
    int c
)
{
#pragma HLS INLINE off

LOAD_WEIGHT_ROW:
    for (int i = 0; i < 3; i++) {
#pragma HLS PIPELINE off
#pragma HLS LOOP_TRIPCOUNT min=3 max=3

LOAD_WEIGHT_COL:
        for (int j = 0; j < 3; j++) {
#pragma HLS PIPELINE off
#pragma HLS LOOP_TRIPCOUNT min=3 max=3

            weight_buffer[i][j] =
                w[(c * 3 + i) * 3 + j];
        }
    }
}

// ------------------------------------------------------------
// 2.3 Compute one output row with a horizontally sliding window.
// ------------------------------------------------------------

static void dw_compute_row(
    data_t line_buffer[3][80],
    wgt_t weight_buffer[3][3],
    data_t *out,
    acc_t bias,
    int c,
    int H,
    int S,
    int8_t shift_value,
    int row
)
{
#pragma HLS INLINE off
    const int Ho = out_size(H, S);
    const int pad = same_pad_begin(H, S, 3);
    data_t window[3][3];

DW_OUTPUT_COL:
    for (int col = 0; col < Ho; ++col) {
#pragma HLS PIPELINE off
#pragma HLS LOOP_TRIPCOUNT min=3 max=80
        // Reuse the last 3-S columns of the previous window.
DW_WINDOW_ROW:
        for (int i = 0; i < 3; ++i) {
#pragma HLS PIPELINE off
#pragma HLS LOOP_TRIPCOUNT min=3 max=3
DW_WINDOW_COL:
            for (int j = 0; j < 3; ++j) {
#pragma HLS PIPELINE off
#pragma HLS LOOP_TRIPCOUNT min=3 max=3
                if (col != 0 && j < 3 - S) {
                    window[i][j] = window[i][j + S];
                } else {
                    const int y = S * row + i - pad;
                    const int x = S * col + j - pad;
                    window[i][j] =
                        (y >= 0 && y < H && x >= 0 && x < H)
                            ? line_buffer[y % 3][x] : (data_t)0;
                }
            }
        }

        acc_t acc = bias;
DW_MAC_ROW:
        for (int i = 0; i < 3; ++i) {
#pragma HLS PIPELINE off
#pragma HLS LOOP_TRIPCOUNT min=3 max=3
DW_MAC_COL:
            for (int j = 0; j < 3; ++j) {
#pragma HLS PIPELINE off
#pragma HLS LOOP_TRIPCOUNT min=3 max=3
                // Skip padding MACs to preserve reference arithmetic.
                const int y = S * row + i - pad;
                const int x = S * col + j - pad;
                if (y >= 0 && y < H && x >= 0 && x < H) {
                    acc += (acc_t)window[i][j] *
                           (acc_t)weight_buffer[i][j];
                }
            }
        }
        out[(c * Ho + row) * Ho + col] = requant(acc, shift_value, 1);
    }
}

// ------------------------------------------------------------
// 2.4 Main Depthwise Function
// ------------------------------------------------------------

void dwconv3x3(
    const data_t *in,
    data_t *out,
    const wgt_t *w,
    const bias_t *b,
    int C,
    int H,
    int S,
    const int8_t *shift
)
{
    // Only three feature-map rows are kept on chip.
    data_t line_buffer[3][80];
    wgt_t weight_buffer[3][3];
#pragma HLS BIND_STORAGE variable=line_buffer type=ram_2p impl=bram

CHANNEL:
    for (int c = 0; c < C; c++) {
#pragma HLS PIPELINE off
#pragma HLS LOOP_FLATTEN off
#pragma HLS LOOP_TRIPCOUNT min=16 max=512

        int row_tags[3] = {-1, -1, -1};
        dw_load_weights(w, weight_buffer, c);
DW_OUTPUT_ROW:
        for (int row = 0; row < out_size(H, S); ++row) {
#pragma HLS PIPELINE off
#pragma HLS LOOP_FLATTEN off
#pragma HLS LOOP_TRIPCOUNT min=3 max=80
            dw_prepare_rows(in, line_buffer, row_tags, c, H, S, row);
            dw_compute_row(line_buffer, weight_buffer, out,
                           b[c], c, H, S, shift[c], row);
        }
    }
}

// ============================================================
// 3. POINTWISE CONVOLUTION 1x1 - BASELINE
// ============================================================

void conv1x1(
    const data_t *in,
    data_t *out,
    const wgt_t *w,
    const bias_t *b,
    int N,
    int M,
    int H,
    const int8_t *shift
)
{
    const int HW = H * H;

ROW:
    for (int row = 0; row < H; row++) {
#pragma HLS LOOP_TRIPCOUNT min=5 max=80

COL:
        for (int col = 0; col < H; col++) {
#pragma HLS LOOP_TRIPCOUNT min=5 max=80

TO:
            for (int to = 0; to < M; to++) {
#pragma HLS LOOP_TRIPCOUNT min=32 max=512

                acc_t acc = b[to];

TI:
                for (int ti = 0; ti < N; ti++) {
#pragma HLS LOOP_TRIPCOUNT min=16 max=512

                    acc +=
                        (acc_t)in[ti * HW + row * H + col] *
                        (acc_t)w[to * N + ti];
                }

                out[to * HW + row * H + col] =
                    requant(acc, shift[to], 1);
            }
        }
    }
}

// ============================================================
// 4. GLOBAL AVERAGE POOLING - BASELINE
// ============================================================

void avgpool(
    const data_t *in,
    data_t *out,
    int C,
    int H
)
{
    const int HW = H * H;

CH:
    for (int c = 0; c < C; c++) {
#pragma HLS LOOP_TRIPCOUNT min=512 max=512

        acc_t sum = 0;

PIX:
        for (int p = 0; p < HW; p++) {
#pragma HLS LOOP_TRIPCOUNT min=25 max=25

            sum += in[c * HW + p];
        }

        out[c] = (data_t)((sum + HW / 2) / HW);
    }
}

// ============================================================
// 5. FULLY CONNECTED - BASELINE
// ============================================================

void fc(
    const data_t *in,
    acc_t *logits,
    const wgt_t *w,
    const bias_t *b,
    int N,
    int M
)
{
TO:
    for (int to = 0; to < M; to++) {
#pragma HLS LOOP_TRIPCOUNT min=1001 max=1001

        acc_t acc = b[to];

TI:
        for (int ti = 0; ti < N; ti++) {
#pragma HLS LOOP_TRIPCOUNT min=512 max=512

            acc +=
                (acc_t)in[ti] *
                (acc_t)w[to * N + ti];
        }

        logits[to] = acc;
    }
}
