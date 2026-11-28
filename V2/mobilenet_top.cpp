/*
 * mobilenet_top.cpp -- KV260 Asymmetric URAM/BRAM Hybrid (100% Legal Resource Fit)
 *
 * Resource Allocation on XCK26:
 *   - uram_weights (8-bit, factor=8)    -> 41 / 64 URAM288
 *   - uram_fm_ab   (2*FM_MAX, factor=4) -> 19 / 64 URAM288 (Total URAM = 60 / 64)
 *   - bram_fm_tmp  (FM_MAX, factor=4)   -> 182 / 288 BRAM18K (Total BRAM ~ 210 / 288)
 */
#include "mobilenet.h"

void mobilenet_top(const data_t *image, data_t *fm, const wgt_t *weights,
                   const bias_t *biases, const int8_t *shifts, acc_t *logits)
{
#pragma HLS INTERFACE m_axi port=image   offset=slave bundle=gmem0 depth=76800
#pragma HLS INTERFACE m_axi port=fm      offset=slave bundle=gmem1 depth=614400
#pragma HLS INTERFACE m_axi port=weights offset=slave bundle=gmem2 depth=1320160
#pragma HLS INTERFACE m_axi port=biases  offset=slave bundle=gmem3 depth=6473
#pragma HLS INTERFACE m_axi port=shifts  offset=slave bundle=gmem3 depth=5472
#pragma HLS INTERFACE m_axi port=logits  offset=slave bundle=gmem0 depth=1001
#pragma HLS INTERFACE s_axilite port=image   bundle=ctrl
#pragma HLS INTERFACE s_axilite port=fm      bundle=ctrl
#pragma HLS INTERFACE s_axilite port=weights bundle=ctrl
#pragma HLS INTERFACE s_axilite port=biases  bundle=ctrl
#pragma HLS INTERFACE s_axilite port=shifts  bundle=ctrl
#pragma HLS INTERFACE s_axilite port=logits  bundle=ctrl
#pragma HLS INTERFACE s_axilite port=return  bundle=ctrl

    // =========================================================================
    // 1. URAM TIER (60 / 64 URAM288): All Weights + Ping-Pong Regions A & B
    // =========================================================================
    static wgt_t uram_weights[807648];
#pragma HLS BIND_STORAGE variable=uram_weights type=ram_2p impl=uram latency=2
#pragma HLS ARRAY_RESHAPE variable=uram_weights cyclic factor=8 dim=1

    data_t uram_fm_ab[2 * FM_MAX];
#pragma HLS BIND_STORAGE variable=uram_fm_ab type=ram_2p impl=uram latency=2
#pragma HLS ARRAY_RESHAPE variable=uram_fm_ab cyclic factor=4 dim=1

    // =========================================================================
    // 2. BRAM TIER (~210 / 288 BRAM18K): Intermediate DW Buffer + Biases/Shifts
    //    - dwconv3x3 ALWAYS reads URAM (uram_fm_ab) and writes BRAM (bram_fm_tmp)
    //    - conv1x1   ALWAYS reads BRAM (bram_fm_tmp) and writes URAM (uram_fm_ab)
    //    -> Zero mixed-core port conflict & zero read/write aliasing!
    // =========================================================================
    data_t bram_fm_tmp[FM_MAX];
#pragma HLS BIND_STORAGE variable=bram_fm_tmp type=ram_2p impl=uram latency=2
#pragma HLS ARRAY_RESHAPE variable=bram_fm_tmp cyclic factor=4 dim=1

    static bias_t bram_biases[6473];
#pragma HLS BIND_STORAGE variable=bram_biases type=ram_1p impl=bram

    static int8_t bram_shifts[5472];
#pragma HLS BIND_STORAGE variable=bram_shifts type=ram_1p impl=lutram

    // =========================================================================
    // 3. Burst-Load Parameters from DDR into On-Chip URAM/BRAM 
    // =========================================================================
LOAD_WEIGHTS: for (int i = 0; i < 807648; i++) {
#pragma HLS PIPELINE II=1
        uram_weights[i] = weights[i];
    }

LOAD_BIASES: for (int i = 0; i < 6473; i++) {
#pragma HLS PIPELINE II=1
        bram_biases[i] = biases[i];
    }

LOAD_SHIFTS: for (int i = 0; i < 5472; i++) {
#pragma HLS PIPELINE II=1
        bram_shifts[i] = shifts[i];
}
    int H       = IMG_SIZE;
    int cin     = IMG_CH;
    int w_off   = 0;
    int b_off   = 0;
    int cur_off = 0; // 0 = Region A, FM_MAX = Region B inside uram_fm_ab

    /* ---- Conv2d_0: standard 3x3 conv, stride 2, 3 -> 16 channels ---- */
    {
        const int d = scaled_depth(MOBILENET_DEFS[0].depth);
        const int s = MOBILENET_DEFS[0].stride;
        conv3x3(image, uram_fm_ab, uram_weights + w_off, bram_biases + b_off,
                cin, d, H, s, bram_shifts + b_off);
        w_off  += d * cin * 9;
        b_off  += d;
        H       = out_size(H, s);
        cin     = d;
        cur_off = 0;
    }

    /* ---- Conv2d_1 ... Conv2d_13: depthwise 3x3 then pointwise 1x1 ---- */
LAYERS: for (int i = 1; i < NUM_DEFS; i++) {
        const int d       = scaled_depth(MOBILENET_DEFS[i].depth);
        const int s       = MOBILENET_DEFS[i].stride;
        const int Ho      = out_size(H, s);
        const int nxt_off = (cur_off == 0) ? FM_MAX : 0;

        dwconv3x3(uram_fm_ab + cur_off, bram_fm_tmp, uram_weights + w_off, bram_biases + b_off,
                  cin, H, s, bram_shifts + b_off);
        w_off += cin * 9;
        b_off += cin;

        conv1x1(bram_fm_tmp, uram_fm_ab + nxt_off, uram_weights + w_off, bram_biases + b_off,
                cin, d, Ho, bram_shifts + b_off);
        w_off += cin * d;
        b_off += d;

        cur_off = nxt_off;
        H       = Ho;
        cin     = d;
    }

    /* ---- Logits: 5x5 average pool, then 1x1 conv to 1001 classes ---- */
    avgpool(uram_fm_ab + cur_off, bram_fm_tmp, cin, H);
    fc(bram_fm_tmp, logits, weights + w_off, bram_biases + b_off,
       cin, NUM_CLASSES);
}
