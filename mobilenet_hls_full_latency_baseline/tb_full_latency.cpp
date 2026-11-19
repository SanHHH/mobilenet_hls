/* Single-inference, deterministic MobileNetV1 C/RTL latency testbench.
 * Uses the SAME full mobilenet_top and complete layer topology as baseline.
 * No optional binary files, random seeds, or extra kernel unit-test calls.
 */
#include "mobilenet.h"
#include "mobilenet_ref.h"
#include <cstdio>

static data_t image[IMG_CH * IMG_SIZE * IMG_SIZE];
static data_t fm[FM_TOTAL];
static wgt_t weights[W_TOTAL];
static bias_t biases[B_TOTAL];
static int8_t shifts[S_TOTAL];
static acc_t rtl_logits[NUM_CLASSES];
static acc_t ref_logits[NUM_CLASSES];
static data_t ref_a[FM_MAX];
static data_t ref_b[FM_MAX];

int main()
{
    for (int i = 0; i < IMG_CH * IMG_SIZE * IMG_SIZE; ++i)
        image[i] = (data_t)(((i * 13 + 7) % 65) - 32);
    for (int i = 0; i < W_TOTAL; ++i)
        weights[i] = (wgt_t)(((i * 7 + 3) % 7) - 3);
    for (int i = 0; i < B_TOTAL; ++i)
        biases[i] = (bias_t)((i % 17) - 8);
    for (int i = 0; i < S_TOTAL; ++i)
        shifts[i] = 4;

    // Exactly one top-level transaction; co-sim reports its complete latency.
    mobilenet_top(image, fm, weights, biases, shifts, rtl_logits);

    ref_mobilenet(image, weights, biases, shifts, ref_logits,
                  ref_a, ref_b, 0);

    int errors = 0;
    for (int i = 0; i < NUM_CLASSES; ++i) {
        if (rtl_logits[i] != ref_logits[i]) {
            if (errors < 5)
                std::printf("Mismatch class %d: got %d expected %d\n",
                            i, (int)rtl_logits[i], (int)ref_logits[i]);
            ++errors;
        }
    }

    if (errors) {
        std::printf("Full MobileNetV1 FAIL (%d mismatches)\n", errors);
        return 1;
    }
    std::printf("Full MobileNetV1 ALL TESTS PASSED (%d logits)\n", NUM_CLASSES);
    return 0;
}
