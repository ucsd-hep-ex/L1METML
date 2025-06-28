#ifndef L1METML_V1_H_
#define L1METML_V1_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"

// Prototype of top level function for C-synthesis
void L1METML_v1(
    input5_t input_cont[N_INPUT_1_5*N_INPUT_2_5], input19_t input_pxpy[N_INPUT_1_19*N_INPUT_2_19], input_t input_cat0[N_INPUT_1_1], input2_t input_cat1[N_INPUT_1_2],
    result_t layer21_out[N_FILT_21]
);

#endif
