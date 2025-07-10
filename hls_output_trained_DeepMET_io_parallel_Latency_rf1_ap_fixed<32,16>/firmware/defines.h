#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

// hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 100
#define N_INPUT_1_2 100
#define N_LAYER_1_3 100
#define N_LAYER_2_3 2
#define N_LAYER_1_4 100
#define N_LAYER_2_4 2
#define N_INPUT_1_5 100
#define N_INPUT_2_5 4
#define OUT_CONCAT_0_6 100
#define OUT_CONCAT_1_6 4
#define OUT_CONCAT_0_7 100
#define OUT_CONCAT_1_7 8
#define N_OUTPUTS_22 100
#define N_FILT_22 12
#define N_LAYER_1_8 100
#define N_LAYER_2_8 12
#define N_OUTPUTS_23 100
#define N_FILT_23 36
#define N_LAYER_1_12 100
#define N_LAYER_2_12 36
#define N_OUTPUTS_24 100
#define N_FILT_24 1
#define N_INPUT_1_19 100
#define N_INPUT_2_19 2
#define N_INPUT_1_19 100
#define N_INPUT_2_19 2
#define N_FILT_21 2

// hls-fpga-machine-learning insert layer-precision
typedef ap_uint<4> input_t;
typedef ap_uint<4> input2_t;
typedef ap_fixed<32,16> layer3_t;
typedef ap_fixed<32,16> embedding0_embeddings_t;
typedef ap_fixed<32,16> layer4_t;
typedef ap_fixed<32,16> embedding1_embeddings_t;
typedef ap_fixed<32,16> input5_t;
typedef ap_fixed<32,16> layer6_t;
typedef ap_fixed<32,16> layer7_t;
typedef ap_fixed<32,16> model_default_t;
typedef ap_fixed<32,16> layer22_t;
typedef ap_fixed<32,16> dense_weight_t;
typedef ap_fixed<32,16> dense_bias_t;
typedef ap_fixed<32,16> layer11_t;
typedef ap_fixed<18,8> activation_table_t;
typedef ap_fixed<32,16> layer23_t;
typedef ap_fixed<32,16> dense_1_weight_t;
typedef ap_fixed<32,16> dense_1_bias_t;
typedef ap_fixed<32,16> layer15_t;
typedef ap_fixed<18,8> activation_1_table_t;
typedef ap_fixed<32,16> layer24_t;
typedef ap_fixed<32,16> met_weight_weight_t;
typedef ap_fixed<32,16> met_weight_bias_t;
typedef ap_fixed<32,16> input19_t;
typedef ap_fixed<32,16> layer20_t;
typedef ap_fixed<32,16> result_t;

#endif
