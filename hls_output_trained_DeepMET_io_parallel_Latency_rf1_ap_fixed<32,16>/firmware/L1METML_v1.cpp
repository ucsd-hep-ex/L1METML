#include <iostream>

#include "L1METML_v1.h"
#include "parameters.h"

void L1METML_v1(
    input5_t input_cont[N_INPUT_1_5*N_INPUT_2_5], input19_t input_pxpy[N_INPUT_1_19*N_INPUT_2_19], input_t input_cat0[N_INPUT_1_1], input2_t input_cat1[N_INPUT_1_2],
    result_t layer21_out[N_FILT_21]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=input_cont complete dim=0
    #pragma HLS ARRAY_RESHAPE variable=input_pxpy complete dim=0
    #pragma HLS ARRAY_RESHAPE variable=input_cat0 complete dim=0
    #pragma HLS ARRAY_RESHAPE variable=input_cat1 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer21_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=input_cont,input_pxpy,input_cat0,input_cat1,layer21_out 
    #pragma HLS DATAFLOW 

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        // hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<embedding0_embeddings_t, 12>(e3, "e3.txt");
        nnet::load_weights_from_txt<embedding1_embeddings_t, 8>(e4, "e4.txt");
        nnet::load_weights_from_txt<dense_weight_t, 96>(w22, "w22.txt");
        nnet::load_weights_from_txt<dense_bias_t, 12>(b22, "b22.txt");
        nnet::load_weights_from_txt<dense_1_weight_t, 432>(w23, "w23.txt");
        nnet::load_weights_from_txt<dense_1_bias_t, 36>(b23, "b23.txt");
        nnet::load_weights_from_txt<met_weight_weight_t, 36>(w24, "w24.txt");
        nnet::load_weights_from_txt<met_weight_bias_t, 1>(b24, "b24.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    layer3_t layer3_out[N_LAYER_1_3*N_LAYER_2_3];
    #pragma HLS ARRAY_PARTITION variable=layer3_out complete dim=0
    nnet::embedding<input_t, layer3_t, config3>(input_cat0, layer3_out, e3); // embedding0
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer3_t>(layer3_out, "embedding0", N_LAYER_1_3*N_LAYER_2_3);
#endif

    layer4_t layer4_out[N_LAYER_1_4*N_LAYER_2_4];
    #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0
    nnet::embedding<input2_t, layer4_t, config4>(input_cat1, layer4_out, e4); // embedding1
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer4_t>(layer4_out, "embedding1", N_LAYER_1_4*N_LAYER_2_4);
#endif

    layer6_t layer6_out[OUT_CONCAT_0_6*OUT_CONCAT_1_6];
    #pragma HLS ARRAY_PARTITION variable=layer6_out complete dim=0
    nnet::concatenate2d<layer3_t, layer4_t, layer6_t, config6>(layer3_out, layer4_out, layer6_out); // concatenate
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer6_t>(layer6_out, "concatenate", OUT_CONCAT_0_6*OUT_CONCAT_1_6);
#endif

    layer7_t layer7_out[OUT_CONCAT_0_7*OUT_CONCAT_1_7];
    #pragma HLS ARRAY_PARTITION variable=layer7_out complete dim=0
    nnet::concatenate2d<input5_t, layer6_t, layer7_t, config7>(input_cont, layer6_out, layer7_out); // concatenate_1
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer7_t>(layer7_out, "concatenate_1", OUT_CONCAT_0_7*OUT_CONCAT_1_7);
#endif

    layer22_t layer22_out[N_OUTPUTS_22*N_FILT_22];
    #pragma HLS ARRAY_PARTITION variable=layer22_out complete dim=0
    nnet::pointwise_conv_1d_cl<layer7_t, layer22_t, config22>(layer7_out, layer22_out, w22, b22); // dense
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer22_t>(layer22_out, "dense", N_OUTPUTS_22*N_FILT_22);
#endif

    layer11_t layer11_out[N_LAYER_1_8*N_LAYER_2_8];
    #pragma HLS ARRAY_PARTITION variable=layer11_out complete dim=0
    nnet::tanh<layer22_t, layer11_t, tanh_config11>(layer22_out, layer11_out); // activation
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer11_t>(layer11_out, "activation", N_LAYER_1_8*N_LAYER_2_8);
#endif

    layer23_t layer23_out[N_OUTPUTS_23*N_FILT_23];
    #pragma HLS ARRAY_PARTITION variable=layer23_out complete dim=0
    nnet::pointwise_conv_1d_cl<layer11_t, layer23_t, config23>(layer11_out, layer23_out, w23, b23); // dense_1
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer23_t>(layer23_out, "dense_1", N_OUTPUTS_23*N_FILT_23);
#endif

    layer15_t layer15_out[N_LAYER_1_12*N_LAYER_2_12];
    #pragma HLS ARRAY_PARTITION variable=layer15_out complete dim=0
    nnet::tanh<layer23_t, layer15_t, tanh_config15>(layer23_out, layer15_out); // activation_1
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer15_t>(layer15_out, "activation_1", N_LAYER_1_12*N_LAYER_2_12);
#endif

    layer24_t layer24_out[N_OUTPUTS_24*N_FILT_24];
    #pragma HLS ARRAY_PARTITION variable=layer24_out complete dim=0
    nnet::pointwise_conv_1d_cl<layer15_t, layer24_t, config24>(layer15_out, layer24_out, w24, b24); // met_weight
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer24_t>(layer24_out, "met_weight", N_OUTPUTS_24*N_FILT_24);
#endif

    layer20_t layer20_out[N_INPUT_1_19*N_INPUT_2_19];
    #pragma HLS ARRAY_PARTITION variable=layer20_out complete dim=0
    nnet::multiply<layer24_t, input19_t, layer20_t, config20>(layer24_out, input_pxpy, layer20_out); // multiply
#ifndef __SYNTHESIS__
    nnet::save_layer_output<layer20_t>(layer20_out, "multiply", N_INPUT_1_19*N_INPUT_2_19);
#endif

    nnet::global_pooling1d_cl<layer20_t, result_t, config21>(layer20_out, layer21_out); // output
#ifndef __SYNTHESIS__
    nnet::save_layer_output<result_t>(layer21_out, "output", N_FILT_21);
#endif

}
