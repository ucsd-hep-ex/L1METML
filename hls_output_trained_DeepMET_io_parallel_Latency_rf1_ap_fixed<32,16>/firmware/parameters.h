#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "ap_fixed.h"
#include "ap_int.h"

#include "nnet_utils/nnet_code_gen.h"
#include "nnet_utils/nnet_helpers.h"
// hls-fpga-machine-learning insert includes
#include "nnet_utils/nnet_activation.h"
#include "nnet_utils/nnet_activation_stream.h"
#include "nnet_utils/nnet_conv1d.h"
#include "nnet_utils/nnet_embed.h"
#include "nnet_utils/nnet_embed_stream.h"
#include "nnet_utils/nnet_merge.h"
#include "nnet_utils/nnet_merge_stream.h"
#include "nnet_utils/nnet_pooling.h"
#include "nnet_utils/nnet_pooling_stream.h"
#include "nnet_utils/nnet_sepconv1d_stream.h"

// hls-fpga-machine-learning insert weights
#include "weights/e3.h"
#include "weights/e4.h"
#include "weights/w22.h"
#include "weights/b22.h"
#include "weights/w23.h"
#include "weights/b23.h"
#include "weights/w24.h"
#include "weights/b24.h"

// hls-fpga-machine-learning insert layer-config
// embedding0
struct config3 : nnet::embed_config {
    static const unsigned n_in = 100;
    static const unsigned n_out = 2;
    static const unsigned vocab_size = 6;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef embedding0_embeddings_t embeddings_t;
};

// embedding1
struct config4 : nnet::embed_config {
    static const unsigned n_in = 100;
    static const unsigned n_out = 2;
    static const unsigned vocab_size = 4;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef embedding1_embeddings_t embeddings_t;
};

// concatenate
struct config6 : nnet::concat_config {
    static const unsigned n_elem1_0 = 100;
    static const unsigned n_elem1_1 = 2;
    static const unsigned n_elem1_2 = 0;
    static const unsigned n_elem2_0 = 100;
    static const unsigned n_elem2_1 = 2;
    static const unsigned n_elem2_2 = 0;

    static const int axis = -1;
};

// concatenate_1
struct config7 : nnet::concat_config {
    static const unsigned n_elem1_0 = 100;
    static const unsigned n_elem1_1 = 4;
    static const unsigned n_elem1_2 = 0;
    static const unsigned n_elem2_0 = 100;
    static const unsigned n_elem2_1 = 4;
    static const unsigned n_elem2_2 = 0;

    static const int axis = -1;
};

// dense
struct config22_mult : nnet::dense_config {
    static const unsigned n_in = 8;
    static const unsigned n_out = 12;
    static const unsigned reuse_factor = 1;
    static const unsigned strategy = nnet::latency;
    static const unsigned n_zeros = 0;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef model_default_t accum_t;
    typedef dense_bias_t bias_t;
    typedef dense_weight_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config22 : nnet::conv1d_config {
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_width = 100;
    static const unsigned n_chan = 8;
    static const unsigned filt_width = 1;
    static const unsigned kernel_size = filt_width;
    static const unsigned n_filt = 12;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;
    static const unsigned out_width = 100;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_width = 100;
    static const ap_uint<filt_width> pixels[min_width];
    static const unsigned n_partitions = 100;
    static const unsigned n_pixels = out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::fill_buffer_22<data_T, CONFIG_T>;
    typedef model_default_t accum_t;
    typedef dense_bias_t bias_t;
    typedef dense_weight_t weight_t;
    typedef config22_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index = nnet::scale_index_unscaled<K, S, W>;
};
const ap_uint<config22::filt_width> config22::pixels[] = {0};

// activation
struct tanh_config11 : nnet::activ_config {
    static const unsigned n_in = 1200;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef activation_table_t table_t;
};

// dense_1
struct config23_mult : nnet::dense_config {
    static const unsigned n_in = 12;
    static const unsigned n_out = 36;
    static const unsigned reuse_factor = 1;
    static const unsigned strategy = nnet::latency;
    static const unsigned n_zeros = 0;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef model_default_t accum_t;
    typedef dense_1_bias_t bias_t;
    typedef dense_1_weight_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config23 : nnet::conv1d_config {
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_width = 100;
    static const unsigned n_chan = 12;
    static const unsigned filt_width = 1;
    static const unsigned kernel_size = filt_width;
    static const unsigned n_filt = 36;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;
    static const unsigned out_width = 100;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_width = 100;
    static const ap_uint<filt_width> pixels[min_width];
    static const unsigned n_partitions = 100;
    static const unsigned n_pixels = out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::fill_buffer_23<data_T, CONFIG_T>;
    typedef model_default_t accum_t;
    typedef dense_1_bias_t bias_t;
    typedef dense_1_weight_t weight_t;
    typedef config23_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index = nnet::scale_index_unscaled<K, S, W>;
};
const ap_uint<config23::filt_width> config23::pixels[] = {0};

// activation_1
struct tanh_config15 : nnet::activ_config {
    static const unsigned n_in = 3600;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef activation_1_table_t table_t;
};

// met_weight
struct config24_mult : nnet::dense_config {
    static const unsigned n_in = 36;
    static const unsigned n_out = 1;
    static const unsigned reuse_factor = 1;
    static const unsigned strategy = nnet::latency;
    static const unsigned n_zeros = 0;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef model_default_t accum_t;
    typedef met_weight_bias_t bias_t;
    typedef met_weight_weight_t weight_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config24 : nnet::conv1d_config {
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_width = 100;
    static const unsigned n_chan = 36;
    static const unsigned filt_width = 1;
    static const unsigned kernel_size = filt_width;
    static const unsigned n_filt = 1;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;
    static const unsigned out_width = 100;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_width = 100;
    static const ap_uint<filt_width> pixels[min_width];
    static const unsigned n_partitions = 100;
    static const unsigned n_pixels = out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::fill_buffer_24<data_T, CONFIG_T>;
    typedef model_default_t accum_t;
    typedef met_weight_bias_t bias_t;
    typedef met_weight_weight_t weight_t;
    typedef config24_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index = nnet::scale_index_unscaled<K, S, W>;
};
const ap_uint<config24::filt_width> config24::pixels[] = {0};

// multiply
struct config20 : nnet::merge_config {
    static const unsigned n_elem = N_OUTPUTS_24*N_FILT_24;
};

// output
struct config21 : nnet::pooling1d_config {
    static const unsigned n_in = 100;
    static const unsigned n_filt = 2;
    static const nnet::Pool_Op pool_op = nnet::Average;
    static const unsigned reuse_factor = 1;
    typedef model_default_t accum_t;
};


#endif
