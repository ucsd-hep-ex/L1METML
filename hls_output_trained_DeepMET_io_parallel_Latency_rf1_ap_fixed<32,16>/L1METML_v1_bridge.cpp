#ifndef L1METML_V1_BRIDGE_H_
#define L1METML_V1_BRIDGE_H_

#include "firmware/L1METML_v1.h"
#include "firmware/nnet_utils/nnet_helpers.h"
#include <algorithm>
#include <map>

// hls-fpga-machine-learning insert bram

namespace nnet {
bool trace_enabled = false;
std::map<std::string, void *> *trace_outputs = NULL;
size_t trace_type_size = sizeof(double);
} // namespace nnet

extern "C" {

struct trace_data {
    const char *name;
    void *data;
};

void allocate_trace_storage(size_t element_size) {
    nnet::trace_enabled = true;
    nnet::trace_outputs = new std::map<std::string, void *>;
    nnet::trace_type_size = element_size;
    nnet::trace_outputs->insert(std::pair<std::string, void *>("embedding0", (void *) malloc(N_LAYER_1_3*N_LAYER_2_3 * element_size)));
    nnet::trace_outputs->insert(std::pair<std::string, void *>("embedding1", (void *) malloc(N_LAYER_1_4*N_LAYER_2_4 * element_size)));
    nnet::trace_outputs->insert(std::pair<std::string, void *>("concatenate", (void *) malloc(OUT_CONCAT_0_6*OUT_CONCAT_1_6 * element_size)));
    nnet::trace_outputs->insert(std::pair<std::string, void *>("concatenate_1", (void *) malloc(OUT_CONCAT_0_7*OUT_CONCAT_1_7 * element_size)));
    nnet::trace_outputs->insert(std::pair<std::string, void *>("dense", (void *) malloc(N_OUTPUTS_22*N_FILT_22 * element_size)));
    nnet::trace_outputs->insert(std::pair<std::string, void *>("activation", (void *) malloc(N_LAYER_1_8*N_LAYER_2_8 * element_size)));
    nnet::trace_outputs->insert(std::pair<std::string, void *>("dense_1", (void *) malloc(N_OUTPUTS_23*N_FILT_23 * element_size)));
    nnet::trace_outputs->insert(std::pair<std::string, void *>("activation_1", (void *) malloc(N_LAYER_1_12*N_LAYER_2_12 * element_size)));
    nnet::trace_outputs->insert(std::pair<std::string, void *>("met_weight", (void *) malloc(N_OUTPUTS_24*N_FILT_24 * element_size)));
    nnet::trace_outputs->insert(std::pair<std::string, void *>("multiply", (void *) malloc(N_INPUT_1_19*N_INPUT_2_19 * element_size)));
    nnet::trace_outputs->insert(std::pair<std::string, void *>("output", (void *) malloc(N_FILT_21 * element_size)));
}

void free_trace_storage() {
    for (std::map<std::string, void *>::iterator i = nnet::trace_outputs->begin(); i != nnet::trace_outputs->end(); i++) {
        void *ptr = i->second;
        free(ptr);
    }
    nnet::trace_outputs->clear();
    delete nnet::trace_outputs;
    nnet::trace_outputs = NULL;
    nnet::trace_enabled = false;
}

void collect_trace_output(struct trace_data *c_trace_outputs) {
    int ii = 0;
    for (std::map<std::string, void *>::iterator i = nnet::trace_outputs->begin(); i != nnet::trace_outputs->end(); i++) {
        c_trace_outputs[ii].name = i->first.c_str();
        c_trace_outputs[ii].data = i->second;
        ii++;
    }
}

// Wrapper of top level function for Python bridge
void L1METML_v1_float(
    float input_cont[N_INPUT_1_5*N_INPUT_2_5], float input_pxpy[N_INPUT_1_19*N_INPUT_2_19], float input_cat0[N_INPUT_1_1], float input_cat1[N_INPUT_1_2],
    float layer21_out[N_FILT_21]
) {

    input5_t input_cont_ap[N_INPUT_1_5*N_INPUT_2_5];
    nnet::convert_data<float, input5_t, N_INPUT_1_5*N_INPUT_2_5>(input_cont, input_cont_ap);
    input19_t input_pxpy_ap[N_INPUT_1_19*N_INPUT_2_19];
    nnet::convert_data<float, input19_t, N_INPUT_1_19*N_INPUT_2_19>(input_pxpy, input_pxpy_ap);
    input_t input_cat0_ap[N_INPUT_1_1];
    nnet::convert_data<float, input_t, N_INPUT_1_1>(input_cat0, input_cat0_ap);
    input2_t input_cat1_ap[N_INPUT_1_2];
    nnet::convert_data<float, input2_t, N_INPUT_1_2>(input_cat1, input_cat1_ap);

    result_t layer21_out_ap[N_FILT_21];

    L1METML_v1(input_cont_ap,input_pxpy_ap,input_cat0_ap,input_cat1_ap,layer21_out_ap);

    nnet::convert_data<result_t, float, N_FILT_21>(layer21_out_ap, layer21_out);
}

void L1METML_v1_double(
    double input_cont[N_INPUT_1_5*N_INPUT_2_5], double input_pxpy[N_INPUT_1_19*N_INPUT_2_19], double input_cat0[N_INPUT_1_1], double input_cat1[N_INPUT_1_2],
    double layer21_out[N_FILT_21]
) {
    input5_t input_cont_ap[N_INPUT_1_5*N_INPUT_2_5];
    nnet::convert_data<double, input5_t, N_INPUT_1_5*N_INPUT_2_5>(input_cont, input_cont_ap);
    input19_t input_pxpy_ap[N_INPUT_1_19*N_INPUT_2_19];
    nnet::convert_data<double, input19_t, N_INPUT_1_19*N_INPUT_2_19>(input_pxpy, input_pxpy_ap);
    input_t input_cat0_ap[N_INPUT_1_1];
    nnet::convert_data<double, input_t, N_INPUT_1_1>(input_cat0, input_cat0_ap);
    input2_t input_cat1_ap[N_INPUT_1_2];
    nnet::convert_data<double, input2_t, N_INPUT_1_2>(input_cat1, input_cat1_ap);

    result_t layer21_out_ap[N_FILT_21];

    L1METML_v1(input_cont_ap,input_pxpy_ap,input_cat0_ap,input_cat1_ap,layer21_out_ap);

    nnet::convert_data<result_t, double, N_FILT_21>(layer21_out_ap, layer21_out);
}
}

#endif
