import tensorflow
from models import dense_embedding
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.models import Model
import numpy as np
import hls4ml
import pandas as pd
from qkeras.utils import _add_supported_quantized_objects
from models import dense_embedding, dense_embedding_quantized
from utils import preProcessing
import h5py

co = {}
_add_supported_quantized_objects(co)


def print_dict(d, indent=0):
    align = 20
    for key, value in d.items():
        print('  ' * indent + str(key), end='')
        if isinstance(value, dict):
            print()
            print_dict(value, indent+1)
        else:
            print(':' + ' ' * (20 - len(key) - 2 * indent) + str(value))


# load full model:
model = tensorflow.keras.models.load_model('models/baseline_DeepMET_quantized/trained_quantized_DeepMET_normfac1000.h5', compile=False, custom_objects=co)
# model = tensorflow.keras.models.load_model('models/baseline_DeepMET_quantized/baseline_DeepMET_quantized.h5', compile=False, custom_objects=co)

reuse_factor = 1
precision = 'ap_fixed<32,16>'
io_type = 'io_parallel'
strategy = 'Latency'
output_dir = 'hls_output_{}_{}_rf{}_{}'.format(io_type, strategy, reuse_factor, precision)
batch_size = 1
synth = False
trace = True
normFac = 1000

# check everthing works
model.summary()
model.save('{}/model.h5'.format(output_dir))

config = hls4ml.utils.config_from_keras_model(model, 
                                              granularity='name',
                                              default_reuse_factor=reuse_factor, 
                                              default_precision=precision)
config['Model']['Strategy'] = strategy
for name in config['LayerName'].keys():
    config['LayerName'][name]['Trace'] = trace
config['LayerName']['input_cat0']['Precision']['result'] = 'ap_uint<4>'
config['LayerName']['input_cat1']['Precision']['result'] = 'ap_uint<4>'
#config['LayerName']['input_cont']['Precision']['result'] = 'ap_fixed<20,10>'
#config['LayerName']['q_dense']['Precision']['accum'] = 'ap_fixed<32,16>'
config['LayerName']['q_dense']['Precision']['weight'] = 'ap_fixed<32,16>'
config['LayerName']['q_dense']['Precision']['bias'] = 'ap_fixed<32,16>'
#config['LayerName']['q_dense_1']['Precision']['accum'] = 'ap_fixed<32,16>'
#config['LayerName']['q_dense_1']['Precision']['weight'] = 'ap_fixed<32,16>'
#config['LayerName']['q_dense_1']['Precision']['bias'] = 'ap_fixed<32,16>'
config['LayerName']['multiply']['n_elem'] = 100
config['LayerName']['output']['n_filt'] = 2
# skip optimize_pointwise_conv
# config['SkipOptimizers'] = ['optimize_pointwise_conv']
# for layer in config['LayerName'].keys():
#    config['LayerName'][layer]['Trace'] = True

print("-----------------------------------")
print_dict(config)
print("-----------------------------------")
hls_model = hls4ml.converters.convert_from_keras_model(model,
                                                       hls_config=config,
                                                       io_type=io_type,
                                                       output_dir=output_dir,
                                                       part='xcvu13p-flga2577-2-e',
                                                       clock_period=5)
hls_model.compile()

hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file='{}/model_hls4ml.png'.format(output_dir))

if synth:
    hls_model.build(synth=synth)
    hls4ml.report.read_vivado_report(output_dir)

f = h5py.File('data/test_data.h5')
# 1000 test events is good enough
X = f['X'][:1000]
y = -f['Y'][:1000]

# preprocessing
X_pre = list(preProcessing(X, normFac=normFac))
X_pre = [np.ascontiguousarray(x) for x in X_pre]

y_pred = model.predict(X_pre)
y_hls = hls_model.predict(X_pre)

met = np.hypot(y[:, 0], y[:, 1])
met_pred = np.hypot(y_pred[:, 0], y_pred[:, 1])
met_hls = np.hypot(y_hls[:, 0], y_hls[:, 1])

import seaborn
import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame.from_dict({'Gen MET': met, 'QKeras MET': met_pred, 'hls4ml MET': met_hls})
plt.figure()
seaborn.pairplot(df)
plt.savefig(f'{output_dir}/profiling_MET.png', dpi=300)

df = pd.DataFrame.from_dict({'Gen MET x': y[:, 0], 'QKeras MET x': y_pred[:, 0], 'hls4ml MET x': y_hls[:, 0]})
plt.figure()
seaborn.pairplot(df)
plt.savefig(f'{output_dir}/profiling_MET_x.png', dpi=300)

df = pd.DataFrame.from_dict({'Gen MET x': y[:, 1], 'QKeras MET x': y_pred[:, 1], 'hls4ml MET x': y_hls[:, 1]})
plt.figure()
seaborn.pairplot(df)
plt.savefig(f'{output_dir}/profiling_MET_y.png', dpi=300)


y_hls, hls4ml_trace = hls_model.trace(X_pre)
keras_trace = hls4ml.model.profiling.get_ymodel_keras(model, X_pre)

for layer in hls4ml_trace.keys():
    plt.figure()
    if layer not in keras_trace: continue
    plt.scatter(hls4ml_trace[layer].flatten(), keras_trace[layer].flatten(), s=0.2)
    min_x = min(np.amin(hls4ml_trace[layer]), np.amin(keras_trace[layer]))
    max_x = max(np.amax(hls4ml_trace[layer]), np.amax(keras_trace[layer]))
    plt.plot([min_x, max_x], [min_x, max_x], c='gray')
    plt.xlabel(f'hls4ml {layer}')
    plt.ylabel(f'QKeras {layer}')
    plt.savefig(f'{output_dir}/profiling_{layer}.png', dpi=300)
