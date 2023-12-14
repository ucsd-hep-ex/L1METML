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
model = tensorflow.keras.models.load_model('models/baseline_DeepMET/trained_DeepMET.h5', compile=False, custom_objects=co)
# model = tensorflow.keras.models.load_model('models/baseline_DeepMET_quantized/baseline_DeepMET_quantized.h5', compile=False, custom_objects=co)

reuse_factor = 1
precision = 'ap_fixed<16,6>'
io_type = 'io_parallel'
strategy = 'Latency'
output_dir = 'hls_output_{}_{}_rf{}_{}'.format(io_type, strategy, reuse_factor, precision)
batch_size = 1
synth = False

# check everthing works
model.summary()
model.save('{}/model.h5'.format(output_dir))

config = hls4ml.utils.config_from_keras_model(model, granularity='name',
                                              default_reuse_factor=reuse_factor, default_precision=precision)
config['Model']['Strategy'] = strategy
config['LayerName']['input_cat0']['Precision']['result'] = 'ap_uint<4>'
config['LayerName']['input_cat1']['Precision']['result'] = 'ap_uint<4>'
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
X = f['X'][:]
y = -f['Y'][:]

# preprocessing
X_pre = list(preProcessing(X, normFac=1))
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
