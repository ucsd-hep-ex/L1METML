import tensorflow
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.models import Model
import numpy as np
import hls4ml
import pandas as pd
from qkeras.utils import _add_supported_quantized_objects
from utils import preProcessing
import h5py
import scipy

co = {}
_add_supported_quantized_objects(co)

print(co)

# load full model:
model_name   = "trained_quantized_DeepMET_normfac1000_128_1layer"
model_path   = "./test/model.h5"
model        = tensorflow.keras.models.load_model(model_path,
                                                  compile=False,
                                                  custom_objects=co
                                                  )

total_bits   = 8
int_bits     = 2
config_options = {
        'granularity': 'name',
        'default_reuse_factor': 3,
        'default_precision': 'ap_fixed<{},{}>'.format(total_bits,int_bits)
        }

build_option = {
        'csim': False,      # C Simulation
        'synth': True,      # Synthesis
        'export': False,    # Export
        'cosim': False,     # C/RTL Co-simulation
        'validation': False # Validation
        }

config = hls4ml.utils.config_from_keras_model(model,**config_options)

strategy     = 'Latency'
config['Model']['Strategy'] = strategy

for name in config['LayerName'].keys():
    print(name,config['LayerName'][name].keys())

trace        = True
for name in config['LayerName'].keys():
    config['LayerName'][name]['Trace'] = trace
config['LayerName']['input_cat0']['Precision']['result'] = 'ap_uint<4>'
config['LayerName']['input_cat1']['Precision']['result'] = 'ap_uint<4>'
config['LayerName']['multiply']['n_elem'] = 128
config['LayerName']['output']['n_filt'] = 2

config["LayerName"]["q_dense"]["ConvImplementation"] = "Pointwise"
config["LayerName"]["met_weight"]["ConvImplementation"] = "Pointwise"

convert_options = {
    'hls_config': config,                # The configuration generated from the Keras model
    'io_type': 'io_parallel',              # I/O interface type
    'part': 'xcvu13p-flga2577-2-e',      # FPGA part number
    'clock_period': 2.7,                 # Clock period in nanoseconds
    'project_name': 'test',              # Project name
    'backend': 'Vitis'                   # Backend to use (Vitis in this case)
    }

output_dir = "_".join([model_name,
    convert_options['io_type'],
    strategy,
    str(config_options['default_reuse_factor']),
    "ap_fixed_{}_{}_test".format(total_bits,int_bits)])

model.summary()
model.save('{}/model.h5'.format(output_dir))

hls_model = hls4ml.converters.convert_from_keras_model(model,**convert_options,output_dir=output_dir)

hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file='{}/model_hls4ml.png'.format(output_dir))

hls_model.compile()

if build_option['synth']:
    hls_model.build(**build_option)
    hls4ml.report.read_vivado_report(output_dir)

f = h5py.File('../L1METML/data/test_data.h5')
# 1000 test events is good enough
X = f['X'][:1000]
y = -f['Y'][:1000]

normFac=1000

# preprocessing
X_pre = list(preProcessing(X, normFac=normFac))
X_pre = [np.ascontiguousarray(x) for x in X_pre]

y_pred = model.predict(X_pre)
y_hls = hls_model.predict(X_pre)

met = np.hypot(y[:, 0], y[:, 1])
met_pred = np.hypot(y_pred[:, 0], y_pred[:, 1]) * normFac
met_hls = np.hypot(y_hls[:, 0], y_hls[:, 1]) * normFac
met_pup_x = np.sum(X[:, :, 1], axis=-1)
met_pup_y = np.sum(X[:, :, 2], axis=-1)
met_pup = np.hypot(met_pup_x, met_pup_y)

import seaborn
import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame.from_dict({'Gen MET': met, 'PUPPI MET': met_pup, 'QKeras MET': met_pred, 'hls4ml MET': met_hls})
plt.figure()
seaborn.pairplot(df, corner=True)
plt.savefig(f'{output_dir}/profiling_MET.png', dpi=300)

df = pd.DataFrame.from_dict({'Gen MET x': y[:, 0], 'PUPPI MET x': met_pup_x, 'QKeras MET x': y_pred[:, 0], 'hls4ml MET x': y_hls[:, 0]})
plt.figure()
seaborn.pairplot(df, corner=True)
plt.savefig(f'{output_dir}/profiling_MET_x.png', dpi=300)

df = pd.DataFrame.from_dict({'Gen MET y': y[:, 1], 'PUPPI MET y': met_pup_y, 'QKeras MET y': y_pred[:, 1], 'hls4ml MET y': y_hls[:, 1]})
plt.figure()
seaborn.pairplot(df, corner=True)
plt.savefig(f'{output_dir}/profiling_MET_y.png', dpi=300)

response_pup = met_pup / met
response_pred = met_pred / met
response_hls = met_hls / met
bins = np.linspace(0, 2, 25)
plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
plt.hist(response_pup, bins=bins, label=f'PUPPI, median={np.median(response_pup):0.2f}, IQR={scipy.stats.iqr(response_pup):0.2f}')
