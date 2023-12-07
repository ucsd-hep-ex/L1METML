import tensorflow
from models import dense_embedding
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.models import Model
import numpy as np
import hls4ml
import pandas as pd
from qkeras.utils import _add_supported_quantized_objects
from models import dense_embedding, dense_embedding_quantized
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

reuse_factor = 1
precision = 'ap_fixed<8,3>'
io_type = 'io_parallel'
strategy = 'Latency'
output_dir = 'hls_output_{}_{}_rf{}'.format(io_type, strategy, reuse_factor)
batch_size = 1

# check everthing works
model.summary()
model.save('{}/model.h5'.format(output_dir))

config = hls4ml.utils.config_from_keras_model(model, granularity='name',
                                              default_reuse_factor=reuse_factor, default_precision=precision)
config['Model']['Strategy'] = strategy
config['LayerName']['input_cat0']['Precision']['result'] = 'ap_uint<4>'
config['LayerName']['input_cat1']['Precision']['result'] = 'ap_uint<4>'
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

# y_1_hls = hls_model.predict([X.astype(np.float32), X_cat0.astype(np.float32), X_cat1.astype(np.float32)])
# df = pd.DataFrame({'keras': y_1.flatten(), 'hls4ml': y_1_hls.flatten()})
# print(df)


hls_model.build(synth=True)
hls4ml.report.read_vivado_report(output_dir)
