import tensorflow
from models import dense_embedding
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.models import Model
import numpy as np
import hls4ml
import pandas as pd
from qkeras.utils import _add_supported_quantized_objects
co = {}
_add_supported_quantized_objects(co)

def print_dict(d, indent=0):
    align=20
    for key, value in d.items():
        print('  ' * indent + str(key), end='')
        if isinstance(value, dict):
            print()
            print_dict(value, indent+1)
        else:
            print(':' + ' ' * (20 - len(key) - 2 * indent) + str(value))

model = tensorflow.keras.models.load_model('output/model.h5', compile=False, custom_objects=co)

# check everthing works
model.summary()

# now let's break up the model
# save a dictionary of the model
model_dict = {layer.name: layer for layer in model.layers}

# first part of the model
partial_model_1 = Model(inputs=model.inputs,
                        outputs=model_dict['conv1d'].input)
partial_model_1.summary()

# second part of the model (to implement in hls4ml)
input_layer = Input(model_dict['conv1d'].input.shape[1:])
x = model_dict['conv1d'](input_layer)
x = model_dict['batch_normalization'](x)
x = model_dict['conv1d_1'](x)
x = model_dict['batch_normalization_1'](x)
x = model_dict['conv1d_2'](x)
x = model_dict['batch_normalization_2'](x)
output_layer = model_dict['conv1d_3'](x)
partial_model_2 = Model(inputs=input_layer, outputs=output_layer)
partial_model_2.summary()

# third part of the model
input_layer_1 = Input(model_dict['input_pxpy'].input.shape[1:])
input_layer_2 = Input(model_dict['conv1d_3'].output.shape[1:])
x = model_dict['multiply']([input_layer_2, input_layer_1])
output_layer = model_dict['output'](x)
partial_model_3 = Model(inputs=[input_layer_1, input_layer_2], outputs=output_layer)
partial_model_3.summary()


# let's check the partial models give the same answer as the original model
# random inputs
X = np.random.rand(2, 100, 4)
Xp = np.random.rand(2, 100, 2)
X_cat0 = np.random.randint(0, 13, size=(2, 100, 1))
X_cat1 = np.random.randint(0, 3, size=(2, 100, 1))
print('input shapes')
print(X.shape)
print(X_cat0.shape)
print(X_cat1.shape)

# original output
y = model.predict([X, Xp, X_cat0, X_cat1])
print('original output')
print(y)

# partial ouputs
y_1 = partial_model_1.predict([X, Xp, X_cat0, X_cat1])
y_2 = partial_model_2.predict(y_1)
y_3 = partial_model_3.predict([Xp, y_2])
print('partial outputs')
print(y_3)

# check if they're equal (error if they're not!)
np.testing.assert_array_equal(y, y_3)

model_to_convert = partial_model_3
config = hls4ml.utils.config_from_keras_model(model_to_convert, granularity='name')

config['Model'] = {}
config['Model']['ReuseFactor'] = 1
config['Model']['Strategy'] = 'Latency'
config['Model']['Precision'] = 'ap_fixed<16,6>'
config['SkipOptimizers'] = ['optimize_pointwise_conv']
for layer in config['LayerName'].keys():
    config['LayerName'][layer]['Trace'] = True

cfg = hls4ml.converters.create_vivado_config(fpga_part='xc7z020clg400-1')
cfg['HLSConfig'] = config
cfg['IOType'] = 'io_parallel'
cfg['Backend'] = 'Vivado'
cfg['ClockPeriod'] = 10
cfg['KerasModel'] = model_to_convert
cfg['OutputDir'] = 'hls_output'

print("-----------------------------------")
print_dict(cfg)
print("-----------------------------------")

hls_model = hls4ml.converters.keras_to_hls(cfg)
hls_model.compile()

y_2_hls = hls_model.predict(y_1)
df = pd.DataFrame({'keras': y_2.flatten(), 'hls4ml': y_2_hls.flatten()})
print(df)
