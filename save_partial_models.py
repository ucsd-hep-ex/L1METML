import tensorflow
from models import dense_embedding
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.models import Model
import numpy as np

# to check if model is loadable
model = tensorflow.keras.models.load_model('output/model.h5', compile=False)

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
