import tensorflow
from weighted_sum_layer import weighted_sum_layer
from models import dense_embedding
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.models import Model
import numpy as np
co = {'weighted_sum_layer':weighted_sum_layer}

# initialize model with random weights
model = dense_embedding(n_features=4, n_features_cat=2, n_dense_layers=3, activation='tanh', 
                        number_of_pupcandis=100, embedding_input_dim={0: 3, 1: 9}, emb_out_dim=8, 
                        with_bias=False, t_mode=1)
# save model to h5
model.save('model_deepmet_new.h5')

# write json just for fun
with open('model_deepmet_new.json', "w") as json_file:
    json_file.write(model.to_json())

# to check if model is loadable
model = tensorflow.keras.models.load_model('model_deepmet_new.h5', custom_objects=co)

# check everthing works
model.summary()

# now let's break up the model
# save a dictionary of the model
model_dict = {layer.name: layer for layer in model.layers}

# first part of the model
partial_model_1 = Model(inputs=model.inputs,
                        outputs=model_dict['dense'].input)
partial_model_1.summary()

# second part of the model (to implement in hls4ml)
input_layer = Input(model_dict['dense'].input.shape[1:])
x = model_dict['dense'](input_layer)
x = model_dict['batch_normalization'](x)
x = model_dict['dense_1'](x)
x = model_dict['batch_normalization_1'](x)
x = model_dict['dense_2'](x)
x = model_dict['batch_normalization_2'](x)
output_layer = model_dict['dense_3'](x)
partial_model_2 = Model(inputs=input_layer, outputs=output_layer)
partial_model_2.summary()

# third part of the model
input_layer_1 = Input(model_dict['input'].input.shape[1:])
input_layer_2 = Input(model_dict['dense_3'].output.shape[1:])
x = model_dict['lambda'](input_layer_1)
x = model_dict['concatenate_1']([input_layer_2, x])
output_layer = model_dict['output'](x)
partial_model_3 = Model(inputs=[input_layer_1, input_layer_2], outputs=output_layer)
partial_model_3.summary()


# let's check the partial models give the same answer as the original model
# random inputs
X = np.random.rand(2, 100, 4)
X_cat0 = np.random.randint(0, 3, size=(2, 100, 1))
X_cat1 = np.random.randint(0, 9, size=(2, 100, 1))
print('input shapes')
print(X.shape)
print(X_cat0.shape)
print(X_cat1.shape)

# original output
y = model.predict([X, X_cat0, X_cat1])
print('original output')
print(y)

# partial ouputs
y_1 = partial_model_1.predict([X, X_cat0, X_cat1])
y_2 = partial_model_2.predict(y_1)
y_3 = partial_model_3.predict([X, y_2])
print('partial outputs')
print(y_3)

# check if they're equal (error if they're not!)
np.testing.assert_array_equal(y, y_3)
