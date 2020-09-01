import keras
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Dropout, Lambda, Conv1D, SpatialDropout1D, Concatenate, Flatten
import keras.backend as K
from keras.backend import slice

def dense(ninputs, noutputs):

    inputs = Input(shape=(ninputs,), name = 'input')

    x = BatchNormalization(name='bn_1')(inputs)
    x = Dense(128, name = 'dense_1', activation='relu', kernel_initializer='glorot_uniform')(x)

    x = Dense(64, name = 'dense_2', activation='relu', kernel_initializer='glorot_uniform')(x)

    x = Dense(64, name = 'dense_3', activation='relu', kernel_initializer='glorot_uniform')(x)

    outputs = Dense(noutputs, name = 'output', activation='linear')(x)

    outputs0 = Lambda(lambda x: slice(x, (0, 0), (-1,  1)))(outputs)
    outputs1 = Lambda(lambda x: slice(x, (0, 1), (-1, -1)))(outputs)

    keras_model = Model(inputs=inputs, outputs=[outputs0, outputs1])

    return keras_model

def dense_conv(ndenseinputs, nconvs, nconvinputs, noutputs):

    inputs_dense = Input(shape=(ndenseinputs,), name = 'input_dense')
    inputs_conv = Input(shape=(nconvs, nconvinputs), name = 'input_conv')

    x = BatchNormalization(name='bn_1')(inputs_dense)
    x = Dense(64, name = 'dense_1', activation='relu', kernel_initializer='glorot_uniform')(x)

    x = Dense(32, name = 'dense_2', activation='relu', kernel_initializer='glorot_uniform')(x)

    x = Dense(32, name = 'dense_3', activation='relu', kernel_initializer='glorot_uniform')(x)

    y= BatchNormalization(name='bn_2')(inputs_conv)
    y = Conv1D(filters=64, kernel_size=(3,), strides=(1,), padding='same', 
               kernel_initializer='he_normal', use_bias=True, name='conv_1',
               activation = 'relu')(y)
    y = Conv1D(filters=32, kernel_size=(3,), strides=(1,), padding='same', 
               kernel_initializer='he_normal', use_bias=True, name='conv_2',
               activation = 'relu')(y)
    y = Conv1D(filters=32, kernel_size=(3,), strides=(1,), padding='same', 
               kernel_initializer='he_normal', use_bias=True, name='conv_3',
               activation = 'relu')(y)

    # Try GRU
    #y = GRU(50,go_backwards=True,implementation=2,name='gru')(y)
    # Try summing over jets
    y = Lambda(lambda x: K.mean(x, axis=-2), input_shape=(nconvs,32)) (y)
    # Try flattening all jets
    #y = Flatten()(y)
    
    concat = Concatenate()([x,y])
    z = Dense(128, name = 'dense_4', activation='relu', kernel_initializer='glorot_uniform')(concat)
    z = Dense(128, name = 'dense_5', activation='relu', kernel_initializer='glorot_uniform')(z)
    outputs = Dense(noutputs, name = 'output', activation='linear')(z)

    outputs0 = Lambda(lambda x: slice(x, (0, 0), (-1,  1)))(outputs)
    outputs1 = Lambda(lambda x: slice(x, (0, 1), (-1, -1)))(outputs)

    keras_model = Model(inputs=[inputs_dense,inputs_conv], outputs=[outputs0, outputs1])

    return keras_model


def dense_conv_all(ndenseinputs, nconvs_1, nconvinputs_1, nconvs_2, nconvinputs_2, noutputs):

    inputs_dense = Input(shape=(ndenseinputs,), name = 'input_dense')
    inputs_conv_1 = Input(shape=(nconvs_1, nconvinputs_1,), name = 'input_conv_1')
    inputs_conv_2 = Input(shape=(nconvs_2, nconvinputs_2,), name = 'input_conv_2')

    x = BatchNormalization(name='bn_1')(inputs_dense)
    x = Dense(64, name = 'dense_1', activation='relu', kernel_initializer='glorot_uniform')(x)

    x = Dense(32, name = 'dense_2', activation='relu', kernel_initializer='glorot_uniform')(x)

    x = Dense(32, name = 'dense_3', activation='relu', kernel_initializer='glorot_uniform')(x)

    y_1 = BatchNormalization(name='bn_2')(inputs_conv_1)
    y_1 = Conv1D(filters=64, kernel_size=(3,), strides=(1,), padding='same', 
               kernel_initializer='he_normal', use_bias=True, name='conv_11',
               activation = 'relu')(y_1)
    y_1 = Conv1D(filters=32, kernel_size=(3,), strides=(1,), padding='same', 
               kernel_initializer='he_normal', use_bias=True, name='conv_12',
               activation = 'relu')(y_1)
    y_1 = Conv1D(filters=32, kernel_size=(3,), strides=(1,), padding='same', 
               kernel_initializer='he_normal', use_bias=True, name='conv_13',
               activation = 'relu')(y_1)

    y_2 = BatchNormalization(name='bn_3')(inputs_conv_2)
    y_2 = Conv1D(filters=64, kernel_size=(3,), strides=(1,), padding='same', 
               kernel_initializer='he_normal', use_bias=True, name='conv_21',
               activation = 'relu')(y_2)
    y_2 = Conv1D(filters=32, kernel_size=(3,), strides=(1,), padding='same', 
               kernel_initializer='he_normal', use_bias=True, name='conv_22',
               activation = 'relu')(y_2)
    y_2 = Conv1D(filters=32, kernel_size=(3,), strides=(1,), padding='same', 
               kernel_initializer='he_normal', use_bias=True, name='conv_23',
               activation = 'relu')(y_2)

    # Try GRU
    #y = GRU(50,go_backwards=True,implementation=2,name='gru')(y)
    # Try summing over jets
    y_1 = Lambda(lambda x: K.mean(x, axis=-2), input_shape=(nconvs_1,32)) (y_1)
    y_2 = Lambda(lambda x: K.mean(x, axis=-2), input_shape=(nconvs_2,32)) (y_2)
    # Try flattening all jets
    #y = Flatten()(y)
    
    concat = Concatenate()([x,y_1,y_2])
    z = Dense(128, name = 'dense_4', activation='relu', kernel_initializer='glorot_uniform')(concat)
    z = Dense(128, name = 'dense_5', activation='relu', kernel_initializer='glorot_uniform')(z)
    outputs = Dense(noutputs, name = 'output', activation='linear')(z)

    outputs0 = Lambda(lambda x: slice(x, (0, 0), (-1,  1)))(outputs)
    outputs1 = Lambda(lambda x: slice(x, (0, 1), (-1, -1)))(outputs)

    keras_model = Model(inputs=[inputs_dense,inputs_conv_1, inputs_conv_2], outputs=[outputs0, outputs1])

    return keras_model

def conv(nconvs, nconvinputs, noutputs):

    inputs_conv = Input(shape=(nconvs, nconvinputs,), name = 'input_conv')

    y = BatchNormalization(name='bn_2')(inputs_conv)
    y = Conv1D(filters=64, kernel_size=(3,), strides=(1,), padding='same', 
               kernel_initializer='he_normal', use_bias=True, name='conv_1',
               activation = 'relu')(y)
    y = Conv1D(filters=32, kernel_size=(3,), strides=(1,), padding='same', 
               kernel_initializer='he_normal', use_bias=True, name='conv_2',
               activation = 'relu')(y)
    y = Conv1D(filters=32, kernel_size=(3,), strides=(1,), padding='same', 
               kernel_initializer='he_normal', use_bias=True, name='conv_3',
               activation = 'relu')(y)

    y = Lambda(lambda x: K.mean(x, axis=-2), input_shape=(nconvs,32)) (y) 
                 
    y = Dense(128, name = 'dense_1', activation='relu', kernel_initializer='glorot_uniform')(y)
    y = Dense(128, name = 'dense_2', activation='relu', kernel_initializer='glorot_uniform')(y)

    outputs = Dense(noutputs, name = 'output', activation='linear')(y)

    outputs0 = Lambda(lambda x: slice(x, (0, 0), (-1,  1)))(outputs)
    outputs1 = Lambda(lambda x: slice(x, (0, 1), (-1, -1)))(outputs)

    keras_model = Model(inputs=inputs_conv, outputs=[outputs0, outputs1])

    return keras_model

# weight multiply bias sum
def weight_multiply_bias_sum(ip):
    w = slice(ip[0], (0, 0, 0), (-1, -1, 1))
    b = slice(ip[0], (0, 0, 1), (-1, -1, -1))
    x = slice(ip[1], (0, 0, 0), (-1, -1, 2))
    return K.sum(w*x+b,axis=-2)

def deepmetlike(nconvs, nconvinputs, noutputs):

    inputs_conv = Input(shape=(nconvs, nconvinputs,), name = 'input_conv')

    y = BatchNormalization(name='bn_1')(inputs_conv)
    y = Conv1D(filters=64, kernel_size=(1,), strides=(1,), padding='same', 
               kernel_initializer='he_normal', use_bias=True, name='conv_1',
               activation = 'relu')(y)
    y = BatchNormalization(name='bn_2')(y)
    y = Conv1D(filters=32, kernel_size=(1,), strides=(1,), padding='same', 
               kernel_initializer='he_normal', use_bias=True, name='conv_2',
               activation = 'relu')(y)
    y = BatchNormalization(name='bn_3')(y)
    y = Conv1D(filters=16, kernel_size=(1,), strides=(1,), padding='same', 
               kernel_initializer='he_normal', use_bias=True, name='conv_3',
               activation = 'relu')(y)
    y = BatchNormalization(name='bn_4')(y)
    y = Conv1D(filters=3, kernel_size=(1,), strides=(1,), padding='same', 
               kernel_initializer='he_normal', use_bias=True, name='conv_4',
               activation = 'linear')(y)

    # Try multiplying
    outputs = Lambda(weight_multiply_bias_sum) ([y,inputs_conv])
    
    outputs0 = Lambda(lambda x: slice(x, (0, 0), (-1,  1)))(outputs)
    outputs1 = Lambda(lambda x: slice(x, (0, 1), (-1, -1)))(outputs)

    keras_model = Model(inputs=inputs_conv, outputs=[outputs0, outputs1])

    return keras_model
