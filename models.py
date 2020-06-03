import keras
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Dropout, Lambda, Conv1D, SpatialDropout1D, Concatenate, Flatten
import keras.backend as K
from keras.backend import slice

def dense(ninputs, noutputs):

    inputs = Input(shape=(ninputs,), name = 'input')

    x = BatchNormalization(name='bn_1')(inputs)
    x = Dense(64, name = 'dense_1', activation='relu', kernel_initializer='glorot_uniform')(x)
    x = Dropout(rate=0.1)(x)

    x = Dense(32, name = 'dense_2', activation='relu', kernel_initializer='glorot_uniform')(x)
    x = Dropout(rate=0.1)(x)

    x = Dense(32, name = 'dense_3', activation='relu', kernel_initializer='glorot_uniform')(x)
    x = Dropout(rate=0.1)(x)
    
    outputs = Dense(noutputs, name = 'output', activation='linear')(x)

    outputs0 = Lambda(lambda x: slice(x, (0, 0), (-1,  1)))(outputs)
    outputs1 = Lambda(lambda x: slice(x, (0, 1), (-1, -1)))(outputs)

    keras_model = Model(inputs=inputs, outputs=[outputs0, outputs1])

    return keras_model

def dense_conv(ndenseinputs, nconvs, nconvinputs, noutputs):

    inputs_dense = Input(shape=(ndenseinputs,), name = 'input_dense')
    inputs_conv = Input(shape=(nconvs, nconvinputs,), name = 'input_conv')

    x = BatchNormalization(name='bn_1')(inputs_dense)
    x = Dense(64, name = 'dense_1', activation='relu', kernel_initializer='glorot_uniform')(x)
    x = Dropout(rate=0.1)(x)

    x = Dense(32, name = 'dense_2', activation='relu', kernel_initializer='glorot_uniform')(x)
    x = Dropout(rate=0.1)(x)

    x = Dense(32, name = 'dense_3', activation='relu', kernel_initializer='glorot_uniform')(x)
    x = Dropout(rate=0.1)(x)

    y = BatchNormalization(name='bn_2')(inputs_conv)
    y = Conv1D(filters=32, kernel_size=(1,), strides=(1,), padding='same', 
               kernel_initializer='he_normal', use_bias=False, name='conv1',
               activation = 'relu')(y)
    y = SpatialDropout1D(rate=0.1)(y)
    y = Conv1D(filters=32, kernel_size=(1,), strides=(1,), padding='same', 
               kernel_initializer='he_normal', use_bias=False, name='conv2',
               activation = 'relu')(y)
    y = SpatialDropout1D(rate=0.1)(y)
    # Try GRU
    #y = GRU(50,go_backwards=True,implementation=2,name='gru')(y)
    # Try summing over jets
    y = Lambda(lambda x: K.mean(x, axis=-2), input_shape=(nconvs,32)) (y)
    # Try flattening all jets
    #y = Flatten()(y)
    
    concat = Concatenate()([x,y])
    z = Dense(100, name = 'dense_4', activation='relu', kernel_initializer='glorot_uniform')(concat)
    outputs = Dense(noutputs, name = 'output', activation='linear')(z)

    outputs0 = Lambda(lambda x: slice(x, (0, 0), (-1,  1)))(outputs)
    outputs1 = Lambda(lambda x: slice(x, (0, 1), (-1, -1)))(outputs)

    keras_model = Model(inputs=[inputs_dense,inputs_conv], outputs=[outputs0, outputs1])

    return keras_model

