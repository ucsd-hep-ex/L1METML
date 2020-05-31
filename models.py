import keras
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Dropout, Lambda
from keras.backend import slice

def dense(ninputs, noutputs):

    inputs = Input(shape=(ninputs,), name = 'input')


    x = BatchNormalization(name='bn_1')(inputs)
    x = Dense(64, name = 'dense_1', activation='relu', kernel_initializer='glorot_uniform')(x)
    x = Dropout(rate=0.1)(x)


    x = Dense(32, name = 'dense_2', activation='relu')(x)
    x = Dropout(rate=0.1)(x)


    x = Dense(32, name = 'dense_3', activation='relu')(x)
    x = Dropout(rate=0.1)(x)

    
    outputs = Dense(noutputs, name = 'output', activation='linear')(x)

    outputs0 = Lambda(lambda x: slice(x, (0, 0), (-1,  1)))(outputs)
    outputs1 = Lambda(lambda x: slice(x, (0, 1), (-1, -1)))(outputs)

    keras_model = Model(inputs=inputs, outputs=[outputs0, outputs1])

    return keras_model

