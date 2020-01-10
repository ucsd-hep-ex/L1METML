import keras
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Dropout, Lambda
from keras.backend import slice

def dense(ninputs, noutputs):

    inputs = Input(shape=(ninputs,), name = 'input')  
    #x = BatchNormalization(name='bn_1')(inputs)
    x = Dense(64, name = 'dense_1', activation='relu')(inputs)
    x = Dropout(rate=0.1)(x)
    x = Dense(32, name = 'dense_2', activation='relu')(x)
    x = Dropout(rate=0.1)(x)
    x = Dense(32, name = 'dense_3', activation='relu')(x)
    x = Dropout(rate=0.1)(x)
    
    outputs = Dense(noutputs, name = 'output', activation='linear')(x)

    outputs0 = Lambda(lambda x: slice(x, (0, 0), (-1,  1)))(outputs)
    outputs1 = Lambda(lambda x: slice(x, (0, 1), (-1, -1)))(outputs)

    outputs_softmax = Dense(2, name = 'output_softmax', activation='softmax')(outputs1)

    outputs_softmax0 = Lambda(lambda x: slice(x, (0, 0), (-1, 1)))(outputs_softmax)
    outputs_softmax1 = Lambda(lambda x: slice(x, (0, 1), (-1, -1)))(outputs_softmax)

    keras_model = Model(inputs=inputs, outputs=[outputs0, outputs_softmax0, outputs_softmax1])

    return keras_model

