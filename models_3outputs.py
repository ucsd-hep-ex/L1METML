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
    
    outputs0 = Dense(1, name = 'output0', activation='linear')(x)

    y = Dense(64, name = 'dense_4', activation='softmax')(inputs)
    y = Dropout(rate=0.1)(y)
    y = Dense(32, name = 'dense_5', activation='softmax')(y)
    y = Dropout(rate=0.1)(y)
    y = Dense(32, name = 'dense_6', activation='softmax')(y)
    y = Dropout(rate=0.1)(y)

    outputs1 = Dense(1, name = 'output1', activation='linear')(y)

    z = Dense(64, name = 'dense_7', activation='softmax')(inputs)
    z = Dropout(rate=0.1)(z)
    z = Dense(32, name = 'dense_8', activation='softmax')(z)
    z = Dropout(rate=0.1)(z)
    z = Dense(32, name = 'dense_9', activation='softmax')(z)
    z = Dropout(rate=0.1)(z)

    outputs2 = Dense(1, name = 'output2', activation='linear')(z)

    keras_model = Model(inputs=inputs, outputs=[outputs0, outputs1, outputs2])

    return keras_model

