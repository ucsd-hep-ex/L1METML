import keras
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Dropout

def dense(ninputs, noutputs):

    inputs = Input(shape=(ninputs,), name = 'input')  
    x = BatchNormalization(name='bn_1')(inputs)
    x = Dense(64, name = 'dense_1', activation='relu')(x)
    x = Dropout(rate=0.1)(x)
    x = Dense(32, name = 'dense_2', activation='relu')(x)
    x = Dropout(rate=0.1)(x)
    x = Dense(32, name = 'dense_3', activation='relu')(x)
    x = Dropout(rate=0.1)(x)
    
    outputs = Dense(noutputs, name = 'output', activation='linear')(x)
    keras_model = Model(inputs=inputs, outputs=outputs)

    return keras_model

