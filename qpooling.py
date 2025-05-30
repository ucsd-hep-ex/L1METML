'''
Implementing quantized pooling layer in 1D 

Author: Daniel Primosch
'''

import numpy as np
from tensorflow.keras import constraints

import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.layers import AveragePooling1D
from tensorflow.keras.layers import GlobalAveragePooling1D
from qkeras.qlayers import QActivation
from qkeras.quantizers import get_quantizer

class QGlobalAveragePooling1D(GlobalAveragePooling1D):
    """Computes the quantized version of GlobalAveragePooling1D."""

    def __init__(self, data_format=None,
                 average_quantizer=None,
                 activation=None,
                 **kwargs):
        
        self.average_quantizer = average_quantizer
        self.average_quantizer_internal = get_quantizer(self.average_quantizer)
        self.quantizers = [self.average_quantizer_internal]

        if activation is not None:
            self.activation = get_quantizer(activation)
        else:
            self.activation = None
        
        super().__init__(data_format=data_format, **kwargs)
        
    def compute_pooling_len(self, input_shape):
        if not isinstance(input_shape, tuple):
            input_shape = input_shape.as_list()
        if self.data_format == 'channels_last':
            return input_shape[1]
        else:
            return input_shape[2]
    
    def call(self, inputs):
        """Performs quantized GlobalAveragePooling1D followed by QActivation.
        
        Modeled after QGLobalAveragePooling2D in QKeras, adapted for 1D.

        Performs the pooling sum and then multipolies the sum with the 
        quantized inverse multiplication factor to get the average value.
        """

        if self.average_quantizer:
            #Calculates pooling sum
            if self.data_format == 'channels_last':
                x = K.sum(inputs, axis=[1], keepdims=self.keepdims)
            else:
                x = K.sum(inputs, axis=[2], keepdims=self.keepdims)
            
            # Calculates the pooling length
            pool_len = self.compute_pooling_len(inputs.shape)

            #Quantizes the inverse multiplication factor
            mult_factor = 1.0 / pool_len
            q_mult_factor = self.average_quantizer_internal(mult_factor)

            #Derives the average pooling value from pooling sum
            x = x * q_mult_factor

        else:
            # If quantizer is not available, calls the keras layer.
            x = super(QGlobalAveragePooling1D, self).call(inputs)
        
        if self.activation is not None:
            return self.activation(x)
        return x

    def get_config(self):
        config = {
            "average_quantizer": constraints.serialize(
                self.average_quantizer_internal
            ),
            "activation": constraints.serialize(
                self.activation
            ),
        }
        base_config = super(QGlobalAveragePooling1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def get_quantization_config(self):
        return {
            "average_quantizer":
                str(self.average_quantizer_internal),
            "activation": str(self.activation)
        }
    
    def get_quantizers(self):
        return self.quantizers