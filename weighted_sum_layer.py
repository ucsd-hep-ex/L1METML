import tensorflow as tf
from tensorflow.keras.layers import Layer

class weighted_sum_layer(Layer):
    '''Either does weight times inputs
    or weight times inputs + bias
    Input to be provided as:
      - Weights
      - ndim biases (if applicable)
      - ndim items to sum
    Currently works for 3-dim input, summing over the 2nd axis'''
    def __init__(self, ndim=2, with_bias=False, **kwargs):
        super(weighted_sum_layer, self).__init__(**kwargs)
        self.with_bias = with_bias
        self.ndim = ndim

    def get_config(self):
        cfg = super(weighted_sum_layer, self).get_config()
        cfg['ndim'] = self.ndim
        cfg['with_bias'] = self.with_bias
        return cfg

    def compute_output_shape(self, input_shape):
        assert input_shape[2] > 1 + self.with_bias * self.ndim
        inshape = list(input_shape)
        out_shape_1 = input_shape[2] - self.ndim - 1 if self.with_bias else input_shape[2]-1
        return tuple((inshape[0], out_shape_1))

    def call(self, inputs):
        # input #B x E x F
        weights = inputs[:, :, 0:1] - 1.  # B x E x 1
        if not self.with_bias:
            tosum = inputs[:, :, 1:]  # B x E x F-1
            weighted = weights * tosum  # broadcast to B x E x F-1
        else:
            tosum = inputs[:, :, self.ndim+1:]  # B x E x F-1
            biases = inputs[:, :, 1:self.ndim+1]
            weighted = weights * (biases + tosum)  # broadcast to B x E x F-1
        return tf.reduce_sum(weighted, axis=1)
