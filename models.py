from tkinter import FALSE
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, BatchNormalization, Dropout, Lambda, Conv1D, SpatialDropout1D, Concatenate, Flatten, Reshape, Multiply, Add, GlobalAveragePooling1D, Activation, Permute, Add, LeakyReLU, Layer, Average
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow import slice
from tensorflow.keras import initializers
import qkeras
from qkeras.qlayers import QDense, QActivation
import numpy as np
import itertools


def dense_embedding(n_features=6,
                    n_features_cat=2,
                    activation='relu',
                    number_of_pupcandis=100,
                    embedding_input_dim={0: 13, 1: 3},
                    emb_out_dim=8,
                    with_bias=True,
                    t_mode=0,
                    units=[64, 32, 16]):
    n_dense_layers = len(units)

    inputs_cont = Input(shape=(number_of_pupcandis, n_features-2), name='input_cont')
    pxpy = Input(shape=(number_of_pupcandis, 2), name='input_pxpy')

    embeddings = []
    inputs = [inputs_cont, pxpy]
    for i_emb in range(n_features_cat):
        input_cat = Input(shape=(number_of_pupcandis, ), name='input_cat{}'.format(i_emb))
        inputs.append(input_cat)
        embedding = Embedding(
            input_dim=embedding_input_dim[i_emb],
            output_dim=emb_out_dim,
            embeddings_initializer=initializers.RandomNormal(
                mean=0,
                stddev=0.4/emb_out_dim),
            name='embedding{}'.format(i_emb))(input_cat)
        embeddings.append(embedding)

    # can concatenate all 3 if updated in hls4ml, for now; do it pairwise
    # x = Concatenate()([inputs_cont] + embeddings)
    emb_concat = Concatenate()(embeddings)
    x = Concatenate()([inputs_cont, emb_concat])

    for i_dense in range(n_dense_layers):
        x = Dense(units[i_dense], activation='linear', kernel_initializer='lecun_uniform')(x)
        x = BatchNormalization(momentum=0.95)(x)
        x = Activation(activation=activation)(x)

    if t_mode == 0:
        x = GlobalAveragePooling1D(name='pool')(x)
        x = Dense(2, name='output', activation='linear')(x)

    if t_mode == 1:
        if with_bias:
            b = Dense(2, name='met_bias', activation='linear', kernel_initializer=initializers.VarianceScaling(scale=0.02))(x)
            pxpy = Add()([pxpy, b])
        w = Dense(1, name='met_weight', activation='linear', kernel_initializer=initializers.VarianceScaling(scale=0.02))(x)
        w = BatchNormalization(trainable=False, name='met_weight_minus_one', epsilon=False)(w)
        x = Multiply()([w, pxpy])

        x = GlobalAveragePooling1D(name='output')(x)
    outputs = x

    keras_model = Model(inputs=inputs, outputs=outputs)

    keras_model.get_layer('met_weight_minus_one').set_weights([np.array([1.]), np.array([-1.]), np.array([0.]), np.array([1.])])

    return keras_model


def dense_embedding_quantized(n_features=6,
                              n_features_cat=2,
                              number_of_pupcandis=100,
                              embedding_input_dim={0: 13, 1: 3},
                              emb_out_dim=2,
                              with_bias=True,
                              t_mode=0,
                              logit_total_bits=7,
                              logit_int_bits=2,
                              activation_total_bits=7,
                              logit_quantizer='quantized_bits',
                              activation_quantizer='quantized_relu',
                              activation_int_bits=2,
                              alpha=1,
                              use_stochastic_rounding=False,
                              units=[64, 32, 16]):
    n_dense_layers = len(units)

    logit_quantizer = getattr(qkeras.quantizers, logit_quantizer)(logit_total_bits, logit_int_bits, alpha=alpha, use_stochastic_rounding=use_stochastic_rounding)
    activation_quantizer = getattr(qkeras.quantizers, activation_quantizer)(activation_total_bits, activation_int_bits)

    inputs_cont = Input(shape=(number_of_pupcandis, n_features-2), name='input_cont')
    pxpy = Input(shape=(number_of_pupcandis, 2), name='input_pxpy')

    embeddings = []
    inputs = [inputs_cont, pxpy]
    for i_emb in range(n_features_cat):
        input_cat = Input(shape=(number_of_pupcandis, ), name='input_cat{}'.format(i_emb))
        inputs.append(input_cat)
        embedding = Embedding(
            input_dim=embedding_input_dim[i_emb],
            output_dim=emb_out_dim,
            embeddings_initializer=initializers.RandomNormal(
                mean=0,
                stddev=0.4/emb_out_dim),
            name='embedding{}'.format(i_emb))(input_cat)
        embeddings.append(embedding)

    # can concatenate all 3 if updated in hls4ml, for now; do it pairwise
    # x = Concatenate()([inputs_cont] + embeddings)
    emb_concat = Concatenate()(embeddings)
    x = Concatenate()([inputs_cont, emb_concat])

    for i_dense in range(n_dense_layers):
        x = QDense(units[i_dense], kernel_quantizer=logit_quantizer, bias_quantizer=logit_quantizer, kernel_initializer='lecun_uniform')(x)
        x = BatchNormalization(momentum=0.95)(x)
        x = QActivation(activation=activation_quantizer)(x)

    if t_mode == 0:
        x = qkeras.qpooling.QGlobalAveragePooling1D(name='pool', quantizer=logit_quantizer)(x)
        # pool size?
        outputs = QDense(2, name='output', bias_quantizer=logit_quantizer, kernel_quantizer=logit_quantizer, activation='linear')(x)

    if t_mode == 1:
        if with_bias:
            b = QDense(2, name='met_bias', kernel_quantizer=logit_quantizer, bias_quantizer=logit_quantizer, kernel_initializer=initializers.VarianceScaling(scale=0.02))(x)
            pxpy = Add()([pxpy, b])
        w = QDense(1, name='met_weight', kernel_quantizer=logit_quantizer, bias_quantizer=logit_quantizer, kernel_initializer=initializers.VarianceScaling(scale=0.02))(x)
        w = BatchNormalization(trainable=False, name='met_weight_minus_one', epsilon=False)(w)
        x = Multiply()([w, pxpy])

        x = GlobalAveragePooling1D(name='output')(x)
    outputs = x

    keras_model = Model(inputs=inputs, outputs=outputs)

    keras_model.get_layer('met_weight_minus_one').set_weights([np.array([1.]), np.array([-1.]), np.array([0.]), np.array([1.])])

    return keras_model


# Create the sender and receiver relations matrices
def assign_matrices(N, Nr):
    Rr = np.zeros([N, Nr], dtype=np.float32)
    Rs = np.zeros([N, Nr], dtype=np.float32)
    receiver_sender_list = [i for i in itertools.product(range(N), range(N)) if i[0] != i[1]]
    for i, (r, s) in enumerate(receiver_sender_list):
        Rr[r, i] = 1
        Rs[s, i] = 1
    return Rs, Rr


class multiply_e(Layer):
    def __init__(self):
        super(multiply_e, self).__init__()

    def call(sef, inputs):
        eij = inputs
        outputs = tf.math.exp(eij)
        return outputs


class softmax(Layer):
    def __init__(self):
        super(softmax, self).__init__()

    def call(self, inputs):
        sum_e, eij = inputs     # sum_e: 1xN   eij = 1xNr
        outputs = tf.math.divide(eij,sum_e)
        return outputs

class attention1(Layer):
    def __init__(self, N, Nr, Fprime):
        super(attention1, self).__init__()
        self.Rs, self.Rr = assign_matrices(N, Nr)
        self.Fprime = Fprime
    def build(self, input_shape):
        self.w = self.add_weight(
                                shape=(self.Fprime, input_shape[-2]),   # F' x F
                                initializer="random_normal",
                                trainable=True, name='w'
                                )
        self.a = self.add_weight(    # 1 x 2F'
                                shape=(1, 2*self.Fprime),   # F' x F
                                initializer="random_normal",
                                trainable=True, name='a'
                                )
    def call(self, inputs):
        wh = tf.tensordot(inputs, self.w, axes=[1,1])  # F' x F  x  (B x F x N)  --> F' x N
        whi = tf.tensordot(wh,self.Rr, axes=[1,0])   # (F' x N) x (N X Nr) ---> (F' x Nr)
        whj = tf.tensordot(wh, self.Rs,axes=[1,0])
        whi_whj = tf.concat([whi, whj], 1)   # ---> (2F' X Nr)
        a_whi_whj = tf.tensordot(whi_whj, self.a, axes=[1,1])    #  (B x 2F' x Nr) x (1 x 2F') ---> (B x Nr x 1)
        return a_whi_whj
    def get_config(self):
        config = super(attention1, self).get_config()
        config.update({
            'Rs': self.Rs,
            'Rr': self.Rr,
            'Fprime': self.Fprime})
        return config

class attention2(Layer):
    def __init__(self, N, Nr):
        super(attention2, self).__init__()
        self.Rs2, self.Rr2 = assign_matrices(N, Nr)
    def call(self, inputs):
        exp = tf.math.exp(inputs)    # B x Nr x 1
        exp_sum = tf.tensordot(exp, self.Rr2, axes = [1,1])   # (B x Nr x 1) x (N x Nr)  ---> (B x 1 x N)
        exp_sum = tf.tensordot(exp_sum, self.Rr2, axes=1)    # (B x 1 x N) x (N x Nr)   ----> (B x 1 x Nr)
        inputs_exp = tf.transpose(inputs,[0,2,1])
        output = tf.math.divide(inputs_exp, exp_sum)
        return output
    def get_config(self):
        config = super(attention2, self).get_config()
        config.update({
            'Rs2': self.Rs2,
            'Rr2': self.Rr2})
        return config

class attn_mult(Layer):
    def __init__(self):
        super(attn_mult, self).__init__()
    def call(self,inputs):
        tensor = inputs[0]
        attn = inputs[1]
        outputs = tf.multiply(tensor, attn)
        return outputs


def graph_embedding(compute_ef, n_features=6,
                    n_features_cat=2,
                    activation='relu',
                    number_of_pupcandis=100,
                    embedding_input_dim={0: 13, 1: 3},
                    emb_out_dim=8,
                    units=[64, 32, 16],
                    edge_list=[]):
    n_dense_layers = len(units)
    name = 'met'

    inputs_cont = Input(shape=(number_of_pupcandis, n_features-2), name='input_cont')
    pxpy = Input(shape=(number_of_pupcandis, 2), name='input_pxpy')

    embeddings = []
    inputs = [inputs_cont, pxpy]
    for i_emb in range(n_features_cat):
        input_cat = Input(shape=(number_of_pupcandis, ), name='input_cat{}'.format(i_emb))
        inputs.append(input_cat)
        embedding = Embedding(
            input_dim=embedding_input_dim[i_emb],
            output_dim=emb_out_dim,
            embeddings_initializer=initializers.RandomNormal(
                mean=0,
                stddev=0.4/emb_out_dim),
            name='embedding{}'.format(i_emb))(input_cat)
        embeddings.append(embedding)

    N = number_of_pupcandis
    Nr = N*(N-1)
    if compute_ef == 1:
        num_of_edge_feat = len(edge_list)
        edge_feat = Input(shape=(Nr, num_of_edge_feat), name='edge_feat')
        inputs.append(edge_feat)

    # can concatenate all 3 if updated in hls4ml, for now; do it pairwise
    # x = Concatenate()([inputs_cont] + embeddings)
    emb_concat = Concatenate()(embeddings)
    x = Concatenate()([inputs_cont, emb_concat])

    N = number_of_pupcandis
    P = n_features+n_features_cat
    Nr = N*(N-1)  # number of relations (edges)

    x = BatchNormalization()(x)

    # Swap axes of input data (batch,nodes,features) -> (batch,features,nodes)
    x = Permute((2, 1), input_shape=x.shape[1:])(x)

    attn1 = attention1(N, Nr, 8)(x)
    attn1 = LeakyReLU()(attn1)
    attn1 = attention2(N, Nr)(attn1)
    attn2 = attention1(N, Nr, 8)(x)
    attn2 = LeakyReLU()(attn2)
    attn2 = attention2(N, Nr)(attn2)
    attn3 = attention1(N, Nr, 8)(x)
    attn3 = LeakyReLU()(attn1)
    attn3 = attention2(N, Nr)(attn1)
    attn4 = attention1(N, Nr, 8)(x)
    attn4 = LeakyReLU()(attn1)
    attn4 = attention2(N, Nr)(attn1)
    



    # Whi Whj
    #wh = Dense(F, use_bias=False, trainable=True)(x)  # nodes, features  ---> nodes, F'  
    #wh = Permute((2,1))(wh)  # nodes, F' ---> F', nodes
    #r = Dense(Nr, use_bias=False, trainable=False, name="receiving")(wh)  # Receiving adjacency matrix    F', Nr
    #s = Dense(Nr, use_bias=False, trainable=False, name="sending")(wh)  # Sending adjacency matrix      F', Nr
    #eq3_concat = Concatenate(axis=1)([r,s])   # 2FxNr
    #eq3 = Dense(1, activation='Leakyrelu', use_bias=False, trainable=True, name="a_vec")(eq3_concat)    # Nr x 1
    ## eq3_vec = [a  b  c  d]
    ##           [a  b  c  d]
    ##           [a  b  c  d]
    ##           [a  b  c  d]

    #sum_e = Dense(N, use_bias=False, trainable=False, name="sending")   # N x 1
    #norm =  Softmax()(sum_e, eq3)

    # Marshaling function
    ORr = Dense(Nr, use_bias=False, trainable=False, name='tmul_{}_1'.format(name))(x)  # Receiving adjacency matrix
    ORs = Dense(Nr, use_bias=False, trainable=False, name='tmul_{}_2'.format(name))(x)  # Sending adjacency matrix
    node_feat = Concatenate(axis=1)([ORr, ORs])  # Concatenates Or and Os  ( no relations features Ra matrix )
    # Outputis new array = [batch, 2x features, edges]

    # Edges MLP
    h = Permute((2, 1), input_shape=node_feat.shape[1:])(node_feat)
    # feature selection scalars are generated from bias vectors from 'scalars' dense layer
    #init_scl_array = np.ones([16 + num_of_edge_feat])
    # init_scl_input = Dense(16, trainable=False, use_bias=False, name='scalars_init')(h)  # This line is is to set next layer's weights to zero so we get just the bias
    #scl = Dense(16+num_of_edge_feat, trainable=True, activation='softmax', bias_initializer=initializers.Ones(), name='scalars')(init_scl_input)
    edge_units = [64, 32, 16]
    n_edge_dense_layers = len(edge_units)
    if compute_ef == 1:
        h = Concatenate(axis=2, name='concatenate_edge')([h, edge_feat])
        # output is [batch, edges, node features + edge features]
        # h = Multiply()([h,scl]) # multiply scalars with features
    for i_dense in range(n_edge_dense_layers):
        h = Dense(edge_units[i_dense], activation='linear', kernel_initializer='lecun_uniform')(h)
        h = BatchNormalization(momentum=0.95)(h)
        h = Activation(activation=activation)(h)
    out_e = h

    # Transpose output and permutes columns 1&2
    out_e = Permute((2, 1))(out_e)
    mult1 = attn_mult()([out_e, attn1])
    mult2 = attn_mult()([out_e, attn2])
    mult3 = attn_mult()([out_e, attn3])
    mult4 = attn_mult()([out_e, attn4])

    # Multiply edges MLP output by receiver nodes matrix Rr
    out_e1 = Dense(N, use_bias=False, trainable=False, name='tmul_{}_3_1'.format(name))(mult1)
    out_e2 = Dense(N, use_bias=False, trainable=False, name='tmul_{}_3_2'.format(name))(mult2)
    out_e3 = Dense(N, use_bias=False, trainable=False, name='tmul_{}_3_3'.format(name))(mult3)
    out_e4 = Dense(N, use_bias=False, trainable=False, name='tmul_{}_3_4'.format(name))(mult4)

    out_e = Average()([out_e1, out_e2, out_e3, out_e4])
    out_e = Activation(activation='relu')(out_e)

    # Nodes MLP (takes as inputs node features and embeding from edges MLP)
    inp_n = Concatenate(axis=1)([x, out_e])

    # Transpose input and permutes columns 1&2
    h = Permute((2, 1), input_shape=inp_n.shape[1:])(inp_n)

    # Nodes MLP
    for i_dense in range(n_dense_layers):
        h = Dense(units[i_dense], activation='linear', kernel_initializer='lecun_uniform')(h)
        h = BatchNormalization(momentum=0.95)(h)
        h = Activation(activation=activation)(h)
    w = Dense(1, name='met_weight', activation='linear', kernel_initializer=initializers.VarianceScaling(scale=0.02))(h)
    w = BatchNormalization(trainable=False, name='met_weight_minus_one', epsilon=False)(w)
    x = Multiply()([w, pxpy])
    outputs = GlobalAveragePooling1D(name='output')(x)

    keras_model = Model(inputs=inputs, outputs=outputs)

    keras_model.get_layer('met_weight_minus_one').set_weights([np.array([1.]), np.array([-1.]), np.array([0.]), np.array([1.])])

    # Create a fully connected adjacency matrix
    Rs, Rr = assign_matrices(N, Nr)
    keras_model.get_layer('tmul_{}_1'.format(name)).set_weights([Rr])
    keras_model.get_layer('tmul_{}_2'.format(name)).set_weights([Rs])
    keras_model.get_layer('tmul_{}_3'.format(name)).set_weights([np.transpose(Rr)])
    #keras_model.get_layer('receiving').set_weights([Rr])
    #keras_model.get_layer('receiving').set_weights([Rs])
    #w_zeros = np.zeros((16, 16+num_of_edge_feat))
    #b_zeros = np.ones((19))
    #init_zeros = np.zeros((16, 16))
    # keras_model.get_layer('scalars_init').set_weights([init_zeros])
    # keras_model.get_layer('scalars').set_weights([w_zeros,b_zeros])

    return keras_model
