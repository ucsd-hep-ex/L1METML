from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, BatchNormalization, Dropout, Lambda, Conv1D, SpatialDropout1D, Concatenate, Flatten, Reshape, Multiply, Add, GlobalAveragePooling1D, Activation, Permute, Layer, LayerNormalization
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


def S(tensor, threshold):
    x = tf.where(tf.math.greater_equal(tensor, T), 1, 0)
    return x


def node_select(compute_ef, n_features=6,
                n_features_cat=2,
                activation='relu',
                number_of_pupcandis=100,
                embedding_input_dim={0: 13, 1: 3},
                emb_out_dim=8,
                units=[64, 32, 16]):

    N = number_of_pupcandis
    Nr = N*(N-1)

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

    emb_concat = Concatenate()(embeddings)
    x = Concatenate()([inputs_cont, emb_concat])
    x = BatchNormalization()(x)  # (batch,nodes,features)

    # Swap axes of input data (batch,nodes,features) -> (batch,features,nodes)
    x = Permute((2, 1), input_shape=x.shape[1:])(x)

    # node selection

    F_prime = 6
    wx = Dense(F_prime, use_bias=False, trainable=True, name='W')(x)  # (nodes,F')

    # Swap axes of input data (batch,nodes,features) -> (batch,F',nodes)
    x = Permute((2, 1), input_shape=x.shape[1:])(x)

    ORs = Dense(Nr, use_bias=False, trainable=False, name='sending'.format(name))(wx)   # Neighborhood aggregation   (F', Nr)
    ORr = Dense(N, use_bias=False, trainable=False, name='receiving'.format(name))(ORs)  # (F', N)

    x = Permute((2, 1), input_shape=x.shape[1:])(ORr)  # (N, F')
    p = Dense(1, activation='relu', use_bias=False, trainable=True, name='p')(x)  # (N,1), W0 weightmatrix

    S = Lambda(S(p, T))  # node selection function

    # Selective Aggregation and Feature Update

    # alpha

    # A matrix
    A = Multiply(S, wx)
    A = Multiply(alhpa, A)

    keras_model = Model(inputs=inputs, outputs=w0)

    Rs, Rr = assign_matrices(N, Nr)
    keras_model.get_layer('sending').set_weights([np.transpose(Rr)])
    keras_model.get_layer('receiving'.format(name)).set_weights([Rs])
    return keras_model

class encode_inputs(Layer):
    def __init__(self, encoding_size, num_of_edge_feat, embedding_input_dim):
        super(encode_inputs, self).__init__()
        self.encoding_size = encoding_size
        self.num_of_edge_feat = num_of_edge_feat
        self.embedding_input_dim = embedding_input_dim

    def call(self, inputs):
        encoding_size = self.encoding_size
        num_of_edge_feat = self.num_of_edge_feat
        embedding_input_dim = self.embedding_input_dim
        features = []
        for cont_feat in range(4):
            feature = tf.gather(inputs, indices=[cont_feat], axis=-1)
            #encoded_feature = Dense(units=encoding_size, name='dense_{}'.format(cont_feat))(feature)
            features.append(feature)
        for disc_feat in range(4,6):
            feature = tf.gather(inputs, indices=[disc_feat], axis=-1)
            #encoded_feature = Embedding(
            #                        input_dim=embedding_input_dim[disc_feat-4], output_dim=encoding_size,
            #                        name='input_cat{}'.format(disc_feat))(feature)
            #    # Convert the index values to embedding representations.
            #features.append(encoded_feature)
            #else:
            #    # Project the numeric feature to encoding_size using linear transformation.
            #    feature = tf.expand_dims(inputs[feature_name], -1)
            #    feature = layers.Dense(units=encoding_size)(feature)
            features.append(feature)
        for edge_feat in range(6,6+num_of_edge_feat):
            feature = tf.gather(inputs, indices=[edge_feat], axis=-1)
            print(np.shape(feature))
            #feature = Dense(units=encoding_size, name='dense_{}'.format(cont_feat))(feature)
            features.append(feature)
        return features
    def get_config(self):
        config = super(encode_inputs, self).get_config()
        config.update({
            'encoding_size': self.encoding_size,
            'num_of_edge_feat': self.num_of_edge_feat,
            'embeding_input_dim': self.embedding_input_dim})
        return config

class GatedResidualNetwork(Layer):
    def __init__(self, units, dropout_rate):
        super(GatedResidualNetwork, self).__init__()
        self.units = units
        self.elu_dense = Dense(units, activation="elu")
        self.linear_dense = Dense(units)
        self.dropout = Dropout(dropout_rate)
        self.gated_linear_unit = GatedLinearUnit(units)
        self.layer_norm = LayerNormalization()
        self.project = Dense(units)

    def call(self, inputs):
        x = self.elu_dense(inputs)
        x = self.linear_dense(x)
        x = self.dropout(x)
        if inputs.shape[-1] != self.units:
            inputs = self.project(inputs)
        x = inputs + self.gated_linear_unit(x)
        x = self.layer_norm(x)
        return x
    
    def get_config(self):
        config = super(GatedResidualNetwork, self).get_config()
        config.update({
            'units': self.units,
            'elu_dense': self.elu_dense,
            'linear_dense': self.linear_dense,
            'dropout': self.dropout,
            'gated_linear_unit': self.gated_linear_unit,
            'layer_norm': self.layer_norm,
            'self.project': self.project})
        return config

class GatedLinearUnit(Layer):
    def __init__(self, units):
        super(GatedLinearUnit, self).__init__()
        self.linear = Dense(units)
        self.sigmoid = Dense(units, activation="sigmoid")

    def call(self, inputs):
        return self.linear(inputs) * self.sigmoid(inputs)

    def get_config(self):
        config = super(GatedLinearUnit, self).get_config()
        config.update({
            'linear': self.linear,
            'sigmoid': self.sigmoid})
        return config

class VariableSelection(Layer):
    def __init__(self, num_features, units, dropout_rate):
        super(VariableSelection, self).__init__()
        self.grns = list()
        # Create a GRN for each feature independently
        for idx in range(num_features):
            grn = GatedResidualNetwork(units, dropout_rate)
            self.grns.append(grn)
        # Create a GRN for the concatenation of all the features
        self.grn_concat = GatedResidualNetwork(units, dropout_rate)
        self.softmax = Dense(units=num_features, activation="softmax")

    def call(self, inputs):
        v = Concatenate(axis=-1)(inputs)
        v = self.grn_concat(v)
        v = tf.expand_dims(self.softmax(v), axis=-1)
        print(np.shape(v))

        x = []
        #print("inputs", len(inputs))
        for idx, input in enumerate(inputs):

            x.append(self.grns[idx](input))
        x = tf.stack(x, axis=2)
        #print(np.shape(v))
        #print(np.shape(x))
        outputs = tf.squeeze(tf.matmul(v, x, transpose_a=True), axis=2)
        print(np.shape(outputs))
        #inputs_cont = outputs[:,:,0:4]
        return outputs

    def get_config(self):
        config = super(VariableSelection, self).get_config()
        config.update({
            'grns': self.grns,
            'grn_concat': self.grn_concat,
            'softmax': self.softmax})
        return config



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

    N = number_of_pupcandis
    Nr = N*(N-1)

    inputs_cont = Input(shape=(number_of_pupcandis, n_features-2), name='input_cont')
    pxpy = Input(shape=(number_of_pupcandis, 2), name='input_pxpy')
    input_cat_0 = Input(shape=(number_of_pupcandis, ), name='input_cat0')
    input_cat_1 = Input(shape=(number_of_pupcandis, ), name='input_cat1')
    if compute_ef == 1:
        num_of_edge_feat = len(edge_list)
        edge_feat = Input(shape=(Nr, num_of_edge_feat), name='edge_feat')

    embeddings = []

    if compute_ef == 1:
        inputs = [inputs_cont, pxpy, input_cat_0, input_cat_1, edge_feat]
    if compute_ef == 0:
        inputs = [inputs_cont, pxpy, input_cat_0, input_cat_1]
    #for i_emb in range(n_features_cat):
    #    #input_cat = Input(shape=(number_of_pupcandis, ), name='input_cat{}'.format(i_emb))
    #    inputs.append(input_cat)
    #    embedding = Embedding(
    #        input_dim=embedding_input_dim[i_emb],
    #        output_dim=emb_out_dim,
    #        embeddings_initializer=initializers.RandomNormal(
    #            mean=0,
    #            stddev=0.4/emb_out_dim),
    #        name='embedding{}'.format(i_emb))(input_cat)
    #    embeddings.append(embedding)

    #if compute_ef == 1:
    #    num_of_edge_feat = len(edge_list)
    #    edge_feat = Input(shape=(Nr, num_of_edge_feat), name='edge_feat')
    #    inputs.append(edge_feat)

    # can concatenate all 3 if updated in hls4ml, for now; do it pairwise
    # x = Concatenate()([inputs_cont] + embeddings)
    #emb_concat = Concatenate()(embeddings)
    #x = Concatenate()([inputs_cont, emb_concat])

    input_cat_0 = Reshape((N, 1))(input_cat_0)
    input_cat_1 = Reshape((N, 1))(input_cat_1)
    x = Concatenate()([inputs_cont, input_cat_0])
    x = Concatenate()([x, input_cat_1])

    N = number_of_pupcandis
    P = n_features+n_features_cat
    Nr = N*(N-1)  # number of relations (edges)


    # Swap axes of input data (batch,nodes,features) -> (batch,features,nodes)
    x = Permute((2, 1), input_shape=x.shape[1:])(x)

    # Marshaling function
    ORr = Dense(Nr, use_bias=False, trainable=False, name='tmul_{}_1'.format(name))(x)  # Receiving adjacency matrix
    ORs = Dense(Nr, use_bias=False, trainable=False, name='tmul_{}_2'.format(name))(x)  # Sending adjacency matrix
    node_feat = Concatenate(axis=1)([ORr, ORs])  # Concatenates Or and Os  ( no relations features Ra matrix )
    # Outputis new array = [batch, 2x features, edges]

    # Edges MLP
    h = Permute((2, 1), input_shape=node_feat.shape[1:])(node_feat)
    encoding_size = 8

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

    features = encode_inputs(encoding_size, num_of_edge_feat, embedding_input_dim)(h)
    encoded_features = []
    for cont_feat in range(4):
        encoded_feature = Dense(units=encoding_size, name='dense_{}'.format(cont_feat))(features[cont_feat])
        encoded_features.append(encoded_feature)
        print(encoded_feature)
    for disc_feat in range(4,6):
        encoded_feature = Embedding(
                                    input_dim=embedding_input_dim[disc_feat-4],
                                    output_dim=encoding_size,
                                    name='input_cat{}'.format(disc_feat))(features[disc_feat])
                # Convert the index values to embedding representations.
        encoded_features.append(encoded_feature)
        print(features[disc_feat])
    for edge_feat in range(6,6+num_of_edge_feat):
        print(edge_feat)
        encoded_feature = Dense(units=encoding_size, name='dense_{}'.format(cont_feat))(features[edge_feat])
        encoded_features.append(encoded_feature)



    dropout_rate = 0.15
    j = VariableSelection((4+2+num_of_edge_feat), 
                    encoding_size, dropout_rate)(features)
    j = BatchNormalization()(j)

    for i_dense in range(n_edge_dense_layers):
        j = Dense(edge_units[i_dense], activation='linear', kernel_initializer='lecun_uniform')(j)
        j = BatchNormalization(momentum=0.95)(j)
        j = Activation(activation=activation)(j)
    out_e = j

    # Transpose output and permutes columns 1&2
    out_e = Permute((2, 1))(out_e)

    # Multiply edges MLP output by receiver nodes matrix Rr
    out_e = Dense(N, use_bias=False, trainable=False, name='tmul_{}_3'.format(name))(out_e)

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
    #w_zeros = np.zeros((16, 16+num_of_edge_feat))
    #b_zeros = np.ones((19))
    #init_zeros = np.zeros((16, 16))
    # keras_model.get_layer('scalars_init').set_weights([init_zeros])
    # keras_model.get_layer('scalars').set_weights([w_zeros,b_zeros])

    return keras_model

