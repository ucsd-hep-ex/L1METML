from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, BatchNormalization, Dropout, Lambda, Conv1D, SpatialDropout1D, Concatenate, Flatten, Reshape, Multiply, Add, GlobalAveragePooling1D, Activation, Permute
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

        #x = GlobalAveragePooling1D(name='output')(x)
        x = Lambda(lambda x: K.sum(x, axis=1), name='output')(x)

    if t_mode == 2: # regresses a single met weight rather than 128
        w = Dense(1, name='puppi_weights', activation='linear', kernel_initializer=initializers.VarianceScaling(scale=0.02))(x)
        w = Lambda(lambda x: K.sum(x, axis=1), name='met_weight')(w) 

        pmet = Lambda(lambda x: K.sum(x, axis=1), name='pmet')(pxpy)
        pmet = BatchNormalization(trainable=False, name='pmet_weight_minus_one', epsilon=False)(pmet)

        x = Multiply()([w, pmet])

    outputs = x

    keras_model = Model(inputs=inputs, outputs=outputs)
    
    if t_mode == 1:
        keras_model.get_layer('met_weight_minus_one').set_weights([np.array([1.]), np.array([-1.]), np.array([0.]), np.array([1.])])
    elif t_mode == 2:
        keras_model.get_layer('pmet_weight_minus_one').set_weights([np.array([1.,1.]), np.array([-1.,-1.]), np.array([0.,0.]), np.array([1., 1.])])
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

    # Marshaling function
    ORr = Dense(Nr, use_bias=False, trainable=False, name='tmul_{}_1'.format(name))(x)  # Receiving adjacency matrix
    ORs = Dense(Nr, use_bias=False, trainable=False, name='tmul_{}_2'.format(name))(x)  # Sending adjacency matrix
    node_feat = Concatenate(axis=1)([ORr, ORs])  # Concatenates Or and Os  ( no relations features Ra matrix )
    # Outputis new array = [batch, 2x features, edges]

    # Edges MLP
    h = Permute((2, 1), input_shape=node_feat.shape[1:])(node_feat)
    edge_units = [64, 32, 16]
    n_edge_dense_layers = len(edge_units)
    if compute_ef == 1:
        h = Concatenate(axis=2, name='concatenate_edge')([h, edge_feat])
    for i_dense in range(n_edge_dense_layers):
        h = Dense(edge_units[i_dense], activation='linear', kernel_initializer='lecun_uniform')(h)
        h = BatchNormalization(momentum=0.95)(h)
        h = Activation(activation=activation)(h)
    out_e = h

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

    return keras_model


def mlp_mixer_embedding(n_features=6,
                       n_features_cat=2,
                       activation='relu',
                       number_of_pupcandis=128,
                       embedding_input_dim={0: 13, 1: 3},
                       emb_out_dim=8,
                       with_bias=True,
                       t_mode=1,
                       mixer_blocks=4,
                       hidden_dim=256,
                       tokens_mlp_dim=64,
                       channels_mlp_dim=128,
                       dropout_rate=0.1):
    """
    MLP-Mixer architecture where puppi candidates are treated as tokens (sequence dimension).
    
    Args:
        n_features: Number of continuous features
        n_features_cat: Number of categorical features  
        activation: Activation function for MLPs
        number_of_pupcandis: Number of puppi candidates (tokens)
        embedding_input_dim: Dictionary mapping categorical feature index to vocab size
        emb_out_dim: Embedding output dimension
        with_bias: Whether to use bias in MET computation
        t_mode: training mode, 0: regressing MET directly, 1: regressing the MET weights
        mixer_blocks
        hidden_dim: Hidden dimension (channels)
        tokens_mlp_dim: Hidden dimension for token-mixing MLP
        channels_mlp_dim: Hidden dimension for channel-mixing MLP
        dropout_rate
    """
    # TODO: modeled after dense_embedding model, check what needs to change
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
    x = Concatenate()([inputs_cont, emb_concat])  # Shape: (batch, tokens, channels)
    
    x = Dense(hidden_dim, kernel_initializer='lecun_uniform', name='patch_projection')(x)
    x = BatchNormalization()(x)
    
    # MLP-Mixer blocks
    for block_idx in range(mixer_blocks):
        residual = x
        x = BatchNormalization(name=f'norm1_block_{block_idx}')(x)
        
        x = Permute((2, 1), name=f'transpose_for_token_mix_{block_idx}')(x)  # (batch, channels, tokens)
        x = Dense(tokens_mlp_dim, activation=activation, kernel_initializer='lecun_uniform', 
                 name=f'token_mix_1_{block_idx}')(x)
        x = Dropout(dropout_rate, name=f'token_mix_dropout_1_{block_idx}')(x)
        x = Dense(number_of_pupcandis, kernel_initializer='lecun_uniform', 
                 name=f'token_mix_2_{block_idx}')(x)
        x = Dropout(dropout_rate, name=f'token_mix_dropout_2_{block_idx}')(x)
        x = Permute((2, 1), name=f'transpose_back_token_mix_{block_idx}')(x)  # (batch, tokens, channels)
        
        x = Add(name=f'residual_1_{block_idx}')([x, residual])
        
        residual = x
        x = BatchNormalization(name=f'norm2_block_{block_idx}')(x)
        
        x = Dense(channels_mlp_dim, activation=activation, kernel_initializer='lecun_uniform',
                 name=f'channel_mix_1_{block_idx}')(x)
        x = Dropout(dropout_rate, name=f'channel_mix_dropout_1_{block_idx}')(x)
        x = Dense(hidden_dim, kernel_initializer='lecun_uniform',
                 name=f'channel_mix_2_{block_idx}')(x)
        x = Dropout(dropout_rate, name=f'channel_mix_dropout_2_{block_idx}')(x)
        
        x = Add(name=f'residual_2_{block_idx}')([x, residual])
    
    x = BatchNormalization(name='final_norm')(x)
    
    if t_mode == 0:
        x = GlobalAveragePooling1D(name='pool')(x)
        x = Dense(2, name='output', activation='linear')(x)
    
    elif t_mode == 1:
        if with_bias:
            b = Dense(2, name='met_bias', activation='linear', 
                     kernel_initializer=initializers.VarianceScaling(scale=0.02))(x)
            pxpy = Add()([pxpy, b])
        
        w = Dense(1, name='met_weight', activation='linear', 
                 kernel_initializer=initializers.VarianceScaling(scale=0.02))(x)
        w = BatchNormalization(trainable=False, name='met_weight_minus_one', epsilon=False)(w)
        x = Multiply()([w, pxpy])
        x = GlobalAveragePooling1D(name='output')(x)
    elif t_mode == 2: # regresses a single met weight rather than 128
        w = Dense(1, name='puppi_weights', activation='linear', kernel_initializer=initializers.VarianceScaling(scale=0.02))(x)
        w = Lambda(lambda x: K.sum(x, axis=1), name='met_weight')(w) 

        pmet = Lambda(lambda x: K.sum(x, axis=1), name='pmet')(pxpy)
        pmet = BatchNormalization(trainable=False, name='pmet_weight_minus_one', epsilon=False)(pmet)

        x = Multiply()([w, pmet])

    outputs = x
    
    keras_model = Model(inputs=inputs, outputs=outputs)
    
    if t_mode == 1:
        keras_model.get_layer('met_weight_minus_one').set_weights([
            np.array([1.]), np.array([-1.]), np.array([0.]), np.array([1.])
        ])
    if t_mode == 2:
        keras_model.get_layer('pmet_weight_minus_one').set_weights([
            np.array([1., 1.]), np.array([-1., -1.]), np.array([0., 0.]), np.array([1., 1.])
        ])
    
    
    return keras_model
