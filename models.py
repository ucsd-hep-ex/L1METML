import keras
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

    #edges=100*99
    #edge_feat = Input(shape=(edges, 3), name='edge_feat')
    #inputs.append(edge_feat)
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

class Distiller(Model):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.teacher = teacher
        self.student = student

    def call(self, inputs):
        outputs = self.student(inputs)
        outputs = tf.reshape(outputs,[-1,2])
        return outputs

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        alpha=0.1,
        temperature=3,
    ):
        """ Configure the distiller.

        Args:
            optimizer: Keras optimizer for the student weights
            metrics: Keras metrics for evaluation
            student_loss_fn: Loss function of difference between student
                predictions and ground-truth
            distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
            alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
            temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        # Unpack data
        x, y = data
        print(y)
        print(np.shape(y))
        #print(x)

        # Forward pass of teacher
        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(x, training=True)

            # Compute losses
            #student_predictions = keras.layers.Reshape([256, 2])(y)
            #print(y)
            print("------")
            print("y")
            print(y)
            #student_predictions = keras.layers.Reshape([1,2])(student_predictions)
            #y = keras.layers.Reshape([1, 2])(y)
            print("y reshape")
            print(y)

            student_loss = self.student_loss_fn(y, student_predictions)

            # Compute scaled distillation loss from https://arxiv.org/abs/1503.02531
            # The magnitudes of the gradients produced by the soft targets scale
            # as 1/T^2, multiply them by T^2 when using both hard and soft targets.
            distillation_loss = (
                self.distillation_loss_fn(
                    tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                    tf.nn.softmax(student_predictions / self.temperature, axis=1),
                )
                * self.temperature**2
            )

            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"student_loss": student_loss, "distillation_loss": distillation_loss}
        )
        return results

    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        y_prediction = self.student(x, training=False)

        # Calculate the loss
        student_loss = self.student_loss_fn(y, y_prediction)

        # Update the metrics.
        self.compiled_metrics.update_state(y, y_prediction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results
