import tensorflow
import tensorflow.keras.backend as K
from tensorflow.keras import optimizers, initializers
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, CSVLogger
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

import numpy as np
import tables
import matplotlib.pyplot as plt
import argparse
import math
#import setGPU
import time
import os
import pathlib
import datetime
import tqdm
import h5py
from glob import glob
import itertools

# Import custom modules

from Write_MET_binned_histogram import *
from cyclical_learning_rate import CyclicLR
from models import *
from utils import *
from loss import custom_loss
from DataGenerator import DataGenerator

import matplotlib.pyplot as plt
import mplhep as hep


def MakeEdgeHist(edge_feat, xname, outputname, nbins=1000, density=False, yname="# of edges"):
    plt.style.use(hep.style.CMS)
    plt.figure(figsize=(10, 8))
    plt.hist(edge_feat, bins=nbins, density=density, histtype='step', facecolor='k', label='Truth')
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.savefig(outputname)
    plt.close()


def deltaR_calc(eta1, phi1, eta2, phi2):
    """ calculate deltaR """
    dphi = (phi1-phi2)
    gt_pi_idx = (dphi > np.pi)
    lt_pi_idx = (dphi < -np.pi)
    dphi[gt_pi_idx] -= 2*np.pi
    dphi[lt_pi_idx] += 2*np.pi
    deta = eta1-eta2
    return np.hypot(deta, dphi)


def kT_calc(pti, ptj, dR):
    min_pt = np.minimum(pti, ptj)
    kT = min_pt * dR
    return kT


def z_calc(pti, ptj):
    epsilon = 1.0e-12
    min_pt = np.minimum(pti, ptj)
    z = min_pt/(pti + ptj + epsilon)
    return z


def mass2_calc(pi, pj):
    pij = pi + pj
    m2 = pij[:, :, 0]**2 - pij[:, :, 1]**2 - pij[:, :, 2]**2 - pij[:, :, 3]**2
    return m2


def get_callbacks(path_out, sample_size, batch_size):
    # early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=40, verbose=1, restore_best_weights=False)

    csv_logger = CSVLogger(f'{path_out}loss_history.log')

    # model checkpoint callback
    # this saves our model architecture + parameters into model.h5
    model_checkpoint = ModelCheckpoint(f'{path_out}model.h5', monitor='val_loss',
                                       verbose=0, save_best_only=True,
                                       save_weights_only=False, mode='auto',
                                       period=1)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=0.000001, cooldown=3, verbose=1)

    lr_scale = 1.
    clr = CyclicLR(base_lr=0.0003*lr_scale, max_lr=0.001*lr_scale, step_size=sample_size/batch_size, mode='triangular2')

    stop_on_nan = tensorflow.keras.callbacks.TerminateOnNaN()

    callbacks = [early_stopping, clr, stop_on_nan, csv_logger, model_checkpoint]

    return callbacks


def test(Yr_test, predict_test, PUPPI_pt, path_out):

    MakePlots(Yr_test, predict_test, PUPPI_pt, path_out=path_out)

    Yr_test = convertXY2PtPhi(Yr_test)
    predict_test = convertXY2PtPhi(predict_test)
    PUPPI_pt = convertXY2PtPhi(PUPPI_pt)

    extract_result(predict_test, Yr_test, path_out, 'TTbar', 'ML')
    extract_result(PUPPI_pt, Yr_test, path_out, 'TTbar', 'PU')

    MET_rel_error_opaque(predict_test[:, 0], PUPPI_pt[:, 0], Yr_test[:, 0], name=''+path_out+'rel_error_opaque.png')
    MET_binned_predict_mean_opaque(predict_test[:, 0], PUPPI_pt[:, 0], Yr_test[:, 0], 20, 0, 500, 0, '.', name=''+path_out+'PrVSGen.png')

    Phi_abs_error_opaque(PUPPI_pt[:, 1], predict_test[:, 1], Yr_test[:, 1], name=path_out+'Phi_abs_err')
    Pt_abs_error_opaque(PUPPI_pt[:, 0], predict_test[:, 0], Yr_test[:, 0], name=path_out+'Pt_abs_error')


def train_dataGenerator(args):
    # general setup
    maxNPF = args.maxNPF
    n_features_pf = 6
    n_features_pf_cat = 2
    normFac = 1.
    epochs = args.epochs
    batch_size = args.batch_size
    preprocessed = True
    t_mode = args.mode
    inputPath = args.input
    path_out = args.output
    quantized = args.quantized
    model = args.model
    units = list(map(int, args.units))
    compute_ef = args.compute_edge_feat
    edge_list = args.edge_features

    # separate files into training, validation, and testing
    filesList = glob(os.path.join(inputPath, '*.root'))
    filesList.sort(reverse=True)

    assert len(filesList) >= 3, "Need at least 3 files for DataGenerator: 1 valid, 1 test, 1 train"

    valid_nfiles = max(1, int(.1*len(filesList)))
    train_nfiles = len(filesList) - 2*valid_nfiles
    test_nfiles = valid_nfiles
    train_filesList = filesList[0:train_nfiles]
    valid_filesList = filesList[train_nfiles: train_nfiles+valid_nfiles]
    test_filesList = filesList[train_nfiles+valid_nfiles:test_nfiles+train_nfiles+valid_nfiles]

    if compute_ef == 1:

        # set up data generators; they perform h5 conversion if necessary and load in data batch by batch
        trainGenerator = DataGenerator(list_files=train_filesList, batch_size=batch_size, maxNPF=maxNPF, compute_ef=1, edge_list=edge_list)
        validGenerator = DataGenerator(list_files=valid_filesList, batch_size=batch_size, maxNPF=maxNPF, compute_ef=1, edge_list=edge_list)
        testGenerator = DataGenerator(list_files=test_filesList, batch_size=batch_size, maxNPF=maxNPF, compute_ef=1, edge_list=edge_list)
        Xr_train, Yr_train = trainGenerator[0]  # this apparenly calls all the attributes, so that we can get the correct input dimensions (train_generator.emb_input_dim)

    else:
        trainGenerator = DataGenerator(list_files=train_filesList, batch_size=batch_size)
        validGenerator = DataGenerator(list_files=valid_filesList, batch_size=batch_size)
        testGenerator = DataGenerator(list_files=test_filesList, batch_size=batch_size)
        Xr_train, Yr_train = trainGenerator[0]  # this apparenly calls all the attributes, so that we can get the correct input dimensions (train_generator.emb_input_dim)

    # Load training model
    if quantized is None:
        if model == 'dense_embedding':
            keras_model = dense_embedding(n_features=n_features_pf,
                                          emb_out_dim=2,
                                          n_features_cat=n_features_pf_cat,
                                          activation='tanh',
                                          embedding_input_dim=trainGenerator.emb_input_dim,
                                          number_of_pupcandis=maxNPF,
                                          t_mode=t_mode,
                                          with_bias=False,
                                          units=units)
        elif model == 'graph_embedding':
            keras_model, keras_model_weights = graph_embedding(n_features=n_features_pf,
                                          emb_out_dim=2,
                                          n_features_cat=n_features_pf_cat,
                                          activation='tanh',
                                          embedding_input_dim=trainGenerator.emb_input_dim,
                                          number_of_pupcandis=maxNPF,
                                          units=units, compute_ef=compute_ef, edge_list=edge_list)

    else:
        logit_total_bits = int(quantized[0])
        logit_int_bits = int(quantized[1])
        activation_total_bits = int(quantized[0])
        activation_int_bits = int(quantized[1])

        keras_model = dense_embedding_quantized(n_features=n_features_pf,
                                                emb_out_dim=2,
                                                n_features_cat=n_features_pf_cat,
                                                activation_quantizer='quantized_relu',
                                                embedding_input_dim=trainGenerator.emb_input_dim,
                                                number_of_pupcandis=maxNPF,
                                                t_mode=t_mode,
                                                with_bias=False,
                                                logit_quantizer='quantized_bits',
                                                logit_total_bits=logit_total_bits,
                                                logit_int_bits=logit_int_bits,
                                                activation_total_bits=activation_total_bits,
                                                activation_int_bits=activation_int_bits,
                                                alpha=1,
                                                use_stochastic_rounding=False,
                                                units=units)

    # Check which model will be used (0 for L1MET Model, 1 for DeepMET Model)
    if t_mode == 0:
        keras_model.compile(optimizer='adam', loss=custom_loss, metrics=['mean_absolute_error', 'mean_squared_error'])
        verbose = 1
    elif t_mode == 1:
        optimizer = optimizers.Adam(lr=1., clipnorm=1.)
        keras_model.compile(loss=custom_loss, optimizer=optimizer,
                            metrics=['mean_absolute_error', 'mean_squared_error'])
        verbose = 1

    # Run training
    print(keras_model.summary())

    start_time = time.time()  # check start time
    history = keras_model.fit(trainGenerator,
                              epochs=epochs,
                              verbose=verbose,  # switch to 1 for more verbosity
                              validation_data=validGenerator,
                              callbacks=get_callbacks(path_out, len(trainGenerator), batch_size))
    end_time = time.time()  # check end time

    predict_test = keras_model.predict(testGenerator) * normFac
    all_PUPPI_pt = []
    Yr_test = []
    for (Xr, Yr) in tqdm.tqdm(testGenerator):
        puppi_pt = np.sum(Xr[1], axis=1)
        all_PUPPI_pt.append(puppi_pt)
        Yr_test.append(Yr)
    
    predict_weights, pxpy = keras_model_weights.predict(testGenerator)
    predict_weights = predict_weights * normFac
    pxpy_puppi = pxpy
    pxpy = pxpy * normFac * predict_weights
    pt_puppi = np.sqrt(pxpy_puppi[:,:,0]**2 + pxpy_puppi[:,:,1]**2)
    pt = np.sqrt(pxpy[:,:,0]**2 + pxpy[:,:,1]**2)
    print(np.shape(pt))
    pt = pt.flatten()
    predict_weights = predict_weights.flatten()
    px_Y = testGenerator[0][1][:,0]
    py_Y = testGenerator[0][1][:,0]
    pt_Y = np.sqrt(px_Y**2 + py_Y**2)

    print("pt puppi:  ", pt_puppi[5])
    print("pt X:  ", pt[5])
    print("pt Y:  ", pt_Y[5])
    print("pt puppi:  ", pt_puppi[32])
    print("pt X:  ", pt[32])
    print("pt Y:  ", pt_Y[32])
    print("pt puppi:  ", pt_puppi[19])
    print("pt X:  ", pt[19])
    print("pt Y:  ", pt_Y[19])
    print("pt puppi:  ", pt_puppi[43])
    print("pt X:  ", pt[43])
    print("pt Y:  ", pt_Y[43])


    bins = np.linspace(1e-10,400,40)
    pt_bin = np.digitize(pt, bins)
    weights_binned_avg = []
    for i in range(1,41):
        bin_i = predict_weights[np.where(pt_bin==i)]
        bin_i = np.absolute(np.average(bin_i))
        weights_binned_avg.append(bin_i)


    plt.style.use(hep.style.CMS)
    plt.figure(figsize=(10, 16))
    figure, axis = plt.subplots(1, 2)
    axis[0].plot(pt, predict_weights, '.')
    axis[0].set_title('weights vs pt (unprofiled)')
    axis[0].set(xlabel='pt', ylabel='weights')
    
    axis[1].plot(bins, weights_binned_avg, '.')
    axis[1].set_title('weights vs pt (profiled)')
    axis[1].set(xlabel='pt', ylabel='weights')
    plt.savefig(f'{path_out}weights_pt.png')
    plt.close()



    PUPPI_pt = normFac * np.concatenate(all_PUPPI_pt)
    Yr_test = normFac * np.concatenate(Yr_test)

    test(Yr_test, predict_test, PUPPI_pt, path_out)

    fi = open("{}time.txt".format(path_out), 'w')

    fi.write("Working Time (s) : {}".format(end_time - start_time))
    fi.write("Working Time (m) : {}".format((end_time - start_time)/60.))

    fi.close()


def train_loadAllData(args):
    # general setup
    maxNPF = args.maxNPF
    n_features_pf = 6
    n_features_pf_cat = 2
    normFac = 1.
    epochs = args.epochs
    batch_size = args.batch_size
    preprocessed = True
    t_mode = args.mode
    inputPath = args.input
    path_out = args.output
    quantized = args.quantized
    units = list(map(int, args.units))
    compute_ef = args.compute_edge_feat
    model = args.model
    edge_list = args.edge_features

    # Read inputs
    # convert root files to h5 and store in same location
    h5files = []
    for ifile in glob(os.path.join(f'{inputPath}', '*.root')):
        h5file_path = ifile.replace('.root', '.h5')
        if not os.path.isfile(h5file_path):
            os.system(f'python convertNanoToHDF5_L1triggerToDeepMET.py -i {ifile} -o {h5file_path}')
        h5files.append(h5file_path)

    # It may be desireable to set specific files as the train, test, valid data sets
    # For now I keep train.py used: selection from a list of indicies

    Xorg, Y = read_input(h5files)
    if maxNPF < 100:
        order = Xorg[:, :, 0].argsort(axis=1)[:, ::-1]
        shape = np.shape(Xorg)
        for x in range(shape[0]):
            Xorg[x, :, :] = Xorg[x, order[x], :]
        Xorg = Xorg[:, 0:maxNPF, :]
    Y = Y / -normFac

    N = maxNPF
    Nr = N*(N-1)

    receiver_sender_list = [i for i in itertools.product(range(N), range(N)) if i[0] != i[1]]
    Xi, Xp, Xc1, Xc2 = preProcessing(Xorg, normFac)

    if compute_ef == 1:
        eta = Xi[:, :, 1]
        phi = Xi[:, :, 2]
        pt = Xi[:, :, 0]
        if ('m2' in edge_list):
            px = Xp[:, :, 0]
            py = Xp[:, :, 1]
            pz = pt*np.sinh(eta)
            energy = np.sqrt(px**2 + py**2 + pz**2)
            p4 = np.stack((energy, px, py, pz), axis=-1)
        receiver_sender_list = [i for i in itertools.product(range(N), range(N)) if i[0] != i[1]]
        edge_idx = np.array(receiver_sender_list)
        edge_stack = []
        if ('dR' in edge_list) or ('kT' in edge_list):
            eta1 = eta[:, edge_idx[:, 0]]
            phi1 = phi[:, edge_idx[:, 0]]
            eta2 = eta[:, edge_idx[:, 1]]
            phi2 = phi[:, edge_idx[:, 1]]
            dR = deltaR_calc(eta1, phi1, eta2, phi2)
            edge_stack.append(dR)
        if ('kT' in edge_list) or ('z' in edge_list):
            pt1 = pt[:, edge_idx[:, 0]]
            pt2 = pt[:, edge_idx[:, 1]]
            if ('kT' in edge_list):
                kT = kT_calc(pt1, pt2, dR)
                edge_stack.append(kT)
            if ('z' in edge_list):
                z = z_calc(pt1, pt2)
                edge_stack.append(z)
        if ('m2' in edge_list):
            p1 = p4[:, edge_idx[:, 0], :]
            p2 = p4[:, edge_idx[:, 1], :]
            m2 = mass2_calc(p1, p2)
            edge_stack.append(m2)
        ef = np.stack(edge_stack, axis=-1)
        Xc = [Xc1, Xc2]
        # dimension parameter for keras model
        emb_input_dim = {i: int(np.max(Xc[i][0:1000])) + 1 for i in range(n_features_pf_cat)}
        # Prepare training/val data
        Xc = [Xc1, Xc2]
        Yr = Y
        Xr = [Xi, Xp] + Xc + [ef]

    else:
        Xi, Xp, Xc1, Xc2 = preProcessing(Xorg, normFac)
        Xc = [Xc1, Xc2]

        emb_input_dim = {
            i: int(np.max(Xc[i][0:1000])) + 1 for i in range(n_features_pf_cat)
        }

        # Prepare training/val data
        Yr = Y
        Xr = [Xi, Xp] + Xc

    indices = np.array([i for i in range(len(Yr))])
    indices_train, indices_test = train_test_split(indices, test_size=1./7., random_state=7)
    indices_train, indices_valid = train_test_split(indices_train, test_size=1./6., random_state=7)
    # roughly the same split as the data generator workflow (train:valid:test=5:1:1)

    Xr_train = [x[indices_train] for x in Xr]
    Xr_test = [x[indices_test] for x in Xr]
    Xr_valid = [x[indices_valid] for x in Xr]
    Yr_train = Yr[indices_train]
    Yr_test = Yr[indices_test]
    Yr_valid = Yr[indices_valid]

    # Load training model
    if quantized is None:
        if model == 'dense_embedding':
            keras_model = dense_embedding(n_features=n_features_pf,
                                          emb_out_dim=2,
                                          n_features_cat=n_features_pf_cat,
                                          activation='tanh',
                                          embedding_input_dim=emb_input_dim,
                                          number_of_pupcandis=maxNPF,
                                          t_mode=t_mode,
                                          with_bias=False,
                                          units=units)

        elif model == 'graph_embedding':
            keras_model = graph_embedding(n_features=n_features_pf,
                                          emb_out_dim=2,
                                          n_features_cat=n_features_pf_cat,
                                          activation='tanh',
                                          embedding_input_dim=emb_input_dim,
                                          number_of_pupcandis=maxNPF,
                                          units=units,
                                          compute_ef=compute_ef,
                                          edge_list=edge_list)

    else:
        logit_total_bits = int(quantized[0])
        logit_int_bits = int(quantized[1])
        activation_total_bits = int(quantized[0])
        activation_int_bits = int(quantized[1])

        keras_model = dense_embedding_quantized(n_features=n_features_pf,
                                                emb_out_dim=2,
                                                n_features_cat=n_features_pf_cat,
                                                activation_quantizer='quantized_relu',
                                                embedding_input_dim=emb_input_dim,
                                                number_of_pupcandis=maxNPF,
                                                t_mode=t_mode,
                                                with_bias=False,

                                                logit_quantizer='quantized_bits',
                                                logit_total_bits=logit_total_bits,
                                                logit_int_bits=logit_int_bits,
                                                activation_total_bits=activation_total_bits,
                                                activation_int_bits=activation_int_bits,
                                                alpha=1,
                                                use_stochastic_rounding=False,
                                                units=units)

    # Check which model will be used (0 for L1MET Model, 1 for DeepMET Model)
    if t_mode == 0:
        keras_model.compile(optimizer='adam', loss=custom_loss, metrics=['mean_absolute_error', 'mean_squared_error'])
        verbose = 1
    elif t_mode == 1:
        optimizer = optimizers.Adam(lr=1., clipnorm=1.)
        keras_model.compile(loss=custom_loss, optimizer=optimizer,
                            metrics=['mean_absolute_error', 'mean_squared_error'])
        verbose = 1

    # Run training
    print(keras_model.summary())

    start_time = time.time()  # check start time
    history = keras_model.fit(Xr_train,
                              Yr_train,
                              epochs=epochs,
                              batch_size=batch_size,
                              verbose=verbose,  # switch to 1 for more verbosity
                              validation_data=(Xr_valid, Yr_valid),
                              callbacks=get_callbacks(path_out, len(Yr_train), batch_size))

    end_time = time.time()  # check end time

    predict_test = keras_model.predict(Xr_test) * normFac
    PUPPI_pt = normFac * np.sum(Xr_test[1], axis=1)
    Yr_test = normFac * Yr_test

    test(Yr_test, predict_test, PUPPI_pt, path_out)

    fi = open("{}time.txt".format(path_out), 'w')

    fi.write("Working Time (s) : {}".format(end_time - start_time))
    fi.write("Working Time (m) : {}".format((end_time - start_time)/60.))

    fi.close()


def main():
    time_path = time.strftime('%Y-%m-%d', time.localtime(time.time()))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--workflowType',
        action='store',
        type=str,
        required=True,
        choices=[
            'dataGenerator',
            'loadAllData'],
        help='designate wheather youre using the data generator or loading all data into memory ')
    parser.add_argument('--input', action='store', type=str, required=True, help='designate input file path')
    parser.add_argument('--output', action='store', type=str, required=True, help='designate output file path')
    parser.add_argument('--mode', action='store', type=int, required=True, choices=[0, 1], help='0 for L1MET, 1 for DeepMET')
    parser.add_argument('--epochs', action='store', type=int, required=False, default=100, help='number of epochs to train for')
    parser.add_argument('--batch-size', action='store', type=int, required=False, default=1024, help='batch size')
    parser.add_argument('--quantized', action='store', required=False, nargs='+', help='optional argument: flag for quantized model and specify [total bits] [int bits]; empty for normal model')
    parser.add_argument('--units', action='store', required=False, nargs='+', help='optional argument: specify number of units in each layer (also sets the number of layers)')
    parser.add_argument('--model', action='store', required=False, choices=['dense_embedding', 'graph_embedding', 'node_select'], default='dense_embedding', help='optional argument: model')
    parser.add_argument('--compute-edge-feat', action='store', type=int, required=False, choices=[0, 1], default=0, help='0 for no edge features, 1 to include edge features')
    parser.add_argument('--maxNPF', action='store', type=int, required=False, default=100, help='maximum number of PUPPI candidates')
    parser.add_argument('--edge-features', action='store', required=False, nargs='+', help='which edge features to use (i.e. dR, kT, z, m2)')

    args = parser.parse_args()
    workflowType = args.workflowType

    os.makedirs(args.output, exist_ok=True)

    if workflowType == 'dataGenerator':
        train_dataGenerator(args)
    elif workflowType == 'loadAllData':
        train_loadAllData(args)


if __name__ == "__main__":
    main()
