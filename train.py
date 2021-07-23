# Import libraries
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

#Import custom modules

from Write_MET_binned_histogram import *
from cyclical_learning_rate import CyclicLR
from models import *
from utils import *
from loss import custom_loss
from DataGenerator import DataGenerator

def get_callbacks(path_out, sample_size, batch_size):
    # early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, restore_best_weights=False)

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

    MakePlots(Yr_test, predict_test, PUPPI_pt, path_out = path_out)
    
    Yr_test = convertXY2PtPhi(Yr_test)
    predict_test = convertXY2PtPhi(predict_test)
    PUPPI_pt = convertXY2PtPhi(PUPPI_pt)

    MET_rel_error_opaque(predict_test[:,0], PUPPI_pt[:,0], Yr_test[:,0], name=''+path_out+'rel_error_opaque.png')
    MET_binned_predict_mean_opaque(predict_test[:,0], PUPPI_pt[:,0], Yr_test[:,0], 20, 0, 500, 0, '.', name=''+path_out+'PrVSGen.png')
    extract_result(predict_test, Yr_test, path_out, 'TTbar', 'ML')
    extract_result(PUPPI_pt, Yr_test, path_out, 'TTbar', 'PU')
    
    Phi_abs_error_opaque(PUPPI_pt[:,1], predict_test[:,1], Yr_test[:,1], name=path_out+'Phi_abs_err')
    Pt_abs_error_opaque(PUPPI_pt[:,0], predict_test[:,0], Yr_test[:,0],name=path_out+'Pt_abs_error')

def trainFrom_Root(args):
    # general setup
    maxNPF = 100
    n_features_pf = 6
    n_features_pf_cat = 2
    normFac = 1.
    epochs = args.epochs
    batch_size = 1024
    preprocessed = True
    t_mode = args.mode
    inputPath = args.input
    path_out = args.output
    quantized = args.quantized

    filesList = []
    for file in os.listdir(inputPath):
        if '.root' in file:
            filesList.append(f'{inputPath}{file}')
    valid_nfiles = int(.1*len(filesList))
    if valid_nfiles == 0:
        valid_nfiles = 1
    train_nfiles = len(filesList)- 2*valid_nfiles
    test_nfiles = valid_nfiles
    train_filesList = filesList[0:train_nfiles]
    valid_filesList = filesList[train_nfiles: train_nfiles+valid_nfiles]
    test_filesList = filesList[train_nfiles+valid_nfiles:test_nfiles+train_nfiles+valid_nfiles]

    trainGenerator = DataGenerator(list_files=train_filesList,batch_size=batch_size)
    validGenerator = DataGenerator(list_files=valid_filesList,batch_size=batch_size)
    testGenerator = DataGenerator(list_files=test_filesList,batch_size=batch_size)
    Xr_train, Yr_train = trainGenerator[0] # this apparenly calls all the methods, so that we can get the correct dimensions (train_generator.emb_input_dim)

    # Load training model
    if quantized == True:
    
        logit_total_bits = 16
        logit_int_bits = 6
        activation_total_bits = 16
        activation_int_bits = 6
        
        keras_model = dense_embedding_quantized(n_features = n_features_pf, n_features_cat=n_features_pf_cat, n_dense_layers=3, activation_quantizer='quantized_relu',embedding_input_dim = trainGenerator.emb_input_dim, number_of_pupcandis = 100, t_mode = t_mode, with_bias=False, logit_quantizer = 'quantized_bits', logit_total_bits=logit_total_bits, logit_int_bits=logit_int_bits, activation_total_bits=activation_total_bits, activation_int_bits=activation_int_bits, alpha='auto', use_stochastic_rounding=False)
    else:
        keras_model = dense_embedding(n_features = n_features_pf, n_features_cat=n_features_pf_cat, n_dense_layers=3, activation='tanh',embedding_input_dim = trainGenerator.emb_input_dim, number_of_pupcandis = maxNPF, t_mode = t_mode, with_bias=False)

    # Check which model will be used (0 for L1MET Model, 1 for DeepMET Model)
    if t_mode == 0:
        keras_model.compile(optimizer='adam', loss=custom_loss, metrics=['mean_absolute_error', 'mean_squared_error'])
        verbose = 1
    elif t_mode == 1:
        optimizer = optimizers.Adam(lr=1., clipnorm=1.)
        keras_model.compile(loss=custom_loss, optimizer=optimizer, 
                            metrics=['mean_absolute_error', 'mean_squared_error'])
        verbose = 1
        

    # Set model config
    # Run training

    print(keras_model.summary())

    start_time = time.time() # check start time
    history = keras_model.fit(trainGenerator,
                              epochs=epochs,
                              verbose=verbose,  # switch to 1 for more verbosity
                              validation_data=validGenerator,
                              callbacks=get_callbacks(path_out, len(trainGenerator), batch_size),
                          )
    end_time = time.time() # check end time
    
    predict_test = keras_model.predict(testGenerator) * normFac
    all_PUPPI_pt = []
    Yr_test = []
    for (Xr, Yr) in tqdm.tqdm(testGenerator):
        puppi_pt = np.sum(Xr[0][:,:,4:6],axis=1)
        all_PUPPI_pt.append(puppi_pt)
        Yr_test.append(Yr)

    PUPPI_pt = normFac * np.concatenate(all_PUPPI_pt)
    Yr_test = normFac * np.concatenate(Yr_test)
    
    test(Yr_test, predict_test, PUPPI_pt, path_out)

    fi = open("{}time.txt".format(path_out), 'w')

    fi.write("Working Time (s) : {}".format(end_time - start_time))
    fi.write("Working Time (m) : {}".format((end_time - start_time)/60.))

    fi.close()
    
def trainFrom_h5(args):
    # general setup
    maxNPF = 100
    n_features_pf = 6
    n_features_pf_cat = 2
    normFac = 1.
    epochs = args.epochs
    batch_size = 1024
    preprocessed = True
    t_mode = args.mode
    inputPath = args.input
    path_out = args.output
    quantized = args.quantized
    # Read inputs
    
    # convert root files to h5 and store in same location
    h5files = []
    for ifile in glob(f'{inputPath}*.root'):
        h5file_path = ifile.replace('.root','.h5')
        if not os.path.isfile(h5file_path):
            os.system(f'python convertNanoToHDF5_L1triggerToDeepMET.py -i {ifile} -o {h5file_path}')
        h5files.append(h5file_path)

    # It may be desireable to set specific files as the train, test, valid data sets
    # For now I keep train.py used: selection from a list of indicies

    Xorg, Y = read_input(h5files)
    Y = Y / -normFac

    Xi, Xc1, Xc2 = preProcessing(Xorg, normFac)
    Xc = [Xc1, Xc2]
    
    emb_input_dim = {
        i:int(np.max(Xc[i][0:1000])) + 1 for i in range(n_features_pf_cat)
    }

    # Prepare training/val data
    Yr = Y
    Xr = [Xi] + Xc

    indices = np.array([i for i in range(len(Yr))])
    indices_train, indices_test = train_test_split(indices, test_size=0.2, random_state= 7)
    indices_train, indices_valid = train_test_split(indices_train, test_size=0.2, random_state=7)

    Xr_train = [x[indices_train] for x in Xr]
    Xr_test = [x[indices_test] for x in Xr]
    Xr_valid = [x[indices_valid] for x in Xr]
    Yr_train = Yr[indices_train]
    Yr_test = Yr[indices_test]
    Yr_valid = Yr[indices_valid]

    # Load training model
    if quantized == True:
    
        logit_total_bits = 16
        logit_int_bits = 6
        activation_total_bits = 16
        activation_int_bits = 6
        
        keras_model = dense_embedding_quantized(n_features = n_features_pf, n_features_cat=n_features_pf_cat, n_dense_layers=3, activation_quantizer='quantized_tanh',embedding_input_dim = emb_input_dim, number_of_pupcandis = 100, t_mode = t_mode, with_bias=False, logit_quantizer = 'quantized_bits', logit_total_bits=logit_total_bits, logit_int_bits=logit_int_bits, activation_total_bits=activation_total_bits, activation_int_bits=activation_int_bits, alpha=1, use_stochastic_rounding=False)
        
    else:
        keras_model = dense_embedding(n_features = n_features_pf, n_features_cat=n_features_pf_cat, n_dense_layers=3, activation='tanh', embedding_input_dim = emb_input_dim, number_of_pupcandis = maxNPF, t_mode = t_mode, with_bias=False)


    # Check which model will be used (0 for L1MET Model, 1 for DeepMET Model)
    if t_mode == 0:
        keras_model.compile(optimizer='adam', loss=custom_loss, metrics=['mean_absolute_error', 'mean_squared_error'])
        verbose = 1
    elif t_mode == 1:
        optimizer = optimizers.Adam(lr=1., clipnorm=1.)
        keras_model.compile(loss=custom_loss, optimizer=optimizer,
                            metrics=['mean_absolute_error', 'mean_squared_error'])
        verbose = 1
        
    # Set model config

    # Run training
    print(keras_model.summary())

    start_time = time.time() # check start time
    history = keras_model.fit(Xr_train,
                              Yr_train,
                              epochs=epochs,
                              batch_size = batch_size,
                              verbose=verbose,  # switch to 1 for more verbosity
                              validation_data=(Xr_valid, Yr_valid),
                              callbacks=get_callbacks(path_out, len(Yr_train), batch_size)
                          )
    end_time = time.time() # check end time
    
    predict_test = keras_model.predict(Xr_test) * normFac
    PUPPI_pt = normFac * np.sum(Xr_test[0][:,:,4:6], axis=1)
    Yr_test = normFac * Yr_test

    test(Yr_test, predict_test, PUPPI_pt, path_out)

    fi = open("{}time.txt".format(path_out), 'w')

    fi.write("Working Time (s) : {}".format(end_time - start_time))
    fi.write("Working Time (m) : {}".format((end_time - start_time)/60.))

    fi.close()

def main():
    time_path = time.strftime('%Y-%m-%d', time.localtime(time.time()))
    path = "./result/"+time_path+"_PUPPICandidates/"

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataType', action='store', type=str, required=True, choices=['h5', 'root'], help='designate input file path')
    parser.add_argument('--input', action='store', type=str, required=True, help='designate input file path')
    parser.add_argument('--output', action='store', type=str, required=True, help='designate output file path')
    parser.add_argument('--mode', action='store', type=int, required=True, choices=[0, 1], help='0 for L1MET, 1 for DeepMET')
    parser.add_argument('--epochs', action='store', type=int, required=False, default=100)
    parser.add_argument('--quantized', action='store_true', required=False, help='flag for quantized model, empty for normal model')
    
    args = parser.parse_args()
    dataType = args.dataType

    os.makedirs(args.output,exist_ok=True)

    if dataType == 'h5':
        trainFrom_h5(args)
    elif dataType == 'root':
        trainFrom_Root(args)

if __name__ == "__main__":
    main()
