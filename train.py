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

#Import custom modules

from Write_MET_binned_histogram import *
from cyclical_learning_rate import CyclicLR
from models import *
from utils import *
from loss import custom_loss
#from epoch_all import epoch_all
from DataGenerator import DataGenerator

def trainFrom_Root(args):


    # general setup

    maxNPF = 100
    n_features_pf = 6
    n_features_pf_cat = 2
    normFac = 1.
    epochs = 1
    batch_size = 1024
    preprocessed = True
    t_mode = args.mode
    inputPath = args.input
    path_out = args.output

    # Make directory for output
    try:
        if not os.path.exists(path_out):
            os.makedirs(path_out)
    except OSError:
        print ('Creating directory' + path_out)
	

    # load in data 3 generators; each recieve different data sets

    # on lxplus
    #data in '/afs/cern.ch/work/d/daekwon/public/L1PF_110X/CMSSW_11_1_2/src/FastPUPPI/NtupleProducer/python/TTbar_PU200_110X_1M'
    
    # on prp
    #data in '../../../l1metmlvol/TTbar_PU200_110X_1M'
    
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
        
    keras_model = dense_embedding(n_features = n_features_pf, n_features_cat=n_features_pf_cat, n_dense_layers=5, activation='tanh',embedding_input_dim = trainGenerator.emb_input_dim, number_of_pupcandis = 100, t_mode = t_mode, with_bias=False)


    # Check which model will be used (0 for L1MET Model, 1 for DeepMET Model)

    if t_mode == 0:
        keras_model.compile(optimizer='adam', loss=custom_loss, metrics=['mean_absolute_error', 'mean_squared_error'])
        #keras_model.compile(optimizer='adam', loss=['mean_squared_error', 'mean_squared_error'], metrics=['mean_absolute_error', 'mean_squared_error'])
        verbose = 1

    if t_mode == 1:
        optimizer = optimizers.Adam(lr=1., clipnorm=1.)
        #keras_model.compile(loss=custom_loss, optimizer=optimizer, 
        keras_model.compile(loss=['mean_absolute_error', 'mean_squared_error'], optimizer=optimizer, 
                       metrics=['mean_absolute_error', 'mean_squared_error'])
        verbose = 1
        

    # Set model config

      # early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

    csv_logger = CSVLogger(f"{path_out}loss_history.log")

      # model checkpoint callback
      # this saves our model architecture + parameters into model.h5

    model_checkpoint = ModelCheckpoint(f'{path_out}/model.h5', monitor='val_loss',
                                       verbose=0, save_best_only=True,
                                       save_weights_only=False, mode='auto',
                                       period=1)

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=4, min_lr=0.000001, cooldown=3, verbose=1)

    lr_scale = 1.
    clr = CyclicLR(base_lr=0.0003*lr_scale, max_lr=0.001*lr_scale, step_size=len(trainGenerator.y)/batch_size, mode='triangular2')

    stop_on_nan = tensorflow.keras.callbacks.TerminateOnNaN()
    
    # Run training

    print(keras_model.summary())
    #plot_model(keras_model, to_file=f'{path_out}/model_plot.png', show_shapes=True, show_layer_names=True)

    start_time = time.time() # check start time
    history = keras_model.fit(trainGenerator,
                        epochs=epochs,
                        verbose=verbose,  # switch to 1 for more verbosity
                        validation_data=validGenerator,
                        callbacks=[early_stopping, clr, stop_on_nan, csv_logger, model_checkpoint],#, reduce_lr], #, lr,   reduce_lr],
                       )
    end_time = time.time() # check end time
    
    keras_model.load_weights(f'{path_out}/model.h5')

    predict_test = keras_model.predict(testGenerator) * normFac
    all_PUPPI_pt = []
    Yr_test = []
    for (Xr, Yr) in tqdm.tqdm(testGenerator):
        Xi = Xr[0]
        puppi_pt = -np.sum(Xi[:,:,4:6],axis=1)
        all_PUPPI_pt.append(puppi_pt)
        Yr_test.append(Yr)

    all_PUPPI_pt = normFac * np.concatenate(all_PUPPI_pt)
    Yr_test = normFac * np.concatenate(Yr_test)
    #Xr_test = normFac * Xr_test
    #test_events = Xr_test[0].shape[0]
    
    MakePlots(Yr_test, predict_test, all_PUPPI_pt, path_out = path_out)
    
    Yr_test = convertXY2PtPhi(Yr_test)
    predict_test = convertXY2PtPhi(predict_test)
    PUPPI_pt = convertXY2PtPhi(PUPPI_pt)

    MET_rel_error_opaque(predict_test[:,0], PUPPI_pt[:,0], Yr_test[:,0], name=''+path_out+'rel_error_opaque.png')
    MET_binned_predict_mean_opaque(predict_test[:,0], PUPPI_pt[:,0], Yr_test[:,0], 20, 0, 500, 0, '.', name=''+path_out+'PrVSGen.png')
    extract_result(predict_test, Yr_test, path_out, 'TTbar', 'ML')
    extract_result(PUPPI_pt, Yr_test, path_out, 'TTbar', 'PU')
    fi = open("{}time.txt".format(path_out), 'w')

    fi.write("Working Time (s) : {}".format(end_time - start_time))
    fi.write("Working Time (m) : {}".format((end_time - start_time)/60.))

    fi.close()


# Configuration

'''if __name__ == "__main__":

    time_path = time.strftime('%Y-%m-%d', time.localtime(time.time()))
    path = "./result/"+time_path+"_PUPPICandidates/"

    parser = argparse.ArgumentParser()
    #parser.add_argument('--input', action='store', type=str, required=True, help='designate input file path')
    parser.add_argument('--output', action='store', type=str, default='{}'.format(path), help='designate output file path')
    parser.add_argument('--mode', action='store', type=int, required=True, help='0 for L1MET, 1 for DeepMET')
        
    args = parser.parse_args()
    main(args) '''
    
    
def trainFrom_h5(args):

    # general setup

    maxNPF = 100
    n_features_pf = 6
    n_features_pf_cat = 2
    normFac = 1.
    epochs = 100
    batch_size = 1024
    preprocessed = True
    t_mode = args.mode
    inputPath = args.input
    path_out = args.output

    # Make directory for output
    try:
        if not os.path.exists(path_out):
            os.makedirs(path_out)
    except OSError:
        print ('Creating directory' + path_out)

    # Read inputs
    
    # convert root files to h5 and store in same location
    i =0
    for file in os.listdir(inputPath):
        if '.root' in file:
            os.system(f'python convertNanoToHDF5_L1triggerToDeepMET.py -i {inputPath}/{file} -o {inputPath}/set{i}.h5')
            i += 1
    # place h5 file names into a .txt file
    writeFile= open(f'{inputPath}/h5files.txt',"w+")
    for file in os.listdir(inputPath):
        if '.h5' in file:
            writeFile.write(f'{inputPath}/{file}\n')
    writeFile.close()
    h5files = f'{inputPath}/h5files.txt'
    h5filesList = f'{inputPath}/h5files.txt'
    
    # It may be desireable to set specific files as the train, test, valid data sets
    # For now I keep train.py used: selection from a list of indicies

    Xorg, Y = read_input(h5filesList)
    Y = Y / -normFac

    Xi, Xc1, Xc2 = preProcessing(Xorg, normFac)
    Xc = [Xc1, Xc2]
    
    emb_input_dim = {
        i:int(np.max(Xc[i][0:1000])) + 1 for i in range(n_features_pf_cat)
    }
    #print(emb_input_dim)


    # Prepare training/val data
    Yr = Y
    Xr = [Xi] + Xc

    # remove events True pT < 50 GeV
    Yr_pt = convertXY2PtPhi(Yr)
    #mask1 = (Yr_pt[:,0] > 50.)
    #Yr = Yr[mask1]
    #Xr = [x[mask1] for x in Xr]

    # check the number of events higher than 300 GeV
    mask2 = (Yr_pt[:,0] > 300)
    Yr_pt = Yr_pt[mask2]
    #print("# of events higher than 300 GeV : {}".format(Yr_pt.shape[0]))

    indices = np.array([i for i in range(len(Yr))])
    #print(indices)
    indices_train, indices_valid = train_test_split(indices, test_size=0.2, random_state= 7)
    indices_train, indices_test = train_test_split(indices_train, test_size=0.2, random_state=7)

    Xr_train = [x[indices_train] for x in Xr]
    Xr_test = [x[indices_test] for x in Xr]
    Xr_valid = [x[indices_valid] for x in Xr]
    Yr_train = Yr[indices_train]
    Yr_test = Yr[indices_test]
    Yr_valid = Yr[indices_valid]

    # Load training model

    keras_model = dense_embedding(n_features = n_features_pf, n_features_cat=n_features_pf_cat, n_dense_layers=5, activation='tanh', embedding_input_dim = emb_input_dim, number_of_pupcandis = 100, t_mode = t_mode, with_bias=False)


    # Check which model will be used (0 for L1MET Model, 1 for DeepMET Model)

    if t_mode == 0:
        keras_model.compile(optimizer='adam', loss=custom_loss, metrics=['mean_absolute_error', 'mean_squared_error'])
        #keras_model.compile(optimizer='adam', loss=['mean_squared_error', 'mean_squared_error'], metrics=['mean_absolute_error', 'mean_squared_error'])
        verbose = 1

    if t_mode == 1:
        optimizer = optimizers.Adam(lr=1., clipnorm=1.)
        #keras_model.compile(loss=custom_loss, optimizer=optimizer,
        keras_model.compile(loss=['mean_absolute_error', 'mean_squared_error'], optimizer=optimizer,
                       metrics=['mean_absolute_error', 'mean_squared_error'])
        verbose = 1
        
    # Set model config

      # early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

    csv_logger = CSVLogger(f"{path_out}loss_history.log")

      # model checkpoint callback
      # this saves our model architecture + parameters into model.h5

    model_checkpoint = ModelCheckpoint(f'{path_out}/model.h5', monitor='val_loss',
                                       verbose=0, save_best_only=True,
                                       save_weights_only=False, mode='auto',
                                       period=1)

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=4, min_lr=0.000001, cooldown=3, verbose=1)

    lr_scale = 1.
    clr = CyclicLR(base_lr=0.0003*lr_scale, max_lr=0.001*lr_scale, step_size=len(Y)/batch_size, mode='triangular2')

    stop_on_nan = tensorflow.keras.callbacks.TerminateOnNaN()

    #print(Xr_train[0].shape[-1])
    #print(Xr_train[1].shape[-1])
    #print(Xr_train[2].shape[-1])
    
    
    # Run training
    print(keras_model.summary())
    #plot_model(keras_model, to_file=f'{path_out}/model_plot.png', show_shapes=True, show_layer_names=True)

    start_time = time.time() # check start time

    history = keras_model.fit(Xr_train,
                        Yr_train,
                        epochs=epochs,
                        batch_size = batch_size,
                        verbose=verbose,  # switch to 1 for more verbosity
                        validation_data=(Xr_valid, Yr_valid),
                        callbacks=[early_stopping, clr, stop_on_nan, csv_logger, model_checkpoint],#, reduce_lr], #, lr,   reduce_lr],
                       )

    end_time = time.time() # check end time
    

    keras_model.load_weights(f'{path_out}/model.h5')

    predict_test = keras_model.predict(Xr_test)
    PUPPI_pt = normFac * np.sum(Xr_test[0][:,:,4:6], axis=1)
    predict_test = predict_test *normFac
    Yr_test = normFac * Yr_test
    #Xr_valid = normFac * Xr_valid

    #test_events = Xr_valid[0].shape[0]

    MakePlots(Yr_test, predict_test, PUPPI_pt, path_out = path_out)

    Yr_test = convertXY2PtPhi(Yr_test)
    predict_test = convertXY2PtPhi(predict_test)
    PUPPI_pt = convertXY2PtPhi(PUPPI_pt)

    MET_rel_error_opaque(predict_test[:,0], PUPPI_pt[:,0], Yr_test[:,0], name=''+path_out+'rel_error_opaque.png')
    MET_binned_predict_mean_opaque(predict_test[:,0], PUPPI_pt[:,0], Yr_test[:,0], 20, 0, 500, 0, '.', name=''+path_out+'PrVSGen.png')
    extract_result(predict_test, Yr_test, path_out, 'TTbar', 'ML')
    extract_result(PUPPI_pt, Yr_test, path_out, 'TTbar', 'PU')
    fi = open("{}time.txt".format(path_out), 'w')

    fi.write("Working Time (s) : {}".format(end_time - start_time))
    fi.write("Working Time (m) : {}".format((end_time - start_time)/60.))

    fi.close()


# Configuration

''' if __name__ == "__main__":

    time_path = time.strftime('%Y-%m-%d', time.localtime(time.time()))
    path = "./result/"+time_path+"_PUPPICandidates/"

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', action='store', type=str, required=True, help='designate input file path')
    parser.add_argument('--output', action='store', type=str, default='{}'.format(path), help='designate output file path')
    parser.add_argument('--mode', action='store', type=int, required=True, help='0 for L1MET, 1 for DeepMET')
        
    args = parser.parse_args()
    main(args) '''

def main():
    time_path = time.strftime('%Y-%m-%d', time.localtime(time.time()))
    path = "./result/"+time_path+"_PUPPICandidates/"

    # 4 arguments
    # train from h5 or root?
    # root file path (input)
    # output path
    # mode 0 or 1
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataType', action='store', type=str, required=True, help='designate input file path')
    parser.add_argument('--input', action='store', type=str, required=True, help='designate input file path')
    parser.add_argument('--output', action='store', type=str, required=True, help='designate output file path')
    parser.add_argument('--mode', action='store',   type=int, required=True, help='0 for L1MET, 1 for DeepMET')
    args = parser.parse_args()

    dataType = args.dataType

    if dataType == 'h5':
        trainFrom_h5(args)

    if dataType == 'root':
        trainFrom_Root(args)

if __name__ == "__main__":
    main()
