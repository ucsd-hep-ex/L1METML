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

def main(args):


    # general setup

    maxNPF = 100
    n_features_pf = 6
    n_features_pf_cat = 2
    normFac = 1.
    epochs = 1
    batch_size = 1024
    preprocessed = True
    t_mode = args.mode
    path_out = args.output

    # Make directory for output
    try:
        if not os.path.exists(path_out):
            os.makedirs(path_out)
    except OSError:
        print ('Creating directory' + path_out)
	

    # load in data 3 generators; each recieve different data sets

    # on lxplus
    #data = '/afs/cern.ch/work/d/daekwon/public/L1PF_110X/CMSSW_11_1_2/src/FastPUPPI/NtupleProducer/python/TTbar_PU200_110X_1M'
    # on prp
    data = '../../../l1metmlvol/TTbar_PU200_110X_1M'

    list_files_Train = [f'{data}/perfNano_TTbar_PU200.110X_set0.root' ,f'{data}/perfNano_TTbar_PU200.110X_set1.root', f'{data}/perfNano_TTbar_PU200.110X_set2.root',f'{data}/perfNano_TTbar_PU200.110X_set3.root',f'{data}/perfNano_TTbar_PU200.110X_set4.root']
    list_files_Valid = [f'{data}/perfNano_TTbar_PU200.110X_set5.root']
    list_files_Test = [f'{data}/perfNano_TTbar_PU200.110X_set6.root']
    
    trainGenerator = DataGenerator(list_files=list_files_Train,batch_size=batch_size)
    validGenerator = DataGenerator(list_files=list_files_Valid,batch_size=batch_size)
    testGenerator = DataGenerator(list_files=list_files_Test,batch_size=batch_size)
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
    
    print(Xr_train[0].shape[-1])
    print(Xr_train[1].shape[-1])
    print(Xr_train[2].shape[-1])
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

    predict_test = keras_model.predict(testGenerator)
    XList = []
    Yr_test = []
    all_px = []
    all_py = []
    #print(tqdm.tqdm.testGenerator)
    for ifile in list_files_Test:
        file = open(ifile, "r")
        line_count = 0
        for line in file:
            if line != "\n":
                line_count += 1
        file.close()
        XList.append(testGenerator.__get_features_labels(ifile, 0, line_count)[0])
        Yr_test.append(testGenerator.__get_features_labels(ifile, 0, line_count)[1])
    for X in XList:
        px = -np.sum(X[:,:,1],axis=1)
        py = -np.sum(X[:,:,2],axis=1)
        all_px.append(px)
        all_py.append(py)
    for Y in YList:
            Y = Y /(-self.normFac)
    all_px = normFac * np.concatenate(all_px)
    all_py = normFac * np.concatenate(all_py)
    predict_test = predict_test *normFac
    print(all_px.shape)
    print(all_py.shape)
    
    #Xr_test = normFac * Xr_test
    #test_events = Xr_test[0].shape[0]
    #Yr_test = normFac * Yr_test
    
    MakePlots(Yr_test, predict_test, PUPPI_pt, path_out = path_out)
    
    Yr_test = convertXY2PtPhi(-Yr_test)
    predict_test = convertXY2PtPhi(predict_test)
    PUPPI_pt = np.array([all_met_x, all_met_y]).T
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
