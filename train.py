import tensorflow
from tensorflow.python.ops import math_ops
import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping
import keras.backend as K
import numpy as np
import tables
import matplotlib.pyplot as plt
import argparse
from models import dense
import math
from Write_MET_binned_histogram import *
from get_jet import * 

def huber_loss(y_true, y_pred, delta=1.0):
    error = y_pred - y_true
    abs_error = K.abs(error)
    quadratic = K.minimum(abs_error, delta)
    linear = abs_error - quadratic
    return 0.5 * K.square(quadratic) + delta * linear

def weight_loss_function(number_of_bin, val_array, min_, max_):
    binning = (max_ - min_)/number_of_bin
    val_events = val_array.shape[0]
    mean = val_events/number_of_bin
    global MET_interval
    MET_interval = np.zeros(number_of_bin)

    for i in range(val_events):
        for j in range(number_of_bin):
            if (binning * j <= math.sqrt(val_array[i,0] ** 2 + val_array[i,1] ** 2) < binning * (j + 1)):
                MET_interval[j] = MET_interval[j] + 1
                break

    def allocate_weight(val_):
        k = 0
        for i in range(number_of_bin):
            if (binning * i <= val_ < binning * (i + 1)):
                break
        k = MET_interval[i]
        if k == 0:
            k = 1
        return mean/k

    weight_array = np.zeros(val_events)
    
    for i in range(val_events):
        weight_array[i] = allocate_weight(math.sqrt(val_array[i,0] ** 2 + val_array[i,1] ** 2))

    return weight_array


def main(args):

    file_path = 'input_MET.h5'
    features = ['L1CHSMet_pt', 'L1CHSMet_phi',
                'L1CaloMet_pt', 'L1CaloMet_phi',
                'L1PFMet_pt', 'L1PFMet_phi',
                'L1PuppiMet_pt', 'L1PuppiMet_phi',
                'L1TKMet_pt', 'L1TKMet_phi',
                'L1TKV5Met_pt', 'L1TKV5Met_phi',
                'L1TKV6Met_pt', 'L1TKV6Met_phi']
    features_jet = ['L1PuppiJets_pt', 'L1PuppiJets_phi', 'L1PuppiJets_eta']

    targets = ['genMet_pt', 'genMet_phi']
    targets_jet = ['GenJets_pt', 'GenJets_phi', 'GenJets_eta']

    ## Set number of jets you will use
    number_of_jets = 3

    feature_MET_array, target_array = get_features_targets(file_path, features, targets)
    feature_jet_array = get_jet_features_targets(file_path, features_jet, number_of_jets)
    nMETs = feature_MET_array.shape[1]
    feature_array = np.concatenate((feature_MET_array, feature_jet_array), axis=1)

    nevents = feature_array.shape[0]
    nfeatures = feature_array.shape[1]
    ntargets = target_array.shape[1]


    # Exclude puppi met < 100 GeV events
    ## Set PUPPI MET min, max cut

    PupMET_cut = 0
    PupMET_cut_max = 500

    mask1= (feature_array[:,0] < PupMET_cut)
    feature_array = feature_array[~mask1]
    target_array = target_array[~mask1]

    mask2= (feature_array[:,0] > PupMET_cut_max)
    feature_array = feature_array[~mask2]
    target_array = target_array[~mask2]

    nevents = feature_array.shape[0]

    # Convert feature from pt, phi to px, py
    feature_array_xy = np.zeros((nevents, nfeatures))
    for i in range(nMETs):
        if i%2 == 0:
            feature_array_xy[:,i] = feature_array[:,i] * np.cos(feature_array[:,i+1])
        if i%2 == 1:
            feature_array_xy[:,i] = feature_array[:,i-1] * np.sin(feature_array[:,i])

    for i in range(3*number_of_jets):
        if i%3 == 0:
            feature_array_xy[:,i+nMETs] = feature_array[:,i+nMETs] * np.cos(feature_array[:,i+nMETs+1])
        if i%3 == 1:
            feature_array_xy[:,i+nMETs] = feature_array[:,i+nMETs-1] * np.sin(feature_array[:,i+nMETs])
        if i%3 == 2:
            feature_array_xy[:,i+nMETs] = feature_array[:,i+nMETs]
    
    # Convert target from pt phi to px, py
    target_array_xy = np.zeros((nevents, ntargets))

    target_array_xy[:,0] = target_array[:,0] * np.cos(target_array[:,1])
    target_array_xy[:,1] = target_array[:,0] * np.sin(target_array[:,1])

    
    # Split datas into train, validation, test set
    X = feature_array_xy
    y = target_array_xy
    
    fulllen = nevents
    tv_frac = 0.10
    tv_num = math.ceil(fulllen*tv_frac)
    splits = np.cumsum([fulllen-2*tv_num,tv_num,tv_num])
    splits = [int(s) for s in splits]

    X_train = X[0:splits[0]]
    X_val = X[splits[1]:splits[2]]
    X_test = X[splits[0]:splits[1]]

    y_train = y[0:splits[0]]
    y_val = y[splits[1]:splits[2]]
    y_test = y[splits[0]:splits[1]]



    # Make weight loss function
    weight_array = weight_loss_function(20, y_train, 0, 500)



    # Set keras train model
    keras_model = dense(nfeatures, ntargets)

    keras_model.compile(optimizer='adam', loss=['mean_squared_error', 'mean_squared_error'], 
                        loss_weights = None, metrics=['mean_absolute_error'])
    print(keras_model.summary())

    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    model_checkpoint = ModelCheckpoint('keras_model_best.h5', monitor='val_loss', save_best_only=True)
    callbacks = [early_stopping, model_checkpoint]



    # fit keras
    keras_model.fit(X_train, [y_train[:,:1], y_train[:,1:]], batch_size=1024, sample_weight = [weight_array, weight_array], 
                epochs=100, validation_data=(X_val, [y_val[:,:1], y_val[:,1:]]), shuffle=True,
                callbacks = callbacks)



    # load created weights
    keras_model.load_weights('keras_model_best.h5')
    
    predict_test = keras_model.predict(X_test)
    predict_test = np.concatenate(predict_test,axis=1)



    # convert px py into pt phi
    test_events = predict_test.shape[0]

    predict_phi = np.zeros((test_events, 2))
    y_test_phi = np.zeros((test_events, 2))
	
    predict_phi[:,0] = np.sqrt((predict_test[:,0]**2 + predict_test[:,1]**2))
    for i in range(test_events):
        if 0 < predict_test[i,1]:
            predict_phi[i,1] = math.acos(predict_test[i,0]/predict_phi[i,0])
        if predict_test[i,1] < 0:
            predict_phi[i,1] = -math.acos(predict_test[i,0]/predict_phi[i,0])
    
        y_test_phi[:,0] = np.sqrt((y_test[:,0]**2 + y_test[:,1]**2))
    for i in range(test_events):
        if 0 < y_test[i,1]:
            y_test_phi[i,1] = math.acos(y_test[i,0]/y_test_phi[i,0])
        if y_test[i,1] < 0:
            y_test_phi[i,1] = -math.acos(y_test[i,0]/y_test_phi[i,0])

    Write_MET_binned_histogram(predict_phi, y_test_phi, 20, 0, 100, 400, name='histogram_all_no100cut.root')

    MET_rel_error(predict_phi[:,0], y_test_phi[:,0], name='rel_error_weight.png')
    MET_binned_predict_mean(predict_phi[:,0], y_test_phi[:,0], 20, 0, 500, 0, '.', name='predict_mean.png')
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
        
    args = parser.parse_args()
    main(args)
