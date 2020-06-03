import tensorflow 
from tensorflow.python.ops import math_ops
import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping
import keras.backend as K
import numpy as np
import tables
import matplotlib.pyplot as plt
import argparse
from models import dense, dense_conv
import math
import setGPU
from Write_MET_binned_histogram import MET_rel_error, MET_binned_predict_mean
from get_jet import get_features_targets, get_jet_features

def huber_loss(y_true, y_pred, delta=1.0):
    error = y_pred - y_true
    abs_error = K.abs(error)
    quadratic = K.minimum(abs_error, delta)
    linear = abs_error - quadratic
    return 0.5 * K.square(quadratic) + delta * linear

def weight_loss_function(number_of_bin, val_array, min_, max_):
    binning = (max_ - min_)/number_of_bin
    bins = np.linspace(min_,max_,number_of_bin+1)

    val_events = val_array.shape[0]
    mean = val_events/number_of_bin
    global MET_interval
    MET_interval = np.zeros(number_of_bin)

    met = np.sqrt(val_array[:,0] ** 2 + val_array[:,1] ** 2)    
    MET_interval, _ = np.histogram(met,bins)

    bin_indices = np.digitize(met,bins)-1
    bin_indices = np.clip(bin_indices,0,number_of_bin-1) # make last bin an overflow

    weight_array = np.full(val_events, mean)/MET_interval[bin_indices]

    return weight_array


def main(args):

    file_path = 'data/input_MET.h5'
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

    # Set number of jets you will use
    number_of_jets = 10

    feature_MET_array, target_array = get_features_targets(file_path, features, targets)
    feature_jet_array = get_jet_features(file_path, features_jet, number_of_jets)
    nMETs = int(feature_MET_array.shape[1]/2)
    print(nMETs)

    nevents = target_array.shape[0]
    nmetfeatures = feature_MET_array.shape[1]
    njets = feature_jet_array.shape[1]
    njetfeatures = feature_jet_array.shape[2]
    ntargets = target_array.shape[1]

    # Exclude puppi met < 100 GeV events
    # Set PUPPI MET min, max cut

    PupMET_cut = 0
    PupMET_cut_max = 500

    mask = (feature_MET_array[:,0] > PupMET_cut) & (feature_MET_array[:,0] < PupMET_cut_max)
    feature_MET_array = feature_MET_array[mask]
    feature_jet_array = feature_jet_array[mask]
    target_array = target_array[mask]

    nevents = target_array.shape[0]

    # Convert feature from pt, phi to px, py
    feature_MET_array_xy = np.zeros((nevents, nmetfeatures))
    for i in range(nMETs):
        feature_MET_array_xy[:,2*i] =   feature_MET_array[:,2*i] * np.cos(feature_MET_array[:,2*i+1])
        feature_MET_array_xy[:,2*i+1] = feature_MET_array[:,2*i] * np.sin(feature_MET_array[:,2*i+1])

    feature_jet_array_xy = np.zeros((nevents, njets, njetfeatures))
    for i in range(number_of_jets):
        feature_jet_array_xy[:,i,0] = feature_jet_array[:,i,0] * np.cos(feature_jet_array[:,i,1])
        feature_jet_array_xy[:,i,1] = feature_jet_array[:,i,0] * np.sin(feature_jet_array[:,i,1])
        feature_jet_array_xy[:,i,2] = feature_jet_array[:,i,2]
    
    # Convert target from pt phi to px, py
    target_array_xy = np.zeros((nevents, ntargets))

    target_array_xy[:,0] = target_array[:,0] * np.cos(target_array[:,1])
    target_array_xy[:,1] = target_array[:,0] * np.sin(target_array[:,1])
    
    # Split datas into train, validation, test set
    X = [feature_MET_array_xy, feature_jet_array_xy]
    y = target_array_xy
    
    fulllen = nevents
    tv_frac = 0.10
    tv_num = math.ceil(fulllen*tv_frac)
    splits = np.cumsum([fulllen-2*tv_num,tv_num,tv_num])
    splits = [int(s) for s in splits]

    X_train = [xi[0:splits[0]] for xi in X]
    X_val = [xi[splits[1]:splits[2]] for xi in X]
    X_test = [xi[splits[0]:splits[1]] for xi in X]

    y_train = y[0:splits[0]]
    y_val = y[splits[1]:splits[2]]
    y_test = y[splits[0]:splits[1]]

    # Make weight loss function
    weight_array = weight_loss_function(20, y_train, 0, 500)

    # Set keras train model
    keras_model = dense_conv(nmetfeatures, njets, njetfeatures, ntargets)

    keras_model.compile(optimizer='adam', loss=['mean_squared_error', 'mean_squared_error'], 
                        loss_weights = None, metrics=['mean_absolute_error'])
    print(keras_model.summary())

    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    model_checkpoint = ModelCheckpoint('keras_model_best.h5', monitor='val_loss', save_best_only=True)
    callbacks = [early_stopping, model_checkpoint]


    # fit keras
    keras_model.fit(X_train, 
                    [y_train[:,:1], y_train[:,1:]], 
                    batch_size=1024, 
                    sample_weight=[weight_array, weight_array], 
                    epochs=100, 
                    validation_data=(X_val, [y_val[:,:1], y_val[:,1:]]), 
                    shuffle=True,
                    callbacks=callbacks)


    # load created weights
    keras_model.load_weights('keras_model_best.h5')
    
    predict_test = keras_model.predict(X_test)
    predict_test = np.concatenate(predict_test,axis=1)

    # convert px py into pt phi
    test_events = predict_test.shape[0]

    predict_phi = np.zeros((test_events, 2))
    y_test_phi = np.zeros((test_events, 2))
	
    predict_phi[:,0] = np.sqrt((predict_test[:,0]**2 + predict_test[:,1]**2))
    predict_phi[:,1] = np.sign(predict_phi[:,1])*np.arccos(predict_test[:,0]/predict_phi[:,0])

    y_test_phi[:,0] = np.sqrt((y_test[:,0]**2 + y_test[:,1]**2))
    y_test_phi[:,1] = np.sign(y_test[:,1])*np.arccos(y_test[:,0]/y_test_phi[:,0])

    #Write_MET_binned_histogram(predict_phi, y_test_phi, 20, 0, 100, 400, name='histogram_all_no100cut.root')

    MET_rel_error(predict_phi[:,0], y_test_phi[:,0], name='rel_error_weight.png')
    MET_binned_predict_mean(predict_phi[:,0], y_test_phi[:,0], 20, 0, 500, 0, '.', name='predict_mean.png')
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
        
    args = parser.parse_args()
    main(args)
