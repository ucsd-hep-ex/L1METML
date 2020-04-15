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
from ROOT import *
from Write_MET_binned_histogram import *

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

def get_features_targets(file_name, features, targets):
    # load file
    h5file = tables.open_file(file_name, 'r')
    nevents = getattr(h5file.root,features[0]).shape[0]
    ntargets = len(targets)
    nfeatures = len(features)

    # allocate arrays
    feature_array = np.zeros((nevents,nfeatures))
    target_array = np.zeros((nevents,ntargets))

    # load feature arrays
    for (i, feat) in enumerate(features):
        feature_array[:,i] = getattr(h5file.root,feat)[:]
    # load target arrays
    for (i, targ) in enumerate(targets):
        target_array[:,i] = getattr(h5file.root,targ)[:]

    h5file.close()
    return feature_array,target_array


def main(args):
    file_path = 'input_MET.h5'
    features = ['L1CHSMet_pt', 'L1CHSMet_phi',
                'L1CaloMet_pt', 'L1CaloMet_phi',
                'L1PFMet_pt', 'L1PFMet_phi',
                'L1PuppiMet_pt', 'L1PuppiMet_phi',
                'L1TKMet_pt', 'L1TKMet_phi',
                'L1TKV5Met_pt', 'L1TKV5Met_phi',
                'L1TKV6Met_pt', 'L1TKV6Met_phi']

    targets = ['genMet_pt', 'genMet_phi']

    feature_array, target_array = get_features_targets(file_path, features, targets)
    nevents = feature_array.shape[0]
    nfeatures = feature_array.shape[1]
    ntargets = target_array.shape[1]


    
    # Exclude Gen met < 100 GeV events
        # if genMET cut is not applied : 0 , applied : 1
    genMET_cut = 1

    if genMET_cut == 1:
        genMET_check = '100cut'
        event_zero = 0
        skip = 0
        for i in range(nevents):
            if (target_array[i,0] < 100):
                event_zero = event_zero + 1

        feature_array_without0 = np.zeros((nevents - event_zero, nfeatures))
        target_array_without0 = np.zeros((nevents - event_zero, 2))

        for i in range(nevents):
            if (target_array[i,0] < 100):
                skip = skip + 1
                continue
            feature_array_without0[i - skip,:] = feature_array[i,:]
            target_array_without0[i - skip,:] = target_array[i,:]

        nevents = feature_array_without0.shape[0]
        feature_array = np.zeros((nevents, nfeatures)) 
        target_array = np.zeros((nevents, 2)) 
        feature_array = feature_array_without0
        target_array = target_array_without0
    else:
        genMET_check = 'nocut'

    

    # Convert feature from pt, phi to px, py
    feature_array_xy = np.zeros((nevents, nfeatures))
    for i in range(nfeatures):
        if i%2 == 0:
            for j in range(nevents):
                feature_array_xy[j,i] = feature_array[j,i] * math.cos(feature_array[j,i+1])
        if i%2 == 1:
            for j in range(nevents):
                feature_array_xy[j,i] = feature_array[j,i-1] * math.sin(feature_array[j,i])

    # Convert target from pt phi to px, py
    target_array_xy = np.zeros((nevents, ntargets))

    for i in range(nevents):
        target_array_xy[i,0] = target_array[i,0] * math.cos(target_array[i,1])
        target_array_xy[i,1] = target_array[i,0] * math.sin(target_array[i,1])


    
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

    def weight_loss(y_true, y_pred):
        return K.mean(math_ops.square((y_pred - y_true))*weight_array, axis=-1)



    # Set keras train model
    keras_model = dense(nfeatures, ntargets)

    keras_model.compile(optimizer='adam', loss=['mean_squared_error', 'mean_squared_error'], 
                        loss_weights = [weight_array, weight_array], metrics=['mean_absolute_error'])
    print(keras_model.summary())

    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    model_checkpoint = ModelCheckpoint('keras_model_best.h5', monitor='val_loss', save_best_only=True)
    callbacks = [early_stopping, model_checkpoint]



    # fit keras
    keras_model.fit(X_train, [y_train[:,:1], y_train[:,1:]], batch_size=1024, sample_weights = [weight_array, weight_array], 
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
	
    for i in range(test_events):
        predict_phi[i,0] = math.sqrt((predict_test[i,0]**2 + predict_test[i,1]**2))
        if 0 < predict_test[i,1]:
            predict_phi[i,1] = math.acos(predict_test[i,0]/predict_phi[i,0])
        if predict_test[i,1] < 0:
            predict_phi[i,1] = -math.acos(predict_test[i,0]/predict_phi[i,0])
    
    for i in range(test_events):
        y_test_phi[i,0] = math.sqrt((y_test[i,0]**2 + y_test[i,1]**2))
        if 0 < y_test[i,1]:
            y_test_phi[i,1] = math.acos(y_test[i,0]/y_test_phi[i,0])
        if y_test[i,1] < 0:
            y_test_phi[i,1] = -math.acos(y_test[i,0]/y_test_phi[i,0])

    Write_MET_binned_histogram(predict_phi, y_test_phi, 20, 0, 100, 400, name='histogram_all_no100cut.root')

    MET_rel_error(predict_phi[:,0], y_test_phi[:,0], name='rel_error_weight.png')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
        
    args = parser.parse_args()
    main(args)
