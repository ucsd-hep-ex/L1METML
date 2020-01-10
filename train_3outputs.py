import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python import ops
import keras
import numpy as np
import tables
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import argparse
from models_3outputs import dense
import math
import keras.backend as K

def huber_loss(y_true, y_pred, delta=1.0):
    error = y_pred - y_true
    abs_error = K.abs(error)
    quadratic = K.minimum(abs_error, delta)
    linear = abs_error - quadratic
    return 0.5 * K.square(quadratic) + delta * linear

def mean_absolute_relative_error(y_true, y_pred):
    if not K.is_tensor(y_pred):
        y_pred = K.constant(y_pred)
    y_true = K.cast(y_true, y_pred.dtype)
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true),
                                            K.epsilon(),
                                            None))
    return K.mean(diff, axis=-1)

def mean_squared_relative_error(y_true, y_pred):
    if not K.is_tensor(y_pred):
        y_pred = K.constant(y_pred)
    y_true = K.cast(y_true, y_pred.dtype)
    diff = K.square((y_true - y_pred) / K.clip(K.square(y_true),
                                            K.epsilon(),
                                            None))
    return K.mean(diff, axis=-1)

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
    nevents_1 = feature_array.shape[0]
    nfeatures = feature_array.shape[1]
    ntargets = target_array.shape[1]

	# Exclude met, phi = 0 events ,,#and MET < 50 GeV
    event_zero = 0
    skip = 0
    for i in range(nevents_1):
        if (feature_array[i,10] == 0 and feature_array[i,11] == 0) or (feature_array[i,12] ==0 and feature_array[i,13] ==0):# or (target_array[i,0] < 50):
            event_zero = event_zero + 1
    feature_array_without0 = np.zeros((nevents_1 - event_zero, 14))
    target_array_without0 = np.zeros((nevents_1 - event_zero, 2))
    for i in range(nevents_1):
        if (feature_array[i,10] == 0 and feature_array[i,11] == 0) or (feature_array[i,12] ==0 and feature_array[i,13] ==0):# or (target_array[i,0] < 50):
            skip = skip + 1
            continue
        feature_array_without0[i - skip,:] = feature_array[i,:]
        target_array_without0[i - skip,:] = target_array[i,:]
    print(feature_array_without0)
    print(target_array_without0)
    nevents = feature_array_without0.shape[0]

    # Split phi into cos^2 and sin^2
    feature_array_cos = np.zeros((nevents, 21))
    target_array_cos = np.zeros((nevents, 3))

    for i in range(7):
        feature_array_cos[:,3*i] = feature_array_without0[:,2*i]
        feature_array_cos[:,(3*i)+1] = np.square(np.cos(feature_array_without0[:,(2*i)+1]))
        feature_array_cos[:,(3*i)+2] = np.square(np.sin(feature_array_without0[:,(2*i)+1]))
    target_array_cos[:,0] = target_array_without0[:,0]
    target_array_cos[:,1] = np.square(np.cos(target_array_without0[:,1]))
    target_array_cos[:,2] = np.square(np.sin(target_array_without0[:,1]))

    nfeatures = feature_array_cos.shape[1]
    ntargets = target_array_cos.shape[1]

    # fit keras model
    X = feature_array_cos
    y = target_array_cos

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


    # Set parameters for weight function
    number_of_interval = 100
    mean = y_val.shape[0]/number_of_interval


    # Devide interval
    MET_interval = np.zeros(number_of_interval)
    for i in range(y_val.shape[0]):
        for j in range(number_of_interval):
            if (5 * j <= y_val[i,0] < 5 * (j+1)):
               MET_interval[j] = MET_interval[j]+1
               
    def weight_array_function_MET(y_true):
        k = 0
        for i in range(number_of_interval):
            if 5 * i <= y_true < 5 * (i+1):
                break
            k = MET_interval[i]
        if k == 0:
            k = 1
        return mean/k

    weight_array_MET= np.zeros(y_test.shape[0])
    for i in range(y_test.shape[0]):
        weight_array_MET[i] = weight_array_function_MET(y_val[i,0])

    def weight_loss_MET(y_true, y_pred):
        return K.mean(math_ops.square((y_pred - y_true))*weight_array_MET, axis=-1)

    keras_model = dense(nfeatures, ntargets)

    keras_model.compile(optimizer='adam', loss=[weight_loss_MET, 'mean_squared_error', 'mean_squared_error'], 
                        loss_weights = [1., 500., 500], metrics=['mean_absolute_error'])
    print(keras_model.summary())

    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    model_checkpoint = ModelCheckpoint('keras_model_best.h5', monitor='val_loss', save_best_only=True)
    callbacks = [early_stopping, model_checkpoint]

    keras_model.fit(X_train, [y_train[:,0], y_train[:,1], y_train[:,2]], batch_size=1024, 
					epochs=100, validation_data=(X_val, [y_val[:,0], y_val[:,1], y_val[:, 2]]), shuffle=True,
                    callbacks = callbacks)

    keras_model.load_weights('keras_model_best.h5')
    
    predict_test = keras_model.predict(X_test)
    predict_test = np.concatenate(predict_test,axis=1)
    print(y_test)
    print(predict_test)

    def print_res(gen_met, predict_met, name='Met_res.pdf'):
        rel_err = (predict_met - gen_met)
        plt.figure()          
        plt.hist(rel_err, bins=np.linspace(-1., 1., 50+1))
        plt.xlabel("Abs. err.")
        plt.ylabel("Events")
        plt.figtext(0.25, 0.90,'CMS',fontweight='bold', wrap=True, horizontalalignment='right', fontsize=14)
        plt.figtext(0.35, 0.90,'preliminary', style='italic', wrap=True, horizontalalignment='center', fontsize=14) 
        plt.savefig(name)
	
    #print_res(y_test[:,0], predict_test[:,0], name = 'MET_pt_abs.pdf')
    print_res(y_test[:,1], predict_test[:,1], name = 'MET_cos_abs.pdf')
    print_res(y_test[:,2], predict_test[:,2], name = 'MET_sin_abs.pdf')
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
        
    args = parser.parse_args()
    main(args)
