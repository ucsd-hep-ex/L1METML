import tensorflow
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
from models import dense
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
	# *************************
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
    target_scale = np.array([100., 1.])
    nevents = feature_array.shape[0]
    nfeatures = feature_array.shape[1]
    ntargets = target_array.shape[1]
    #target_array = target_array/target_scale

    # Convert feature from pt, phi to px, py
    feature_array_xy = np.zeros((nevents, nfeatures))
    
    for i in range(14):
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
    
    # Set parameters for weight function
    number_of_interval = 200
    mean = y_test.shape[0]/number_of_interval

    # Devide interval
    x_interval = np.zeros(200)
    for i in range(y_test.shape[0]):
        for j in range(200):
            if 5 * (j-100) <= y_test[i,0] < 5 * (j-99):
               x_interval[j] = x_interval[j]+1
               
    def weight_array_function_x(y_true):
        k = 0
        for i in range(200):
            if 5*(i-100)<=y_true<5*(j-99):
                break
            k = x_interval[j]
        if k == 0:
            k = 1
        return mean/k
    weight_array_x = np.zeros(y_test.shape[0])
    for i in range(y_test.shape[0]):
        weight_array_x[i] = weight_array_function_x(y_test[i,0])

    def weight_loss_x(y_true, y_pred):
        return K.mean(math_ops.square((y_pred - y_true)*weight_array_x), axis=-1)

    y_interval = np.zeros(200)
    for i in range(y_test.shape[0]):
        for j in range(200):
            if 5 * (j-100) <= y_test[i,1] < 5 * (j-99):
               y_interval[j] = y_interval[j]+1

    def weight_array_function_y(y_true):
        k = 0
        for i in range(200):
            if 5*(i-100)<=y_true<5*(j-99):
                break
            k = y_interval[j]
        if k == 0:
            k = 1
        return mean/k
    weight_array_y = np.zeros(y_test.shape[0])
    for i in range(y_test.shape[0]):
        weight_array_y[i] = weight_array_function_y(y_test[i,0])

    def weight_loss_y(y_true, y_pred):
        return K.mean(math_ops.square((y_pred - y_true)*weight_array_y), axis=-1)

    keras_model = dense(nfeatures, ntargets)

    keras_model.compile(optimizer='adam', loss=[weight_loss_x, weight_loss_y], 
                        loss_weights = [10., 1.], metrics=['mean_absolute_error'])
    print(keras_model.summary())

    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    model_checkpoint = ModelCheckpoint('keras_model_best.h5', monitor='val_loss', save_best_only=True)
    callbacks = [early_stopping, model_checkpoint]



    keras_model.fit(X_train, [y_train[:,:1], y_train[:,1:]], batch_size=1024, 
					epochs=100, validation_data=(X_val, [y_val[:,:1], y_val[:,1:]]), shuffle=True,
                    callbacks = callbacks)

	#***********************
    keras_model.load_weights('keras_model_best.h5')
    
    predict_test = keras_model.predict(X_test)
    predict_test = np.concatenate(predict_test,axis=1)
    print(y_test)
    print(predict_test)

    def print_res(gen_met, predict_met, name='Met_res.pdf'):
        rel_err = (predict_met - gen_met)
        plt.figure()
        plt.hist(rel_err, bins=np.linspace(-200., 200., 50+1))
        plt.xlabel("(predict - true)")
        plt.ylabel("Events")
        plt.figtext(0.25, 0.90,'CMS',fontweight='bold', wrap=True, horizontalalignment='right', fontsize=14)
        plt.figtext(0.35, 0.90,'preliminary', style='italic', wrap=True, horizontalalignment='center', fontsize=14) 
        plt.savefig(name)
	
    print_res(y_test[:,0], predict_test[:,0], name = 'MET_px_res.pdf')
    print_res(y_test[:,1], predict_test[:,1], name = 'MET_py_res.pdf')
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
        
    args = parser.parse_args()
    main(args)
