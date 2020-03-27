import tensorflow as tf
import keras
import numpy as np
import tables
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
import argparse
from models_3outputs import dense
import math
import keras.backend as K


def delta_phi(phi1, phi2):
    return np.mod(phi1 - phi2 + np.pi, 2*np.pi) - np.pi

def huber_loss(y_true, y_pred, delta=1.0):
    error = y_pred - y_true
    abs_error = K.abs(error)
    quadratic = K.minimum(abs_error, delta)
    linear = abs_error - quadratic
    return 0.5 * K.square(quadratic) + delta * linear

def huber_loss_relative(y_true, y_pred, delta=1.0):
    rel_error = K.abs((y_true - y_pred) / K.clip(K.abs(y_true),
                                                K.epsilon(),
                                                 None))
    quadratic = K.minimum(rel_error, delta)
    linear = rel_error - quadratic
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

def mean_squared_percentage_error(y_true, y_pred):
    if not K.is_tensor(y_pred):
        y_pred = K.constant(y_pred)
    y_true = K.cast(y_true, y_pred.dtype)
    diff = K.square((y_true - y_pred) / K.clip(K.square(y_true),
                                            K.epsilon(),
                                            None))
    return 100.*K.mean(diff, axis=-1)

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
    features = ['L1PuppiMet_pt', 'L1PuppiMet_phi',
                'L1CHSMet_pt', 'L1CHSMet_phi',
                'L1CaloMet_pt', 'L1CaloMet_phi',
                'L1PFMet_pt', 'L1PFMet_phi',
                'L1TKMet_pt', 'L1TKMet_phi',
                #'L1TKV5Met_pt', 'L1TKV5Met_phi',
                #'L1TKV6Met_pt', 'L1TKV6Met_phi',
    ]

    targets = ['genMet_pt', 'genMet_phi']

    feature_array, target_array = get_features_targets(file_path, features, targets)
    nevents_1 = feature_array.shape[0]
    nfeatures = feature_array.shape[1]
    ntargets = target_array.shape[1]

    # Exclude puppi met, phi = 0 events and gen met < 50 GeV
    mask = ( ((feature_array[:,0] == 0) & (feature_array[:,1] == 0)) | (target_array[:,0] < 50) )
    feature_array = feature_array[~mask]
    target_array = target_array[~mask]

    nevents = feature_array.shape[0]

    # Split target phi into cos^2 and sin^2
    new_target_array = np.zeros((nevents, 3))
    new_target_array[:,0] = target_array[:,0]/feature_array[:,0]
    new_target_array[:,1] = np.square(np.cos(target_array[:,1]))
    new_target_array[:,2] = np.square(np.sin(target_array[:,1]))

    target_array = new_target_array[:,0:1]

    nfeatures = feature_array.shape[1]
    ntargets = target_array.shape[1]

    # fit keras model
    X = feature_array
    y = target_array

    fulllen = nevents
    tv_frac = 0.10
    tv_num = math.ceil(fulllen*tv_frac)
    splits = np.cumsum([fulllen-2*tv_num,tv_num,tv_num])
    splits = [int(s) for s in splits]

    X_train = X[0:splits[0]]
    X_val = X[splits[1]:splits[2]]
    X_test = X[splits[0]:splits[1]]

    print(X_test)

    X_scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = X_scaler.transform(X_train)
    X_val = X_scaler.transform(X_val)
    X_test = X_scaler.transform(X_test)

    y_train = y[0:splits[0]]
    y_val = y[splits[1]:splits[2]]
    y_test = y[splits[0]:splits[1]]

    print(y_test)

    y_scaler =  preprocessing.StandardScaler().fit(y_train)
    #y_train = y_scaler.transform(y_train)
    #y_val = y_scaler.transform(y_val)
    #y_test = y_scaler.transform(y_test)

    # Set parameters for weight function
    #w_train = 1./(9.76272*np.exp(-2.04719e-02*y_train[:,0]))
    w_train = np.ones_like(y_train[:,0])

    keras_model = dense(nfeatures, ntargets)

    #keras_model.compile(optimizer='adam', loss=['mean_absolute_percentage_error', 'mean_absolute_percentage_error'],
    #                    loss_weights = [1., 0.], metrics=['mean_absolute_error','mean_squared_error'])

    loss_name = 'mean_absolute_percentage_error'
    loss = loss_name

    keras_model.compile(optimizer='adam', loss=loss,
                        metrics=['mean_absolute_error','mean_squared_error'])
    print(keras_model.summary())

    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    model_checkpoint = ModelCheckpoint('keras_model_best.h5', monitor='val_loss', save_best_only=True)
    callbacks = [early_stopping, model_checkpoint]

    keras_model.fit(X_train, y_train, batch_size=1024, #[y_train[:,0], y_train[:,1:]], batch_size=1024, 
                    epochs=100, validation_data=(X_val, y_val), shuffle=True, # [y_val[:,0], y_val[:,1:]]), shuffle=True,
                    callbacks = callbacks, sample_weight = w_train)#[w_train, w_train])

    keras_model.load_weights('keras_model_best.h5')
    
    predict_test = keras_model.predict(X_test)
    #predict_test = np.concatenate(predict_test,axis=1)

    print(X_test)
    print(y_test)
    print(predict_test)
    X_test = X_scaler.inverse_transform(X_test)
    #y_test = y_scaler.inverse_transform(y_test)
    #predict_test = y_scaler.inverse_transform(predict_test)

    print(X_test)
    print(y_test)
    print(predict_test)

    def print_res(gen_met, predict_met, name='Met_res.pdf',phi=False):
        if phi:
            gen_phi = np.arctan2(np.sqrt(gen_met[:,1]), np.sqrt(gen_met[:,0]))
            predict_phi = np.arctan2(np.sqrt(predict_met[:,1]), np.sqrt(predict_met[:,0]))
            rel_err = delta_phi(predict_phi,gen_phi)
        else:
            rel_err = (predict_met - gen_met)/gen_met
        bins=np.linspace(-1., 1., 50+1)
        from scipy.stats import norm
        (mu, sigma) = norm.fit(rel_err)
        mean = np.mean(rel_err)
        std = np.std(rel_err)
        median = np.median(rel_err)
        minus1sigma = np.quantile(rel_err,q=0.16)
        plus1sigma = np.quantile(rel_err,q=0.84)
        plt.hist(rel_err, bins=bins,alpha=0.5,label='mean=%.2f,std.=%.2f,median=%.2f,68%%=[%.2f,%.2f]'%(mean,std,median,minus1sigma,plus1sigma))
        
        bins=np.linspace(-1., 1., 500+1)
        y = norm.pdf( bins, mu, sigma)
        y /= sum(y) 
        y *= 10.*rel_err.shape[0]
        l = plt.plot(bins, y, '--', linewidth=2)

    plt.figure()          	
    print_res(y_test[:,0]*X_test[:,0], predict_test[:,0]*X_test[:,0], name = 'nnMET_res.pdf', phi=False)
    #print_res(y_test[:,0]*X_test[:,0], X_test[:,0], name = 'puppiMET_res.pdf', phi=False)
    plt.legend(loc='upper right')
    plt.xlabel("Rel. err.")
    plt.ylabel("Events")
    plt.xlim(-1,1)
    plt.ylim(0,600)
    plt.figtext(0.20, 0.90,'CMS',fontweight='bold', wrap=True, horizontalalignment='right', fontsize=14)
    plt.figtext(0.30, 0.90,'preliminary', style='italic', wrap=True, horizontalalignment='center', fontsize=14) 
    plt.figtext(0.70, 0.90,loss_name,  wrap=True, horizontalalignment='center', fontsize=11) 
    plt.savefig('nnMET_puppiMET_res.pdf')
        
    #print_res(y_test[:,1:], predict_test[:,1:], name = 'nnMET_phi_res.pdf',phi=True)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
        
    args = parser.parse_args()
    main(args)
