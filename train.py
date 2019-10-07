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

    keras_model = dense(nfeatures, ntargets)

    keras_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    print(keras_model.summary())

    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    model_checkpoint = ModelCheckpoint('keras_model_best.h5', monitor='val_loss', save_best_only=True)
    callbacks = [early_stopping, model_checkpoint]

    # fit keras model
    X = feature_array
    y = target_array

    fulllen = nevents
    tv_frac = 0.10
    tv_num = math.ceil(fulllen*tv_frac)
    splits = np.cumsum([fulllen-2*tv_num,tv_num,tv_num])
    splits = [int(s) for s in splits]
    print(splits)

    X_train = X[0:splits[0]]
    X_val = X[splits[1]:splits[2]]
    X_test = X[splits[0]:splits[1]]

    y_train = y[0:splits[0]]
    y_val = y[splits[1]:splits[2]]
    y_test = y[splits[0]:splits[1]]
    
    keras_model.fit(X_train, y_train, batch_size=1024, 
                    epochs=10, validation_data=(X_val, y_val), shuffle=True,
                    callbacks = callbacks)

    keras_model.load_weights('keras_model_best.h5')
    
    predict_test = keras_model.predict(X_test)

    
    plt.figure()       
    plt.hist(predict_test[:,0]-y_test[:,0], bins=np.linspace(0.5, 1.5, 100+1))
    plt.xlabel("Rel. err. for genMet_pt")
    plt.ylabel("Events")
    plt.figtext(0.25, 0.90,'CMS',fontweight='bold', wrap=True, horizontalalignment='right', fontsize=14)
    plt.figtext(0.35, 0.90,'preliminary', style='italic', wrap=True, horizontalalignment='center', fontsize=14) 
    plt.savefig('Met_res.pdf')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
        
    args = parser.parse_args()
    main(args)
