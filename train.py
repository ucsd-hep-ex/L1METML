import tensorflow 
from tensorflow.python.ops import math_ops
import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
import keras.backend as K
import numpy as np
import tables
import matplotlib.pyplot as plt
import argparse
import math
import setGPU
import time
import os
from Write_MET_binned_histogram import * # MET_rel_error, MET_abs_error, MET_binned_predict_mean, Phi_abs_error, dist, histo_2D, Write_MET_binned_histogram # Write_MET_binned_histogram function needs ROOT. Maybe ROOT version over 6.22 supports for py3?
from get_jet import get_features_targets, get_jet_features
from models import *# dense, dense_conv, conv, deepmetlike
from utils import custom_loss, flatting


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


    file_path = 'data/input_MET_PupCandi.h5'
    features = ['L1CHSMet_pt', 'L1CHSMet_phi',
                'L1CaloMet_pt', 'L1CaloMet_phi',
                'L1PFMet_pt', 'L1PFMet_phi',
                'L1PuppiMet_pt', 'L1PuppiMet_phi',
                'L1TKMet_pt', 'L1TKMet_phi',
                'L1TKV5Met_pt', 'L1TKV5Met_phi',
                'L1TKV6Met_pt', 'L1TKV6Met_phi']

    features_jet = ['L1PuppiJets_pt', 'L1PuppiJets_phi', 
                    'L1PuppiJets_eta', 'L1PuppiJets_mass']

    features_pupcandi = ['L1PuppiCands_pt','L1PuppiCands_phi','L1PuppiCands_eta', 'L1PuppiCands_puppiWeight',
                         'L1PuppiCands_charge','L1PuppiCands_pdgId']

    targets = ['genMet_pt', 'genMet_phi']
    targets_jet = ['GenJets_pt', 'GenJets_phi', 'GenJets_eta']

    # Set number of jets and pupcandis you will use
    number_of_jets = 20
    number_of_pupcandis = 100

    feature_MET_array, target_array = get_features_targets(file_path, features, targets)
    feature_jet_array = get_jet_features(file_path, features_jet, number_of_jets)
    feature_pupcandi_array = get_jet_features(file_path, features_pupcandi, number_of_pupcandis)
    nMETs = int(feature_MET_array.shape[1]/2)

    nevents = target_array.shape[0]
    nmetfeatures = feature_MET_array.shape[1]
    njets = feature_jet_array.shape[1]
    njetfeatures = feature_jet_array.shape[2]
    npupcandis = feature_pupcandi_array.shape[1]
    npupcandifeatures = feature_pupcandi_array.shape[2]
    ntargets = target_array.shape[1]

    # Exclude puppi met < +PupMET_cut+ GeV events
    # Set PUPPI MET min, max cut

    PupMET_cut = 0
    PupMET_cut_max = 500
    weights_path = ''+str(PupMET_cut)+'cut'

    mask = (feature_MET_array[:,6] > PupMET_cut) & (feature_MET_array[:,0] < PupMET_cut_max)
    print("mask : {}".format(mask))
    feature_MET_array = feature_MET_array[mask]
    feature_jet_array = feature_jet_array[mask]
    feature_pupcandi_array = feature_pupcandi_array[mask]
    target_array = target_array[mask]

    nevents = target_array.shape[0]

    # Exclude Gen met < +TarMET_cut+ GeV events
    # Set Gen MET min, max cut

    TarMET_cut = 5
    TarMET_cut_max = 500

    mask1 = (target_array[:,0] > TarMET_cut) & (target_array[:,0] < TarMET_cut_max)
    feature_MET_array = feature_MET_array[mask1]
    feature_jet_array = feature_jet_array[mask1]
    feature_pupcandi_array = feature_pupcandi_array[mask1]
    target_array = target_array[mask1]

    nevents = target_array.shape[0]
    

    # Flatting the sample

    mask2 = flatting(target_array)
    feature_MET_array = feature_MET_array[mask2]
    feature_jet_array = feature_jet_array[mask2]
    feature_pupcandi_array = feature_pupcandi_array[mask2]
    target_array = target_array[mask2]

    nevents = target_array.shape[0]

    # Shuffle again
    shuffler = np.random.permutation(len(target_array))
    print(shuffler)
    feature_MET_array = feature_MET_array[shuffler]
    feature_jet_array = feature_jet_array[shuffler,:,:]
    feature_pupcandi_array = feature_pupcandi_array[shuffler,:,:]
    target_array = target_array[shuffler]
    

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
        feature_jet_array_xy[:,i,3] = feature_jet_array[:,i,3]

    
    feature_pupcandi_array_xy = np.zeros((nevents, npupcandis, npupcandifeatures))
    for i in range(number_of_pupcandis):
        feature_pupcandi_array_xy[:,i,0] = feature_pupcandi_array[:,i,0] * np.cos(feature_pupcandi_array[:,i,1])
        feature_pupcandi_array_xy[:,i,1] = feature_pupcandi_array[:,i,0] * np.sin(feature_pupcandi_array[:,i,1])
        feature_pupcandi_array_xy[:,i,2] = feature_pupcandi_array[:,i,2]
        feature_pupcandi_array_xy[:,i,3] = feature_pupcandi_array[:,i,3]
        feature_pupcandi_array_xy[:,i,4] = feature_pupcandi_array[:,i,4]
        feature_pupcandi_array_xy[:,i,5] = feature_pupcandi_array[:,i,5]
    
    
	#labeling
    A=feature_pupcandi_array_xy[:,:,4:5]
    A=np.where(A==-1,3,A)
    A=np.where(A==0,4,A)
    A=np.where(A==1,5,A)

    B=feature_pupcandi_array_xy[:,:,5:]
    B=np.where(B==-211,0,B)
    B=np.where(B==-130,1,B)
    B=np.where(B==-22,2,B)
    B=np.where(B==-13,3,B)
    B=np.where(B==-11,4,B)
    B=np.where(B==11,5,B)
    B=np.where(B==13,6,B)
    B=np.where(B==22,7,B)
    B=np.where(B==130,8,B)
    B=np.where(B==211,9,B)
    
    
    # Convert target from pt phi to px, py
    target_array_xy = np.zeros((nevents, ntargets))

    target_array_xy[:,0] = target_array[:,0] * np.cos(target_array[:,1])
    target_array_xy[:,1] = target_array[:,0] * np.sin(target_array[:,1])
    
    # Split datas into train, validation, test set

    # for test!!! (applying embedding)################
    inputs = np.concatenate((feature_pupcandi_array[:,:,0:1], feature_pupcandi_array_xy[:,:,0:2]), axis=-1)
    inputs_cat0 = A 
    inputs_cat1 = B
    Xc = [inputs_cat0, inputs_cat1]

    X = [inputs]+[inputs_cat0]+[inputs_cat0]

    embedding_input_dim = {i : int(np.max(Xc[i])) + 1 for i in range(2)}

    #X = [feature_MET_array_xy, feature_jet_array_xy, feature_pupcandi_array_xy[:,:,(0,1)]]
    y = target_array_xy
    A = feature_MET_array[:,(6,7)]
    
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

    A_train = A[0:splits[0]]
    A_val = A[splits[1]:splits[2]]
    A_test = A[splits[0]:splits[1]]

    # Make weight loss function
    weight_array = weight_loss_function(20, y_train, 0, 300)
    # Set keras train model (and correct input)

    # met+jet-based model
    #keras_model = dense_conv(nmetfeatures, njets, njetfeatures, ntargets)
    #X_train = X_train[:2]
    #X_val = X_val[:2]
    #X_val = X_test[:2]

    # met+jet+PupCandi-based model
    #keras_model = dense_conv_all(nmetfeatures, njets, njetfeatures, npupcandis, npupcandifeatures, ntargets)

    # only-met-based model
    #keras_model = dense(nmetfeatures, ntargets); 
    #X_train = X_trai#n[0]
    #X_val = X_val[0]
    #X_test = X_test[0]

    # only-jet-based models
    #keras_model = conv(njets, njetfeatures, ntargets)
    #keras_model = deepmetlike(njets, njetfeatures, ntargets)
    #X_train = X_train[1]
    #X_val = X_val[1]
    #X_test = X_test[1]

    # only-Candi-based models
    #keras_model = conv(npupcandis, npupcandifeatures, ntargets)
    #keras_model = conv(npupcandis, npupcandifeatures, ntargets)
    #keras_model = deepmetlike(njets, njetfeatures, ntargets)
    #X_train = X_train[2]
    #X_val = X_val[2]
    #X_test = X_test[2]

    # test!!! Dense embedding ############
    print('feature number = {}'.format(inputs.shape[-1]))
    #keras_model = dense_embedding(n_features=inputs.shape[-1], n_features_cat=2, n_dense_layers=3, activation='tanh', embedding_input_dim = embedding_input_dim)
    keras_model = CNN_embedding(n_features=inputs.shape[-1], n_features_cat=2, n_dense_layers=3, activation='tanh', embedding_input_dim = embedding_input_dim)


    # print variables
    print()
    print("# \t\tGen MET cut\t :\t %.1f" % TarMET_cut)
    print("# \t\tPUPPI MET cut\t :\t %.1f" % PupMET_cut)
    print("# \t\tNumber of event\t : \t %d" % nevents)
    print("# \t\tNumber of training event\t : \t %d" % A_train.shape[0])
    print("# \t\tNumber of test event\t : \t %d" % A_test.shape[0])
    print()

    
    # Set the path where the result plots and model weights will be saved.
    time_path = time.strftime('%Y_%m_%d', time.localtime(time.time()))
    # path for various GenMET cut
    #path='./result/GenMET_cut_result_'+time_path+'/'+str(TarMET_cut)+'-'+str(TarMET_cut_max)+'/'
    # path for various PuppiMET cut
    path='./result/result_'+time_path+'/CNN/'+str(PupMET_cut)+'-'+str(PupMET_cut_max)+'/'
    #path='./result/result_'+time_path+'/'+str(PupMET_cut)+'-'+str(PupMET_cut_max)+'/'
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print ('Creating directory' + path)


    keras_model.compile(optimizer='adam', loss=custom_loss, metrics=['mean_absolute_error', 'mean_squared_error'])
    print(keras_model.summary())

    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    model_checkpoint = ModelCheckpoint(''+path+'keras_model_best.h5', monitor='val_loss', save_best_only=True)
    csv_logger = CSVLogger('loss_data.log')
    callbacks = [early_stopping, model_checkpoint, csv_logger]


    # fit keras
     
    keras_model.fit(X_train, y_train, 
                    batch_size=1024, 
                    #sample_weight=[weight_array, weight_array], 
                    epochs=200, 
                    validation_data=(X_val, y_val), 
                    shuffle=True,
                    callbacks=callbacks)
    

    # load created weights
    keras_model.load_weights(''+path+'keras_model_best.h5')
    
    predict_test = keras_model.predict(X_test)
    #predict_test = np.concatenate(predict_test,axis=1)

    # convert px py into pt phi
    test_events = predict_test.shape[0]

    predict_phi = np.zeros((test_events, 2))
    y_test_phi = np.zeros((test_events, 2))
	
    predict_phi[:,0] = np.sqrt((predict_test[:,0]**2 + predict_test[:,1]**2))
    predict_phi[:,1] = np.sign(predict_test[:,1])*np.arccos(predict_test[:,0]/predict_phi[:,0])

    y_test_phi[:,0] = np.sqrt((y_test[:,0]**2 + y_test[:,1]**2))
    y_test_phi[:,1] = np.sign(y_test[:,1])*np.arccos(y_test[:,0]/y_test_phi[:,0])


    print(predict_phi)
    print(y_test_phi)
    


    # Create rootfile with histograms to make resolution plot
    Write_MET_binned_histogram(predict_phi, y_test_phi, 20, 0, 100, 400, name=''+path+'histogram_predicted_'+str(PupMET_cut)+'.root')
    Write_MET_binned_histogram(A_test, y_test_phi, 20, 0, 100, 400, name='histogram_puppi_'+str(PupMET_cut)+'.root')

    # For plots
    MET_rel_error(predict_phi[:,0], y_test_phi[:,0], name=''+path+'rel_error.png')
    #MET_abs_error(predict_phi[:,0], y_test_phi[:,0], name=''+path+'rel_abs.png')
    #Phi_abs_error(predict_phi[:,1], y_test_phi[:,1], name=''+path+'Phi_error.png')
    MET_binned_predict_mean(predict_phi[:,0], y_test_phi[:,0], 20, 0, 500, 0, '.', name=''+path+'PrVSGen.png')
    #dist(predict_phi[:,0], name=''+path+'predict_dist.png')
    dist(y_test_phi[:,0], name=''+path+'Gen_dist.png')
    histo_2D(predict_phi[:,0], y_test_phi[:,0], name=''+path+'2D_histo.png')

    #MET_rel_error(A_test[:,0], y_test_phi[:,0], name='rel_error_weight.png')
    #MET_binned_predict_mean(A_test[:,0], y_test_phi[:,0], 20, 0, 500, 0, '.', name='predict_mean.png')
    extract_result(predict_phi, y_test_phi, path, PupMET_cut, PupMET_cut_max)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
        
    args = parser.parse_args()
    main(args)
