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
#import setGPU
import time
import os
from Write_MET_binned_histogram import *#MET_rel_error_opaque, Phi_abs_error, Phi_abs_error_opaque, response_ab, response_parallel
from cyclical_learning_rate import CyclicLR
from models import *# dense, dense_conv, conv, deepmetlike
from utils import *# load_input, custom_loss, flatting


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

    # Set the path where the result plots and model weights will be saved.

    preprocessed = args.input 
    target_array_xy, feature_MET_array, feature_MET_array_xy, feature_pupcandi_array_xy = load_input(preprocessed)
		
    path = args.output
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print ('Creating directory' + path)

    
    pupcandi_phi = np.zeros((target_array_xy.shape[0], 2))
    pupcandi_phi = np.sum(feature_pupcandi_array_xy[:,:,:2], axis=1)

    target_array_xy_phi = convertXY2PtPhi(target_array_xy)
    pupcandi_phi = convertXY2PtPhi(pupcandi_phi)

		# Test plot for gen MET dist
    dist(target_array_xy_phi[:,0], -150, 500, 100, name=''+path+'Gen_dist_before_training.png')
    dist(feature_MET_array[:,6], -150, 500, 100, name=''+path+'Gen_dist_before_training.png')
    MET_rel_error(pupcandi_phi[:,0], target_array_xy_phi[:,0], name=''+path+'rel_error_before_training.png')
    MET_rel_error(feature_MET_array[:,6], target_array_xy_phi[:,0], name=''+path+'rel_error_before_training.png')




    # Exclude Gen met < +TarMET_cut+ GeV events
    # Set Gen MET min, max cut
    
    PupMET_cut = -500
    PupMET_cut_max = 500
    TarMET_cut = -500
    TarMET_cut_max = 500

    mask1 = ((target_array_xy_phi[:,0]) < TarMET_cut_max)
    feature_MET_array = feature_MET_array[mask1]
    feature_MET_array_xy = feature_MET_array_xy[mask1]
    #feature_PFcandi_array = feature_PFcandi_array[mask1]
    feature_pupcandi_array_xy = feature_pupcandi_array_xy[mask1]
    target_array_xy = target_array_xy[mask1]
    
    nevents = target_array_xy.shape[0]
    print(nevents)
    print(nevents)
    nmetfeatures = feature_MET_array_xy.shape[1]
    njets = feature_pupcandi_array_xy.shape[1]
    njetfeatures = feature_pupcandi_array_xy.shape[2]
    npupcandis = feature_pupcandi_array_xy.shape[1]
    npupcandifeatures = feature_pupcandi_array_xy.shape[2]
    #nPFcandis = feature_PFcandi_array_xy.shape[1]
    #nPFcandifeatures = feature_PFcandi_array_xy.shape[2]
    ntargets = target_array_xy.shape[1]

    
    # Split datas into train, validation, test set

    # for test!!! (applying embedding)################

    # for PUPPI candidtaes
    
    #inputs = np.concatenate((feature_pupcandi_array[:,:,0:1], feature_pupcandi_array_xy[:,:,0:2]), axis=-1)
    
    inputs = feature_pupcandi_array_xy[:,:,:4]
    inputs_cat0 = feature_pupcandi_array_xy[:,:,4:5]
    inputs_cat1 = feature_pupcandi_array_xy[:,:,5:]
    
    # for PF candidtaes
    
    #inputs = np.concatenate((feature_PFcandi_array[:,:,0:1], feature_PFcandi_array_xy[:,:,0:2]), axis=-1)
    '''
    inputs = feature_PFcandi_array_xy[:,:,:4]
    inputs_cat0 = feature_PFcandi_array_xy[:,:,4:5]
    inputs_cat1 = feature_PFcandi_array_xy[:,:,5:]
    '''

    Xc = [inputs_cat0, inputs_cat1]

    X = [inputs]+[inputs_cat0]+[inputs_cat0]

    #embedding_input_dim = {i : int(np.max(Xc[i])) + 1 for i in range(2)}
    #embedding_input_dim = [int(np.max(Xc[i])) + 1 for i in range(2)]
    embedding_input_dim = [11, 11]

    #X = [feature_MET_array_xy, feature_pupcandi_array_xy, feature_pupcandi_array_xy[:,:,(0,1)]]
    y = target_array_xy
    A = feature_MET_array[:,(6,7)]
    
    fulllen = 170000
    #fulllen =nevents
    #fulllen =args.entry 
    tv_frac = 0.10
    tv_num = math.ceil(fulllen*tv_frac)
    #splits = np.cumsum([800000,90000,90000])
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
    keras_model = dense_embedding(n_features=inputs.shape[-1], n_features_cat=2, n_dense_layers=3, activation='tanh', embedding_input_dim = embedding_input_dim, number_of_pupcandis=100)


    # print variables
    print()
    print("# \t\tGen MET cut\t :\t %.1f" % TarMET_cut)
    print("# \t\tPUPPI MET cut\t :\t %.1f" % PupMET_cut)
    print("# \t\tNumber of event(total)\t : \t %d" % nevents)
    print("# \t\tNumber of event(used)\t : \t %d" % fulllen)
    print("# \t\tNumber of training event\t : \t %d" % A_train.shape[0])
    print("# \t\tNumber of test event\t : \t %d" % A_test.shape[0])
    print()

    

    #keras_model.compile(optimizer='adam', loss=custom_loss, metrics=['mean_absolute_error', 'mean_squared_error'])
    keras_model.compile(optimizer='adam', loss=custom_loss, metrics=['mean_squared_error', 'mean_squared_error'])
    print(keras_model.summary())

    lr_scale = 0.5
    batch_size=1024
    clr = CyclicLR(base_lr=0.0003*lr_scale, max_lr=0.001*lr_scale, step_size=len(y)/batch_size, mode='triangular2')

    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    model_checkpoint = ModelCheckpoint(''+path+'keras_model_best.h5', monitor='val_loss', save_best_only=True)
    csv_logger = CSVLogger('{}loss_data.log'.format(path))
    callbacks = [early_stopping, model_checkpoint, csv_logger, clr]

    
    # fit keras
     
    keras_model.fit(X_train, y_train, 
                    batch_size=batch_size, 
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
    Write_MET_binned_histogram(predict_phi, y_test_phi, 20, 0, 100, 300, name=''+path+'histogram_predicted_'+str(PupMET_cut)+'.root')
    Write_MET_binned_histogram(A_test, y_test_phi, 20, 0, 100, 300, name=''+path+'histogram_puppi_'+str(PupMET_cut)+'.root')

    # plot response
    response_ab(predict_phi, y_test_phi, 20, 0, 100, 300, path_ = path, name=''+path+'response_'+str(PupMET_cut)+'.png')
    response_parallel_opaque(predict_phi, A_test, y_test_phi, 20, 0, 100, 300, path_ = path, name=''+path+'response_'+str(PupMET_cut)+'')
    response_parallel(A_test, y_test_phi, 20, 0, 100, 300, path_ = path, name=''+path+'response_PUPPI'+str(PupMET_cut)+'')

    # For plots
    MET_rel_error_opaque(predict_phi[:,0], A_test[:,0], y_test_phi[:,0], name=''+path+'rel_error_opaque.png')
    #MET_rel_error(predict_phi[:,0], y_test_phi[:,0], name=''+path+'rel_error.png')
    #MET_abs_error(predict_phi[:,0], y_test_phi[:,0], name=''+path+'rel_abs.png')
    #Phi_abs_error(predict_phi[:,1], y_test_phi[:,1], name=''+path+'Phi_error.png')
    Phi_abs_error_opaque(predict_phi[:,1], y_test_phi[:,1], A_test[:,1], name=''+path+'Phi_error_opaque.png')
    MET_binned_predict_mean_opaque(predict_phi[:,0], A_test[:,0], y_test_phi[:,0], 20, 0, 150, 0, '.', name=''+path+'PrVSGen.png')
    #dist(predict_phi[:,0], name=''+path+'predict_dist.png')
    dist(y_test_phi[:,0], -150, 500, 100, name=''+path+'Gen_dist.png')
    histo_2D(predict_phi[:,0], y_test_phi[:,0], name=''+path+'2D_histo.png')

    #MET_rel_error(A_test[:,0], y_test_phi[:,0], name='rel_error_weight.png')
    #MET_binned_predict_mean(A_test[:,0], y_test_phi[:,0], 20, 0, 500, 0, '.', name='predict_mean.png')
    extract_result(predict_phi, y_test_phi, path, PupMET_cut, PupMET_cut_max)
    

if __name__ == "__main__":

    time_path = time.strftime('%Y-%m-%d', time.localtime(time.time()))
    path = "./result/"+time_path+"_PUPPICandidates/"

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', action='store', type=str, required=True, help='designate input file path')
    parser.add_argument('--output', action='store', type=str, default='{}'.format(path), help='designate output file path')
    #parser.add_argument('--entry', action='store', type=int, required=True, help='set number of events')
        
    args = parser.parse_args()
    main(args)
