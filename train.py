import numpy as np
import tables
import argparse
import math
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
    print("Working")

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

    mask = (feature_MET_array[:,6] > PupMET_cut) & (feature_MET_array[:,0] < PupMET_cut_max)
    feature_MET_array = feature_MET_array[mask]
    feature_jet_array = feature_jet_array[mask]
    feature_pupcandi_array = feature_pupcandi_array[mask]
    target_array = target_array[mask]

    nevents = target_array.shape[0]

    # Exclude Gen met < +TarMET_cut+ GeV events
    # Set Gen MET min, max cut

    TarMET_cut = 0.5
    TarMET_cut_max = 500

    mask1 = (target_array[:,0] > TarMET_cut) & (target_array[:,0] < TarMET_cut_max)
    feature_MET_array = feature_MET_array[mask1]
    feature_jet_array = feature_jet_array[mask1]
    feature_pupcandi_array = feature_pupcandi_array[mask1]
    target_array = target_array[mask1]

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
    A=feature_pupcandi_array_xy[:,:,4]
    A=np.where(A==-1,3,A)
    A=np.where(A==0,4,A)
    A=np.where(A==1,5,A)

    B=feature_pupcandi_array_xy[:,:,5]
    B=np.where(B==-211,0,B)
    B=np.where(B==-22,1,B)
    B=np.where(B==-13,2,B)
    B=np.where(B==-11,3,B)
    B=np.where(B==11,4,B)
    B=np.where(B==13,5,B)
    B=np.where(B==22,6,B)
    B=np.where(B==130,7,B)
    B=np.where(B==211,8,B)

    print(A)
    print(B)
	
    
    # Convert target from pt phi to px, py
    target_array_xy = np.zeros((nevents, ntargets))

    target_array_xy[:,0] = target_array[:,0] * np.cos(target_array[:,1])
    target_array_xy[:,1] = target_array[:,0] * np.sin(target_array[:,1])
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
        
    args = parser.parse_args()
    main(args)
