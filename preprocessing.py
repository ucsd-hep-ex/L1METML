import numpy as np
import tables
import argparse
import math
import setGPU
import os
from get_jet import get_features_targets, get_jet_features
from utils import custom_loss, flatting

def main(args):


    file_path = 'data/input_MET_PFCandi.h5'
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

    features_PFcandi = ['L1PFCands_pt','L1PFCands_phi','L1PFCands_eta', 'L1PFCands_puppiWeight',
                         'L1PFCands_charge','L1PFCands_pdgId']

    targets = ['genMet_pt', 'genMet_phi']
    targets_jet = ['GenJets_pt', 'GenJets_phi', 'GenJets_eta']

    # Set number of jets and pupcandis you will use
    number_of_jets = 20
    number_of_pupcandis = 100
    number_of_PFcandis = 700

    feature_MET_array, target_array = get_features_targets(file_path, features, targets)
    feature_jet_array = get_jet_features(file_path, features_jet, number_of_jets)
    feature_pupcandi_array = get_jet_features(file_path, features_pupcandi, number_of_pupcandis)
    feature_PFcandi_array = get_jet_features(file_path, features_PFcandi, number_of_PFcandis)
    nMETs = int(feature_MET_array.shape[1]/2)

    nevents = target_array.shape[0]
    nmetfeatures = feature_MET_array.shape[1]
    njets = feature_jet_array.shape[1]
    njetfeatures = feature_jet_array.shape[2]
    npupcandis = feature_pupcandi_array.shape[1]
    npupcandifeatures = feature_pupcandi_array.shape[2]
    nPFcandis = feature_PFcandi_array.shape[1]
    nPFcandifeatures = feature_PFcandi_array.shape[2]
    ntargets = target_array.shape[1]

    print("Getting file complite.")

    # Exclude puppi met < +PupMET_cut+ GeV events
    # Set PUPPI MET min, max cut

    PupMET_cut = 50
    PupMET_cut_max = 500
    weights_path = ''+str(PupMET_cut)+'cut'

    mask = (feature_MET_array[:,6] > PupMET_cut) & (feature_MET_array[:,0] < PupMET_cut_max)
    print("mask : {}".format(mask))
    feature_MET_array = feature_MET_array[mask]
    feature_jet_array = feature_jet_array[mask]
    feature_pupcandi_array = feature_pupcandi_array[mask]
    feature_PFcandi_array = feature_PFcandi_array[mask]
    target_array = target_array[mask]

    nevents = target_array.shape[0]

    # Exclude Gen met < +TarMET_cut+ GeV events
    # Set Gen MET min, max cut

    TarMET_cut = 5
    TarMET_cut_max = 500

    mask1 = (target_array[:,0] > TarMET_cut) & (target_array[:,0] < TarMET_cut_max)
    feature_MET_array = feature_MET_array[mask1]
    feature_jet_array = feature_jet_array[mask1]
    feature_PFcandi_array = feature_PFcandi_array[mask1]
    feature_pupcandi_array = feature_pupcandi_array[mask1]
    target_array = target_array[mask1]

    nevents = target_array.shape[0]
    
    print("Excluding complite.")

    # Flatting the sample

    mask2 = flatting(target_array)
    feature_MET_array = feature_MET_array[mask2]
    feature_jet_array = feature_jet_array[mask2]
    feature_pupcandi_array = feature_pupcandi_array[mask2]
    feature_PFcandi_array = feature_PFcandi_array[mask2]
    target_array = target_array[mask2]

    nevents = target_array.shape[0]

    # Shuffle again
    shuffler = np.random.permutation(len(target_array))
    print(shuffler)
    feature_MET_array = feature_MET_array[shuffler]
    feature_jet_array = feature_jet_array[shuffler,:,:]
    feature_pupcandi_array = feature_pupcandi_array[shuffler,:,:]
    feature_PFcandi_array = feature_PFcandi_array[shuffler,:,:]
    target_array = target_array[shuffler]

    print("Flatting complite")
    

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
    
    feature_PFcandi_array_xy = np.zeros((nevents, nPFcandis, nPFcandifeatures))
    for i in range(number_of_PFcandis):
        feature_PFcandi_array_xy[:,i,0] = feature_PFcandi_array[:,i,0] * np.cos(feature_PFcandi_array[:,i,1])
        feature_PFcandi_array_xy[:,i,1] = feature_PFcandi_array[:,i,0] * np.sin(feature_PFcandi_array[:,i,1])
        feature_PFcandi_array_xy[:,i,2] = feature_PFcandi_array[:,i,2]
        feature_PFcandi_array_xy[:,i,3] = feature_PFcandi_array[:,i,3]
        feature_PFcandi_array_xy[:,i,4] = feature_PFcandi_array[:,i,4]
        feature_PFcandi_array_xy[:,i,5] = feature_PFcandi_array[:,i,5]
    
   
    print("Converting complite")

    #labeling

    feature_pupcandi_array_xy[:,:,4]=np.where(feature_pupcandi_array_xy[:,:,4]==-1,3,feature_pupcandi_array_xy[:,:,4])
    feature_pupcandi_array_xy[:,:,4]=np.where(feature_pupcandi_array_xy[:,:,4]==0,4,feature_pupcandi_array_xy[:,:,4])
    feature_pupcandi_array_xy[:,:,4]=np.where(feature_pupcandi_array_xy[:,:,4]==1,5,feature_pupcandi_array_xy[:,:,4])

    feature_pupcandi_array_xy[:,:,5]=np.where(feature_pupcandi_array_xy[:,:,5]==-211,0,feature_pupcandi_array_xy[:,:,5])
    feature_pupcandi_array_xy[:,:,5]=np.where(feature_pupcandi_array_xy[:,:,5]==-130,1,feature_pupcandi_array_xy[:,:,5])
    feature_pupcandi_array_xy[:,:,5]=np.where(feature_pupcandi_array_xy[:,:,5]==-22,2,feature_pupcandi_array_xy[:,:,5])
    feature_pupcandi_array_xy[:,:,5]=np.where(feature_pupcandi_array_xy[:,:,5]==-13,3,feature_pupcandi_array_xy[:,:,5])
    feature_pupcandi_array_xy[:,:,5]=np.where(feature_pupcandi_array_xy[:,:,5]==-11,4,feature_pupcandi_array_xy[:,:,5])
    feature_pupcandi_array_xy[:,:,5]=np.where(feature_pupcandi_array_xy[:,:,5]==11,5,feature_pupcandi_array_xy[:,:,5])
    feature_pupcandi_array_xy[:,:,5]=np.where(feature_pupcandi_array_xy[:,:,5]==13,6,feature_pupcandi_array_xy[:,:,5])
    feature_pupcandi_array_xy[:,:,5]=np.where(feature_pupcandi_array_xy[:,:,5]==22,7,feature_pupcandi_array_xy[:,:,5])
    feature_pupcandi_array_xy[:,:,5]=np.where(feature_pupcandi_array_xy[:,:,5]==130,8,feature_pupcandi_array_xy[:,:,5])
    feature_pupcandi_array_xy[:,:,5]=np.where(feature_pupcandi_array_xy[:,:,5]==211,9,feature_pupcandi_array_xy[:,:,5])


    feature_PFcandi_array_xy[:,:,4]=np.where(feature_PFcandi_array_xy[:,:,4]==-1,3,feature_PFcandi_array_xy[:,:,4])
    feature_PFcandi_array_xy[:,:,4]=np.where(feature_PFcandi_array_xy[:,:,4]==0,4,feature_PFcandi_array_xy[:,:,4])
    feature_PFcandi_array_xy[:,:,4]=np.where(feature_PFcandi_array_xy[:,:,4]==1,5,feature_PFcandi_array_xy[:,:,4])

    feature_PFcandi_array_xy[:,:,5]=np.where(feature_PFcandi_array_xy[:,:,5]==-211,0,feature_PFcandi_array_xy[:,:,5])
    feature_PFcandi_array_xy[:,:,5]=np.where(feature_PFcandi_array_xy[:,:,5]==-130,1,feature_PFcandi_array_xy[:,:,5])
    feature_PFcandi_array_xy[:,:,5]=np.where(feature_PFcandi_array_xy[:,:,5]==-22,2,feature_PFcandi_array_xy[:,:,5])
    feature_PFcandi_array_xy[:,:,5]=np.where(feature_PFcandi_array_xy[:,:,5]==-13,3,feature_PFcandi_array_xy[:,:,5])
    feature_PFcandi_array_xy[:,:,5]=np.where(feature_PFcandi_array_xy[:,:,5]==-11,4,feature_PFcandi_array_xy[:,:,5])
    feature_PFcandi_array_xy[:,:,5]=np.where(feature_PFcandi_array_xy[:,:,5]==11,5,feature_PFcandi_array_xy[:,:,5])
    feature_PFcandi_array_xy[:,:,5]=np.where(feature_PFcandi_array_xy[:,:,5]==13,6,feature_PFcandi_array_xy[:,:,5])
    feature_PFcandi_array_xy[:,:,5]=np.where(feature_PFcandi_array_xy[:,:,5]==22,7,feature_PFcandi_array_xy[:,:,5])
    feature_PFcandi_array_xy[:,:,5]=np.where(feature_PFcandi_array_xy[:,:,5]==130,8,feature_PFcandi_array_xy[:,:,5])
    feature_PFcandi_array_xy[:,:,5]=np.where(feature_PFcandi_array_xy[:,:,5]==211,9,feature_PFcandi_array_xy[:,:,5])




    for i in range(feature_pupcandi_array_xy.shape[0]):
        for j in range(feature_pupcandi_array_xy.shape[1]):
            if feature_pupcandi_array_xy[i,j,4] == 3:
                continue
            elif feature_pupcandi_array_xy[i,j,4] == 4:
                continue
            elif feature_pupcandi_array_xy[i,j,4] == 5:
                continue
            print("{}th event {}th candi : {}".format(i,j,feature_pupcandi_array_xy[i,j,4]))
    print("Labeling complite")
    
    
    # Convert target from pt phi to px, py
    target_array_xy = np.zeros((nevents, ntargets))

    target_array_xy[:,0] = target_array[:,0] * np.cos(target_array[:,1])
    target_array_xy[:,1] = target_array[:,0] * np.sin(target_array[:,1])



    # Save datas into files

    path = "./preprocessed/"


    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print ('Creating directory' + path)


    np.save(""+path+"feat_MET_array_xy_{}-{}".format(PupMET_cut, PupMET_cut_max), feature_MET_array_xy)
    np.save(""+path+"feat_MET_array_{}-{}".format(PupMET_cut, PupMET_cut_max), feature_MET_array)
    np.save(""+path+"feat_jet_array_xy_{}-{}".format(PupMET_cut, PupMET_cut_max), feature_jet_array_xy)
    np.save(""+path+"feat_Pup_array_xy_{}-{}".format(PupMET_cut, PupMET_cut_max), feature_pupcandi_array_xy)
    np.save(""+path+"feat_PF_array_xy_{}-{}".format(PupMET_cut, PupMET_cut_max), feature_PFcandi_array_xy)
    np.save(""+path+"targ_MET_array_xy_{}-{}".format(PupMET_cut, PupMET_cut_max), target_array_xy)




    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
        
    args = parser.parse_args()
    main(args)
