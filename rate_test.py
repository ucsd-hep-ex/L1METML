import h5py
import os
import numpy as np
import argparse
import collections as co
import matplotlib.pyplot as plt

# local import
from utils import *

# Will test how much difference between PUPPI MET and Predicted MET in ID.

def main(args):
    
    print("\n*****************************************\n")

    # load Predicted MET (TTbar-1, SingleNeutrino-0) 
    Tarray_ML = np.load("{}TTbar_feature_array_MLMET.npy".format(args.input))
    print("reading", args.input, "TTbar_feature_array_MLMET.npy")
    Sarray_ML = np.load("{}SingleNeutrino_feature_array_MLMET.npy".format(args.input))
    print("reading", args.input, "SingleNeutrino_feature_array_MLMET.npy")
    Tarray_ML = np.concatenate((Tarray_ML[:,0:1], 1+np.zeros((Tarray_ML.shape[0],1))), axis=1)
    Sarray_ML = np.concatenate((Sarray_ML[:,0:1], np.zeros((Sarray_ML.shape[0],1))), axis=1)


    print("finish reading ML MET files")
    print("\n*****************************************\n")

    # load PUPPI MET (TTbar-1, SingleNeutrino-0)
    Tarray_PU = np.load("{}TTbar_feature_array_PUMET.npy".format(args.input))
    print("reading", args.input, "TTbar_feature_array_PUMET.npy")
    Sarray_PU = np.load("{}SingleNeutrino_feature_array_PUMET.npy".format(args.input))
    print("reading", args.input, "SingleNeutrino_feature_array_PUMET.npy")
    Tarray_PU = np.concatenate((Tarray_PU[:,0:1], 1+np.zeros((Tarray_PU.shape[0],1))), axis=1)
    Sarray_PU = np.concatenate((Sarray_PU[:,0:1], np.zeros((Sarray_PU.shape[0],1))), axis=1)


    print("finish reading PUPPI MET files")
    print("\n*****************************************\n")



    # concatenate TTbar and SingleNeutrino and shuffle
    array_ML = np.concatenate((Tarray_ML, Sarray_ML), axis=0)
    np.random.shuffle(array_ML)
    array_PU = np.concatenate((Tarray_PU, Sarray_PU), axis=0)
    np.random.shuffle(array_PU)

    # 
    ML_TPR = np.zeros(300)
    ML_FPR = np.zeros(300)
    PU_TPR = np.zeros(300)
    PU_FPR = np.zeros(300)
    for i in range(300):
        mask_ML = array_ML[:,0] > 2*i
        ML_TTbar = array_ML[mask_ML]
        ML_SingleNeutrino = array_ML[~mask_ML]
        unique_TP, counts_TP = np.unique(ML_TTbar[:,1], return_counts=True)
        unique_FN, counts_FN = np.unique(ML_SingleNeutrino[:,1], return_counts=True)
        TP_dict = dict(zip(unique_TP, counts_TP))
        FN_dict = dict(zip(unique_FN, counts_FN))

        if 1.0 not in TP_dict:
            TP_dict[1.0] = 0
        if 0.0 not in TP_dict:
            TP_dict[0.0] = 0
        if 1.0 not in FN_dict:
            FN_dict[1.0] = 0
        if 0.0 not in FN_dict:
            FN_dict[0.0] = 0

        ML_TPR[i] = TP_dict[1.0] / (TP_dict[1.0] + FN_dict[1.0])
        ML_FPR[i] = TP_dict[0.0] / (TP_dict[0.0] + FN_dict[0.0])


        mask_PU = array_PU[:,0] > 2*i
        PU_TTbar = array_PU[mask_PU]
        PU_SingleNeutrino = array_PU[~mask_PU]
        unique_TP, counts_TP = np.unique(PU_TTbar[:,1], return_counts=True)
        unique_FN, counts_FN = np.unique(PU_SingleNeutrino[:,1], return_counts=True)
        TP_dict = dict(zip(unique_TP, counts_TP))
        FN_dict = dict(zip(unique_FN, counts_FN))

        if 1.0 not in TP_dict:
            TP_dict[1.0] = 0
        if 0.0 not in TP_dict:
            TP_dict[0.0] = 0
        if 1.0 not in FN_dict:
            FN_dict[1.0] = 0
        if 0.0 not in FN_dict:
            FN_dict[0.0] = 0

        PU_TPR[i] = TP_dict[1.0] / (TP_dict[1.0] + FN_dict[1.0])
        PU_FPR[i] = TP_dict[0.0] / (TP_dict[0.0] + FN_dict[0.0])
        
    plt.plot(ML_FPR, ML_TPR)
    plt.plot(PU_FPR, PU_TPR, '-r')
    plt.show()

    

# Configuration

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', action='store', type=str, required=True, help='designate ML input file path')
        
    args = parser.parse_args()
    main(args)
