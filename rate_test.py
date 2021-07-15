from sklearn.metrics import auc

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
    Tarray_ML = np.load("{}TTbar_feature_array_MLMET.npy".format(args.TT))
    print("reading", args.TT,"TTbar_feature_array_MLMET.npy")
    Sarray_ML = np.load("{}SingleNeutrino_feature_array_MLMET.npy".format(args.SN))
    print("reading", args.SN,"SingleNeutrino_feature_array_MLMET.npy")
    Tarray_ML = np.concatenate((Tarray_ML[:,0:1], 1+np.zeros((Tarray_ML.shape[0],1))), axis=1)
    Sarray_ML = np.concatenate((Sarray_ML[:,0:1], np.zeros((Sarray_ML.shape[0],1))), axis=1)


    print("finish reading ML MET files")
    print("\n*****************************************\n")

    # load PUPPI MET (TTbar-1, SingleNeutrino-0)
    Tarray_PU = np.load("{}TTbar_feature_array_PUMET.npy".format(args.TT))
    print("reading", args.TT,"TTbar_feature_array_PUMET.npy")
    Sarray_PU = np.load("{}SingleNeutrino_feature_array_PUMET.npy".format(args.SN))
    print("reading", args.SN,"SingleNeutrino_feature_array_PUMET.npy")
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
    ML_array = np.zeros((300,3))
    ML_Rate = np.zeros((300,2))
    PU_array = np.zeros((300,3))
    PU_Rate = np.zeros((300,2))

    for i in range(300):
        # ML
        mask_ML = array_ML[:,0] > 2*i
        ML_TTbar = array_ML[mask_ML]
        ML_SingleNeutrino = array_ML[~mask_ML]
        unique_TP, counts_TP = np.unique(ML_TTbar[:,1], return_counts=True)
        unique_FN, counts_FN = np.unique(ML_SingleNeutrino[:,1], return_counts=True)
        ML_TP_dict = dict(zip(unique_TP, counts_TP))
        ML_FN_dict = dict(zip(unique_FN, counts_FN))

        if 1.0 not in ML_TP_dict:
            ML_TP_dict[1.0] = 0
        if 0.0 not in ML_TP_dict:
            ML_TP_dict[0.0] = 0
        if 1.0 not in ML_FN_dict:
            ML_FN_dict[1.0] = 0
        if 0.0 not in ML_FN_dict:
            ML_FN_dict[0.0] = 0

        # save plot data. -> TPR, FPR, trigger rate...
        ML_array[i,0] = ML_TP_dict[1.0] / (ML_TP_dict[1.0] + ML_FN_dict[1.0])
        ML_array[i,1] = ML_TP_dict[0.0] / (ML_TP_dict[0.0] + ML_FN_dict[0.0])
        ML_array[i,2] = 2*i
        ML_Rate[i,0] = 2*i
        ML_Rate[i,1] = (ML_TP_dict[0.0] + ML_TP_dict[1.0])/(ML_TP_dict[0.0] + ML_TP_dict[1.0] + ML_FN_dict[0.0] + ML_FN_dict[1.0])


        # PU
        mask_PU = array_PU[:,0] > 2*i
        PU_TTbar = array_PU[mask_PU]
        PU_SingleNeutrino = array_PU[~mask_PU]
        unique_TP, counts_TP = np.unique(PU_TTbar[:,1], return_counts=True)
        unique_FN, counts_FN = np.unique(PU_SingleNeutrino[:,1], return_counts=True)
        PU_TP_dict = dict(zip(unique_TP, counts_TP))
        PU_FN_dict = dict(zip(unique_FN, counts_FN))

        if 1.0 not in PU_TP_dict:
            PU_TP_dict[1.0] = 0
        if 0.0 not in PU_TP_dict:
            PU_TP_dict[0.0] = 0
        if 1.0 not in PU_FN_dict:
            PU_FN_dict[1.0] = 0
        if 0.0 not in PU_FN_dict:
            PU_FN_dict[0.0] = 0

        # save plot data. -> TPR, FPR, trigger rate...
        PU_array[i,0] = PU_TP_dict[1.0] / (PU_TP_dict[1.0] + PU_FN_dict[1.0])
        PU_array[i,1] = PU_TP_dict[0.0] / (PU_TP_dict[0.0] + PU_FN_dict[0.0])
        PU_array[i,2] = 2*i
        PU_Rate[i,0] = 2*i
        PU_Rate[i,1] = (PU_TP_dict[0.0] + PU_TP_dict[1.0])/(PU_TP_dict[0.0] + PU_TP_dict[1.0] + PU_FN_dict[0.0] + PU_FN_dict[1.0])

    which_plot = args.plot

    if which_plot == "ROC":
        ML_AUC = auc(ML_array[:,1], ML_array[:,0])
        PU_AUC = auc(PU_array[:,1], PU_array[:,0])

        print("ML AUC : {}".format(ML_AUC))
        print("PU AUC : {}".format(PU_AUC))
            
        plt.plot(ML_array[:.1], ML_array[:,0], label='ML ROC, AUC = {}'.format(round(ML_AUC,3)))
        plt.plot(PU_array[:,1], PU_array[:,0], '-r', label='PUPPI ROC, AUC = {}'.format(round(PU_AUC,3)))
        plt.grid(True, axis='x', color='gray', alpha=0.5, linestyle='--')
        plt.xlabel('FPR')
        plt.xscale('log')
        plt.xlim(0.00001,0.1)
        plt.grid(True, axis='y', color='gray', alpha=0.5, linestyle='--')
        plt.ylabel('TPR')
        plt.title('ROC')
        plt.legend()
        #plt.show()
        plt.savefig('ROC_curve.png')


    elif which_plot == "trigger":
        plt.plot(ML_Rate[:,0], ML_Rate[:,1], label='ML Trigger Rate')
        plt.plot(ML_Rate[:,0], ML_array[:,0], label='ML Efficiency')
        plt.plot(PU_Rate[:,0], PU_Rate[:,1], label='PU Trigger Rate')
        plt.plot(PU_Rate[:,0], PU_array[:,0], label='PU Efficiency')
        plt.grid(True, axis='x', color='gray', alpha=0.5, linestyle='--')
        plt.xlabel('MET Threshold')
        plt.xlim(0.0, 300.0)
        plt.grid(True, axis='y', color='gray', alpha=0.5, linestyle='--')
        plt.ylabel('Trigger rate (event passed trigger/all event)')
        plt.title('Trigger rate')
        plt.legend()
        #plt.show()
        plt.savefig('trigger_rate.png')
    

# Configuration

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--TT', action='store', type=str, required=True, help='designate TTbar input file path')
    parser.add_argument('--SN', action='store', type=str, required=True, help='designate SingleNeutrino input file path')
    parser.add_argument('--plot', action='store', type=str, required=True, help='ROC for ROC curve, trigger for trigger rate ')
        
    args = parser.parse_args()
    main(args)
