from sklearn.metrics import auc
#import ROOT

import h5py
import os
import numpy as np
import argparse
import collections as co
import matplotlib.pyplot as plt

# local import
#from utils import *

# Will test how much difference between PUPPI MET and Predicted MET in ID.

def main(args):
    
    print("\n*****************************************\n")

    # load Predicted MET (TTbar-1, SingleNeutrino-0) 
    Tarray_ML = np.load("{}TTbar_feature_array_MLMET.npy".format(args.input))
    Tarray_ML_target = np.load("{}TTbar_target_array_MLMET.npy".format(args.input))
    print("reading", args.TT,"TTbar_feature_array_MLMET.npy")
    Sarray_ML = np.load("{}SingleNeutrino_feature_array_MLMET.npy".format(args.input))
    Sarray_ML_target = np.load("{}SingleNeutrino_target_array_MLMET.npy".format(args.input))
    print("reading", args.SN,"SingleNeutrino_feature_array_MLMET.npy")
    Tarray_ML = np.concatenate((Tarray_ML[:,0:1], Tarray_ML_target[:,0:1], 1+np.zeros((Tarray_ML.shape[0],1))), axis=1)
    Sarray_ML = np.concatenate((Sarray_ML[:,0:1], Sarray_ML_target[:,0:1], np.zeros((Sarray_ML.shape[0],1))), axis=1)


    print("finish reading ML MET files")
    print("\n*****************************************\n")

    # load PUPPI MET (TTbar-1, SingleNeutrino-0)
    Tarray_PU = np.load("{}TTbar_feature_array_PUMET.npy".format(args.input))
    Tarray_PU_target = np.load("{}TTbar_target_array_PUMET.npy".format(args.input))
    print("reading", args.TT,"TTbar_feature_array_PUMET.npy")
    Sarray_PU = np.load("{}SingleNeutrino_feature_array_PUMET.npy".format(args.input))
    Sarray_PU_target = np.load("{}SingleNeutrino_target_array_PUMET.npy".format(args.input))
    print("reading", args.SN,"SingleNeutrino_feature_array_PUMET.npy")
    Tarray_PU = np.concatenate((Tarray_PU[:,0:1], Tarray_PU_target[:,0:1], 1+np.zeros((Tarray_PU.shape[0],1))), axis=1)
    Sarray_PU = np.concatenate((Sarray_PU[:,0:1], Sarray_PU_target[:,0:1], np.zeros((Sarray_PU.shape[0],1))), axis=1)


    print("finish reading PUPPI MET files")
    print("\n*****************************************\n")



    # concatenate TTbar and SingleNeutrino and shuffle
    array_ML = Tarray_ML
    array_ML_SN = Sarray_ML
    array_ML_Tot = np.concatenate((Tarray_ML, Sarray_ML), axis=0)
    np.random.shuffle(array_ML_Tot)

    array_PU = Tarray_PU
    array_PU_SN = Sarray_PU
    array_PU_Tot = np.concatenate((Tarray_PU, Sarray_PU), axis=0)
    np.random.shuffle(array_PU_Tot)

    # 
    ML_array = np.zeros((300,3))
    ML_Rate = np.zeros((300,2))
    PU_array = np.zeros((300,3))
    PU_Rate = np.zeros((300 ,2))

    bin_number = 300.
    step = 2.
    ML = {}
    PU = {}

    ML_rate = np.zeros(int(bin_number))
    PU_rate = np.zeros(int(bin_number))
    ML_rate_SN = np.zeros(int(bin_number))
    PU_rate_SN = np.zeros(int(bin_number))
    ML_rate_target = np.zeros(int(bin_number))
    PU_rate_target = np.zeros(int(bin_number))
    ML_rate_SN_target = np.zeros(int(bin_number))
    PU_rate_SN_target = np.zeros(int(bin_number))

    for i in range(int(bin_number)):
        # ML
        mask_ML = array_ML[:,0] > step*i
        mask_ML_SN = array_ML_SN[:,0] > step*i
        mask_ML_target = array_ML[:,1] > step*i
        mask_ML_SN_target = array_ML_SN[:,1] > step*i

        mask_ML_Tot = array_ML_Tot[:,0] > step*i

        ML_TTbar = array_ML[mask_ML]
        ML_SN = array_ML_SN[mask_ML_SN]
        ML_TTbar_target = array_ML[mask_ML_target]
        ML_SN_target = array_ML_SN[mask_ML_SN_target]

        ML_Tot = array_ML_Tot[mask]
        ML_Tot_CutOff = array_ML_Tot[~mask]

        unique_TP, counts_TP = np.unique(ML_Tot[:,2], return_counts=True)
        unique_FN, counts_FN = np.unique(ML_Tot_CutOff[:,2], return_counts=True)
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

        # save plot data. -> TPR, FPR
        ML_array[i,0] = ML_TP_dict[1.0] / (ML_TP_dict[1.0] + ML_FN_dict[1.0]+1) # TPR
        ML_array[i,1] = ML_TP_dict[0.0] / (ML_TP_dict[0.0] + ML_FN_dict[0.0]+1) # FPR
        ML_array[i,2] = step*i

        ML_rate[i] = ML_TTbar.shape[0]/array_ML.shape[0]
        ML_rate_SN[i] = ML_SN.shape[0]/array_ML_SN.shape[0]
        ML_rate_target[i] = ML_TTbar_target.shape[0]/array_ML.shape[0]
        ML_rate_SN_target[i] = ML_SN_target.shape[0]/array_ML_SN.shape[0]

        if (step*i) % 100. == 0.:
            ML['{}-T'.format(step*i)] = ML_Tot[:,1]
            ML['{}-T_PR'.format(step*i)] = ML_Tot[:,0]
            ML['{}-A'.format(step*i)] = array_ML_Tot[:,1]


        # PU
        mask_PU = array_PU[:,0] > step*i
        mask_PU_SN = array_PU_SN[:,0] > step*i
        mask_PU_target = array_PU[:,1] > step*i
        mask_PU_SN_target = array_PU_SN[:,1] > step*i

        mask_PU_Tot = array_PU_Tot[:,0] > step*i

        PU_TTbar = array_PU[mask_PU]
        PU_TTbar_SN = array_PU_SN[mask_PU_SN]
        PU_TTbar_target = array_PU[mask_PU_target]
        PU_TTbar_SN_target = array_PU_SN[mask_PU_SN_target]

        PU_Tot = array_PU_Tot[mask_PU_Tot]
        PU_Tot_CutOff = array_PU_Tot[~mask_PU_Tot]

        unique_TP, counts_TP = np.unique(PU_Tot[:,2], return_counts=True)
        unique_FN, counts_FN = np.unique(PU_Tot_CutOff[:,2], return_counts=True)
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

        # save plot data. -> TPR, FPR
        PU_array[i,0] = PU_TP_dict[1.0] / (PU_TP_dict[1.0] + PU_FN_dict[1.0]+1) # TPR
        PU_array[i,1] = PU_TP_dict[0.0] / (PU_TP_dict[0.0] + PU_FN_dict[0.0]+1) # FPR
        PU_array[i,2] = step*i

        PU_rate[i] = PU_TTbar.shape[0]/array_PU.shape[0]
        PU_rate_SN[i] = PU_TTbar_SN.shape[0]/array_PU_SN.shape[0]
        PU_rate_target[i] = PU_TTbar_target.shape[0]/array_PU.shape[0]
        PU_rate_SN_target[i] = PU_TTbar_SN_target.shape[0]/array_PU_SN.shape[0]

        if (step*i) % 100. == 0.:
            PU['{}-T'.format(step*i)] = PU_Tot[:,1]
            PU['{}-T_PR'.format(step*i)] = PU_Tot[:,0]
            PU['{}-A'.format(step*i)] = array_PU_Tot[:,1]
           

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

    elif which_plot == "turnon":
        range_ = 300
        bin_number = 100
        step = range_/bin_number
        
        for j in range(3):
            ML_All_array = ML['{}-A'.format(100.*j)]
            ML_Pass_array = ML['{}-T'.format(100.*j)]
            ML_Pass_PR_array = ML['{}-T_PR'.format(100.*j)]
            PU_All_array = PU['{}-A'.format(100.*j)]
            PU_Pass_array = PU['{}-T'.format(100.*j)]
            PU_Pass_PR_array = PU['{}-T_PR'.format(100.*j)]

            rat_ML=np.zeros((bin_number,int(step)))
            for i in range(bin_number):
                all_cond = ML_All_array[(ML_All_array < (step*(i+1))) & (ML_All_array >= (step*i))]

                TT_cond = ML_Pass_array[(ML_Pass_array < (step*(i+1))) & (ML_Pass_array >= (step*i))]
                TT_PR_cond = ML_Pass_PR_array[(ML_Pass_array < (step*(i+1))) & (ML_Pass_array >= (step*i))]

                if i == 33:
                    plt.hist(TT_PR_cond, bins=np.linspace(0., 300., 100), alpha=0.5, label='ML_preX', color='blue')
                    plt.hist(TT_cond, bins=np.linspace(0., 300., 100), alpha=0.5, label='genX', color='green')
                    #plt.show()

                if all_cond.size ==0:
                    div = 1
                else:
                    div = all_cond.size
                rat_ML[i,1] = (TT_cond.size/div)
                rat_ML[i,0] = step*i + 1.

            rat_PU=np.zeros((bin_number, int(step)))
            for i in range(bin_number):
                all_cond = PU_All_array[(PU_All_array < (step*(i+1))) & (PU_All_array>= (step*i))]

                TT_cond = PU_Pass_array[(PU_Pass_array < (step*(i+1))) & (PU_Pass_array >= (step*i))]
                TT_PR_cond_PU = PU_Pass_PR_array[(PU_Pass_array < (step*(i+1))) & (PU_Pass_array >= (step*i))]

                if i == 33:
                    plt.hist(TT_PR_cond_PU, bins=np.linspace(0., 300., 100), alpha=0.5, label='PU_preX', color='red')
                    #plt.hist(TT_cond, bins=np.linspace(0., 300., 100), alpha=0.5, label='genX_PU', color='black')
                    plt.legend()
                    plt.savefig('{}_dist.png'.format(i*step))
                    plt.show()

                if all_cond.size ==0:
                    div = 1
                else:
                    div = all_cond.size
                rat_PU[i,1] = (TT_cond.size/div)
                rat_PU[i,0] = step*i + 1.



            plt.figsize=(14,14)
            plt.plot(rat_ML[:,0], rat_ML[:,1], 'bo', label='ML')
            plt.plot(rat_PU[:,0], rat_PU[:,1], 'ro', label='PU')
            plt.legend()
            plt.xlabel('GenMET [GeV]')
            plt.ylabel('Efficiency')
            plt.savefig('{}_turnon.png'.format(j*100))
            plt.show()

    elif which_plot == "rate":
        x_ = range(0, int(step*bin_number), int(step))
        plt.plot(x_, ML_rate, 'bo', label='ML')
        plt.grid(True, axis='x', color='gray', alpha=0.5, linestyle='--')
        plt.plot(x_, PU_rate, 'ro', label='PUPPI')
        plt.grid(True, axis='y', color='gray', alpha=0.5, linestyle='--')
        plt.plot(x_, ML_rate_target, 'go', label='True')
        plt.xlim(0,200)
        #plt.yscale("log")
        plt.legend()
        plt.xlabel('MET threshold (ML, PU MET) [GeV]')
        plt.ylabel('TTbar efficiency')
        plt.savefig('triggerrate_SN_nolog_200.png')
        plt.show()


    elif which_plot == "rate_com":
        x_ = range(0, int(step*bin_number), int(step))
        plt.plot(ML_rate, ML_rate_SN*31000, 'bo', label='ML')
        plt.plot(PU_rate, PU_rate_SN*31000, 'ro', label='PUPPI')
        plt.plot(ML_rate_target, ML_rate_SN_target*31000, 'go', label='True')
        plt.grid(True, axis='y', color='gray', alpha=0.5, linestyle='--')
        plt.grid(True, axis='x', color='gray', alpha=0.5, linestyle='--')
        #plt.xlim(0,200)
        plt.yscale("log")
        plt.legend()
        plt.xlabel('TTbar efficiency')
        plt.ylabel('SingleNeutrino rate [kHz]')
        plt.savefig('combined_True.png')
        plt.show()

    

# Configuration

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', action='store', type=str, required=True, help='designate input file path (= output path of training)')
    parser.add_argument('--plot', action='store', type=str, required=True, help='ROC for ROC curve, trigger for trigger rate ')
        
    args = parser.parse_args()
    main(args)
