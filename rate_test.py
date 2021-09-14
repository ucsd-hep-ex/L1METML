from sklearn.metrics import auc

import h5py
import os
import numpy as np
import numpy.ma
import time
import argparse
import collections as co
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
import tensorflow as tf

# Will test how much difference between PUPPI MET and Predicted MET in ID.

def main(args):
    
    print("\n*****************************************\n")

    # load Predicted MET (TTbar-1, SingleNeutrino-0) 
    Tarray_ML = np.load(os.path.join(args.input, "TTbar_feature_array_MLMET.npy"))
    Tarray_ML_target = np.load(os.path.join(args.input, "TTbar_target_array_MLMET.npy"))
    print("reading",args.input,"TTbar_feature_array_MLMET.npy")
    Sarray_ML = np.load(os.path.join(args.input, "SingleNeutrino_feature_array_MLMET.npy"))
    Sarray_ML_target = np.load(os.path.join(args.input, "SingleNeutrino_target_array_MLMET.npy"))
    print("reading",args.input,"SingleNeutrino_feature_array_MLMET.npy")
    Tarray_ML = np.concatenate((Tarray_ML[:,0:1], Tarray_ML_target[:,0:1], 1+np.zeros((Tarray_ML.shape[0],1))), axis=1)
    Sarray_ML = np.concatenate((Sarray_ML[:,0:1], Sarray_ML_target[:,0:1], np.zeros((Sarray_ML.shape[0],1))), axis=1)


    print("finish reading ML MET files")
    print("\n*****************************************\n")

    # load PUPPI MET (TTbar-1, SingleNeutrino-0)
    Tarray_PU = np.load(os.path.join(args.input, "TTbar_feature_array_PUMET.npy"))
    Tarray_PU_target = np.load(os.path.join(args.input, "TTbar_target_array_PUMET.npy"))
    print("reading",args.input,"TTbar_feature_array_PUMET.npy")
    Sarray_PU = np.load(os.path.join(args.input, "SingleNeutrino_feature_array_PUMET.npy"))
    Sarray_PU_target = np.load(os.path.join(args.input, "SingleNeutrino_target_array_PUMET.npy"))
    print("reading",args.input,"SingleNeutrino_feature_array_PUMET.npy")
    Tarray_PU = np.concatenate((Tarray_PU[:,0:1], Tarray_PU_target[:,0:1], 1+np.zeros((Tarray_PU.shape[0],1))), axis=1)
    Sarray_PU = np.concatenate((Sarray_PU[:,0:1], Sarray_PU_target[:,0:1], np.zeros((Sarray_PU.shape[0],1))), axis=1)


    print("finish reading PUPPI MET files")
    print("\n*****************************************\n")



    # concatenate TTbar and SingleNeutrino and shuffle
    ML1 = Tarray_ML
    ML0 = Sarray_ML

    PU1 = Tarray_PU
    PU0 = Sarray_PU

    bin_number = 300.
    step = 2.

    ML_array = np.zeros((int(bin_number),3))
    PU_array = np.zeros((int(bin_number),3))

    ML_rate = np.zeros(int(bin_number))
    PU_rate = np.zeros(int(bin_number))
    ML_rate_SN = np.zeros(int(bin_number))
    PU_rate_SN = np.zeros(int(bin_number))
    target_rate = np.zeros(int(bin_number))
    target_rate_SN = np.zeros(int(bin_number))


    All1_count = ML1.shape[0]
    All0_count = ML0.shape[0]

    for i in range(int(bin_number)):
        # ML
        
        ML1_count = np.sum(ML1[:,0] > i*step)
        ML0_count = np.sum(ML0[:,0] > i*step)
        Ta1_count = np.sum(PU1[:,1] > i*step)
        Ta0_count = np.sum(PU0[:,1] > i*step)

        TP = ML1_count
        FP = ML0_count
        FN = All1_count - ML1_count
        TN = All0_count - ML0_count


        # save plot data. -> TPR, FPR
        ML_array[i,0] = TP / (TP + FN + 1) # TPR
        ML_array[i,1] = FP / (FP + TN) # FPR
        ML_array[i,2] = step*i

        ML_rate[i] = ML1_count/All1_count
        ML_rate_SN[i] = ML0_count/All0_count
        target_rate[i] = Ta1_count/All1_count
        target_rate_SN[i] = Ta0_count/All0_count

        #PU

        PU1_count = np.sum(PU1[:,0] > i*step)
        PU0_count = np.sum(PU0[:,0] > i*step)

        TP = PU1_count
        FP = PU0_count
        FN = All1_count - PU1_count
        TN = All0_count - PU0_count

        # save plot data. -> TPR, FPR
        PU_array[i,0] = TP / (TP + FN + 1) # TPR
        PU_array[i,1] = FP / (FP + TN) # FPR
        PU_array[i,2] = step*i

        PU_rate[i] = PU1_count/All1_count
        PU_rate_SN[i] = PU0_count/All0_count


    which_plot = args.plot

    if which_plot == "ROC":
        ML_AUC = auc(ML_array[:,1], ML_array[:,0])
        PU_AUC = auc(PU_array[:,1], PU_array[:,0])

        print("ML AUC : {}".format(ML_AUC))
        print("PU AUC : {}".format(PU_AUC))
            
        plt.plot(ML_array[:,1], ML_array[:,0], label='ML ROC, AUC = {}'.format(round(ML_AUC,3)))
        plt.plot(PU_array[:,1], PU_array[:,0], '-r', label='PUPPI ROC, AUC = {}'.format(round(PU_AUC,3)))
        plt.grid(True, axis='x', color='gray', alpha=0.5, linestyle='--')
        plt.xlabel('FPR')
        plt.xlim(0.,1.)
        plt.grid(True, axis='y', color='gray', alpha=0.5, linestyle='--')
        plt.ylabel('TPR')
        plt.title('ROC')
        plt.legend()
        plt.savefig('ROC_curve.png')

    elif which_plot == "rate":
        x_ = range(0, int(step*bin_number), int(step))
        plt.plot(x_, ML_rate, 'bo', label='ML')
        plt.plot(x_, PU_rate, 'ro', label='PUPPI')
        plt.grid(True, axis='x', color='gray', alpha=0.5, linestyle='--')
        plt.grid(True, axis='y', color='gray', alpha=0.5, linestyle='--')
        plt.xlim(0,200)
        plt.legend()
        plt.xlabel('MET threshold (ML, PU MET) [GeV]')
        plt.ylabel('TTbar efficiency')
        plt.savefig('triggerrate_SN_nolog_200.png')
        plt.show()


    elif which_plot == "rate_com":
        x_ = range(0, int(step*bin_number), int(step))
        plt.plot(ML_rate, ML_rate_SN*31000, 'bo', label='ML')
        plt.plot(PU_rate, PU_rate_SN*31000, 'ro', label='PUPPI')
        plt.grid(True, axis='y', color='gray', alpha=0.5, linestyle='--')
        plt.grid(True, axis='x', color='gray', alpha=0.5, linestyle='--')
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
