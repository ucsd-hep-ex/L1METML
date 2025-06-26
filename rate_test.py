"""
Rate test for ML MET vs PUPPI MET

Compares performance of ML MET and PUPPI MET by analyzing 
ROC curve and trigger rates for various datasets.

"""

from sklearn.metrics import auc, roc_curve, roc_auc_score

import h5py
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import numpy.ma
import time
import argparse
import collections as co
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
import tensorflow as tf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
@dataclass
class AnalysisConfig:
    """Configuration for rate analysis."""
    bin_number: int = 300
    step: float = 2.0
    trigger_rate_khz: float = 31000.0
    max_threshold_gev: int = 200
    output_dir: str = "plots"

@dataclass
class DataArrays:
    '''Holds numpy arrays for ML and PUPPI MET data.'''
    #TODO: change to use signal/bkg instead of ttbar/sn
    # and handle more samples
    ttbar_ml: np.ndarray
    single_neutrino_ml: np.ndarray
    ttbar_puppi: np.ndarray
    single_neutrino_puppi: np.ndarray
class DataLoader:
    """Handles loading and preprocessing of numpy arrays."""
    """Different from training data-laoder"""
    #TODO: change to use signal/bkg instead of ttbar/sn
    # and handle more samples
    
    def __init__(self, input_path: str):
        self.input_path = Path(input_path)

    def load_arrays(self) -> DataArrays:
        """Load and preprocess all required arrays."""
        logger.info(f"Loading data from {self.input_path}")

        #Load ML arrays
        ttbar_ml = self._load_and_process_arrays("TTbar", "MLMET")
        sn_ml = self._load_and_process_arrays("SingleNeutrino", "MLMET", is_signal=False)

        #Load PUPPI arrays
        ttbar_puppi = self._load_and_process_arrays("TTbar", "PUMET")
        sn_puppi = self._load_and_process_arrays("SingleNeutrino", "PUMET", is_signal=False)

        return DataArrays(ttbar_ml, sn_ml, ttbar_puppi, sn_puppi)

    def _load_and_process_arrays(self, sample: str, met_type: str, is_signal: bool = True) -> np.ndarray:
        """Load and process arrays for a specific sample and MET type."""
        #TODO: consistently change naming to prediction instead of feature
        feature_file = f"{sample}_feature_array_{met_type}.npy" 
        target_file = f"{sample}_target_array_{met_type}.npy"

        feature_array = np.load(self.input_path / feature_file)  
        target_array = np.load(self.input_path / target_file)

        logger.info(f"Loaded {feature_file} and {target_file}")

        # Create label column (1 for signal, 0 fopr backgroudn)
        label = np.ones((feature_array.shape[0],1)) if is_signal else np.zeros((feature_array.shape[0],1)) 

        return np.concatenate([feature_array[:, 0:1], target_array[:, 0:1], label], axis=1)

class RateAnalyzer:
    """Performs the rate and classification analysis for ML and PUPPI MET."""

    def __init__(self, config: AnalysisConfig):
        self.config = config

    def calculate_rates(self, data: DataArrays) -> Dict[str, np.ndarray]:
        """Calculate trigger rates for both ML and PUPPI methods."""
        bin_count = int(self.config.bin_number)

        # Initialize arrays to hold rates and thresholds
        #TODO: adjust for sig/bkg instead of ttbar/sn
        rates = {
            'ml_ttbar': np.zeros(bin_count),
            'ml_sn': np.zeros(bin_count),
            'puppi_ttbar': np.zeros(bin_count),
            'puppi_sn': np.zeros(bin_count),
            'ml_roc': np.zeros((bin_count, 3)), # TPR, FPR, threshold
            'puppi_roc': np.zeros((bin_count, 3)) # TPR, FPR, threshold
        }

        ttbar_total = data.ttbar_ml.shape[0]
        sn_total = data.single_neutrino_ml.shape[0]

        for i in range(bin_count):
            threshold = i * self.config.step

            # ML rates
            ml_ttbar_pass = np.sum(data.ttbar_ml[:,0] > threshold)
            ml_sn_pass = np.sum(data.single_neutrino_ml[:,0] > threshold)

            rates['ml_ttbar'][i] = ml_ttbar_pass / ttbar_total
            rates['ml_sn'][i] = ml_sn_pass / sn_total

            # ML ROC values
            rates['ml_roc'][i] = self._calculate_roc_point(
                ml_ttbar_pass,
                ml_sn_pass,
                ttbar_total,
                sn_total,
                threshold
            )

            # PUPPI rates
            puppi_ttbar_pass = np.sum(data.ttbar_puppi[:,0] > threshold)
            puppi_sn_pass = np.sum(data.single_neutrino_puppi[:,0] > threshold)

            rates['puppi_ttbar'][i] = puppi_ttbar_pass / ttbar_total
            rates['puppi_sn'][i] = puppi_sn_pass / sn_total

            # PUPPI ROC values
            rates['puppi_roc'][i] = self._calculate_roc_point(
                puppi_ttbar_pass,
                puppi_sn_pass,
                ttbar_total,
                sn_total,
                threshold
            )

            return rates
        
    def _calculate_roc_point(self, tp: int, fp: int, total_pos: int, total_neg: int, threshold: float) -> np.ndarray:
        """Calculate TPR, FPR for a single threshold."""
        fn = total_pos - tp
        tn = total_neg - fp

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        return np.array([tpr, fpr, threshold])
    
    def calculate_auc(self, data: DataArrays) -> Tuple[float, float]:
        """Calcualte AUC using sklearn."""
        # Prepare data for sklearn
        y_true = np.concatenate([
            np.ones(data.ttbar_ml.shape[0]),
            np.zeros(data.single_neutrino_ml.shape[0])
        ])

        y_score_ml = np.concatenate([data.ttbar_ml[:, 0], data.single_neutrino_ml[:, 0]])
        y_score_puppi = np.concatenate([data.ttbar_puppi[:, 0], data.single_neutrino_puppi[:, 0]])

        ml_auc = roc_auc_score(y_true, y_score_ml)
        puppi_auc = roc_auc_score(y_true, y_score_puppi)

        return ml_auc, puppi_auc
    

def main(args):

    print("\n*****************************************\n")

    # load Predicted MET (TTbar-1, SingleNeutrino-0)
    Tarray_ML = np.load(os.path.join(args.input, "TTbar_feature_array_MLMET.npy"))
    Tarray_ML_target = np.load(os.path.join(args.input, "TTbar_target_array_MLMET.npy"))
    print("reading", args.input, "TTbar_feature_array_MLMET.npy")
    Sarray_ML = np.load(os.path.join(args.input, "SingleNeutrino_feature_array_MLMET.npy"))
    Sarray_ML_target = np.load(os.path.join(args.input, "SingleNeutrino_target_array_MLMET.npy"))
    print("reading", args.input, "SingleNeutrino_feature_array_MLMET.npy")
    Tarray_ML = np.concatenate((Tarray_ML[:, 0:1], Tarray_ML_target[:, 0:1], 1+np.zeros((Tarray_ML.shape[0], 1))), axis=1)
    Sarray_ML = np.concatenate((Sarray_ML[:, 0:1], Sarray_ML_target[:, 0:1], np.zeros((Sarray_ML.shape[0], 1))), axis=1)

    print("finish reading ML MET files")
    print("\n*****************************************\n")

    # load PUPPI MET (TTbar-1, SingleNeutrino-0)
    Tarray_PU = np.load(os.path.join(args.input, "TTbar_feature_array_PUMET.npy"))
    Tarray_PU_target = np.load(os.path.join(args.input, "TTbar_target_array_PUMET.npy"))
    print("reading", args.input, "TTbar_feature_array_PUMET.npy")
    Sarray_PU = np.load(os.path.join(args.input, "SingleNeutrino_feature_array_PUMET.npy"))
    Sarray_PU_target = np.load(os.path.join(args.input, "SingleNeutrino_target_array_PUMET.npy"))
    print("reading", args.input, "SingleNeutrino_feature_array_PUMET.npy")
    Tarray_PU = np.concatenate((Tarray_PU[:, 0:1], Tarray_PU_target[:, 0:1], 1+np.zeros((Tarray_PU.shape[0], 1))), axis=1)
    Sarray_PU = np.concatenate((Sarray_PU[:, 0:1], Sarray_PU_target[:, 0:1], np.zeros((Sarray_PU.shape[0], 1))), axis=1)

    print("finish reading PUPPI MET files")
    print("\n*****************************************\n")

    # concatenate TTbar and SingleNeutrino and shuffle
    ML1 = Tarray_ML
    ML0 = Sarray_ML

    PU1 = Tarray_PU
    PU0 = Sarray_PU

    bin_number = 300.
    step = 2.

    ML_array = np.zeros((int(bin_number), 3))
    PU_array = np.zeros((int(bin_number), 3))

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

        ML1_count = np.sum(ML1[:, 0] > i*step)
        ML0_count = np.sum(ML0[:, 0] > i*step)
        Ta1_count = np.sum(PU1[:, 1] > i*step)
        Ta0_count = np.sum(PU0[:, 1] > i*step)

        TP = ML1_count
        FP = ML0_count
        FN = All1_count - ML1_count
        TN = All0_count - ML0_count

        # save plot data. -> TPR, FPR
        ML_array[i, 0] = TP / (TP + FN)  # TPR
        ML_array[i, 1] = FP / (FP + TN)  # FPR
        ML_array[i, 2] = step*i

        ML_rate[i] = ML1_count/All1_count
        ML_rate_SN[i] = ML0_count/All0_count
        target_rate[i] = Ta1_count/All1_count
        target_rate_SN[i] = Ta0_count/All0_count

        # PU

        PU1_count = np.sum(PU1[:, 0] > i*step)
        PU0_count = np.sum(PU0[:, 0] > i*step)

        TP = PU1_count
        FP = PU0_count
        FN = All1_count - PU1_count
        TN = All0_count - PU0_count

        # save plot data. -> TPR, FPR
        PU_array[i, 0] = TP / (TP + FN)  # TPR
        PU_array[i, 1] = FP / (FP + TN)  # FPR
        PU_array[i, 2] = step*i

        PU_rate[i] = PU1_count/All1_count
        PU_rate_SN[i] = PU0_count/All0_count

    which_plot = args.plot

    if which_plot == "ROC":
        #ML_sort_idx = np.argsort(ML_array[:, 1])
        #PU_sort_idx = np.argsort(PU_array[:, 1])

        #ML_AUC = auc(ML_array[:, 1], ML_array[:, 0])
        #PU_AUC = auc(PU_array[:, 1], PU_array[:, 0])
        
        #new AUC calculation, last zeroed out all contributions
        y_true  = np.r_[np.ones(All1_count), np.zeros(All0_count)]
        y_score = np.r_[ML1[:,0],            ML0[:,0]]        
        y_score_PU = np.r_[PU1[:,0],            PU0[:,0]]       
        #fpr, tpr, _ = roc_curve(y_true, y_score)   # unique, monotonic FPR
        ml_auc      = roc_auc_score(y_true, y_score)
        pu_auc     = roc_auc_score(y_true, y_score_PU)
        print("ML AUC : {}".format(ml_auc))
        print("PU AUC : {}".format(pu_auc))

        plt.plot(ML_array[:, 0], ML_array[:, 1], label='ML ROC, AUC = {}'.format(round(ml_auc, 3)))
        plt.plot(PU_array[:, 0], PU_array[:, 1], '-r', label='PUPPI ROC, AUC = {}'.format(round(pu_auc, 3)))
        plt.grid(True, axis='x', color='gray', alpha=0.5, linestyle='--')
        plt.xlabel('Signal Efficiency (TPR)')
        plt.xlim(0., 1.)
        plt.yscale("log")
        plt.ylim(1e-6, 1.1)
        plt.grid(True, axis='y', color='gray', alpha=0.5, linestyle='--')
        plt.ylabel('Backgground Efficiency (FPR)')
        plt.title('ttbar vs single neutrino - ROC')
        plt.legend()
        plt.savefig('ROC_curve.png')

    elif which_plot == "rate":
        x_ = range(0, int(step*bin_number), int(step))
        plt.plot(x_, ML_rate, 'bo', label='ML')
        plt.plot(x_, PU_rate, 'ro', label='PUPPI')
        plt.grid(True, axis='x', color='gray', alpha=0.5, linestyle='--')
        plt.grid(True, axis='y', color='gray', alpha=0.5, linestyle='--')
        plt.xlim(0, 200)
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

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze L1 MET ML model performance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--input', 
        type=str, 
        required=True,
        help='Input directory containing numpy arrays (output path of training)'
    )
    
    parser.add_argument(
        '--plot', 
        type=str, 
        required=True,
        choices=['ROC', 'rate', 'rate_com'],
        help='Type of plot to generate'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='plots',
        help='Output directory for plots'
    )
    
    return parser.parse_args()


# Configuration
if __name__ == "__main__":
    args = parse_arguments()
    main(args)
