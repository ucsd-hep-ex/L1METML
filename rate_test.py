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
from enum import Enum
import numpy as np
import numpy.ma
import time
import argparse
import collections as co
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Defining available datasets
class Dataset(Enum):
    TTBAR = "TTbar"
    SINGLE_NEUTRINO = "SingleNeutrino"
    HIGGS_TO_INVISIBLE = "HtoInvisible"
    SUSY = "SUSY"

# Signal/background mappings
PHYSICS_MAPPINGS = {
    Dataset.TTBAR: "signal",
    Dataset.HIGGS_TO_INVISIBLE: "signal", 
    Dataset.SUSY: "signal",
    Dataset.SINGLE_NEUTRINO: "background" 
}
@dataclass
class AnalysisConfig:
    """Configuration for rate analysis."""
    signal_dataset: Dataset #set via CL arg

    bin_number: int = 300
    step: float = 2.0
    trigger_rate_khz: float = 31000.0
    max_threshold_gev: int = 200
    output_dir: str = "plots"

    background_dataset: Dataset = Dataset.SINGLE_NEUTRINO

@dataclass
class DataArrays:
    '''Holds numpy arrays for ML and PUPPI MET data.'''
    signal_ml: np.ndarray
    background_ml: np.ndarray
    signal_puppi: np.ndarray
    background_puppi: np.ndarray

class DataLoader:
    """Handles loading and preprocessing of numpy arrays."""
    """Different from training data-laoder"""
    
    def __init__(self, input_path: str,config: AnalysisConfig):
        self.input_path = Path(input_path)
        self.config = config

    def load_arrays(self) -> DataArrays:
        """Load and preprocess all required arrays."""
        logger.info(f"Loading data from {self.input_path}")

        #Load ML arrays
        signal_ml = self._load_and_process_arrays(
            self.config.signal_dataset.value, "MLMET"
            )
        background_ml = self._load_and_process_arrays(
            self.config.background_dataset.value, "MLMET", is_signal=False
            )

        #Load PUPPI arrays
        signal_puppi = self._load_and_process_arrays(
            self.config.signal_dataset.value, "PUMET"
            )
        background_puppi = self._load_and_process_arrays(
            self.config.background_dataset.value, "PUMET", is_signal=False
            )

        return DataArrays(signal_ml, background_ml, signal_puppi, background_puppi)

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
        rates = {
            'ml_signal': np.zeros(bin_count),
            'ml_background': np.zeros(bin_count),
            'puppi_signal': np.zeros(bin_count),
            'puppi_background': np.zeros(bin_count),
            'ml_roc': np.zeros((bin_count, 3)), # TPR, FPR, threshold
            'puppi_roc': np.zeros((bin_count, 3)) # TPR, FPR, threshold
        }

        signal_total = data.signal_ml.shape[0]
        background_total = data.signal_ml.shape[0]

        for i in range(bin_count):
            threshold = i * self.config.step

            # ML rates
            ml_sig_pass = np.sum(data.signal_ml[:,0] > threshold)
            ml_bg_pass = np.sum(data.background_ml[:,0] > threshold)

            rates['ml_signal'][i] = ml_sig_pass / signal_total
            rates['ml_background'][i] = ml_bg_pass / background_total

            # ML ROC values
            rates['ml_roc'][i] = self._calculate_roc_point(
                ml_sig_pass,
                ml_bg_pass,
                signal_total,
                background_total,
                threshold
            )

            # PUPPI rates
            puppi_sig_pass = np.sum(data.signal_puppi[:,0] > threshold)
            puppi_bg_pass = np.sum(data.background_puppi[:,0] > threshold)

            rates['puppi_signal'][i] = puppi_sig_pass / signal_total
            rates['puppi_background'][i] = puppi_bg_pass / background_total

            # PUPPI ROC values
            rates['puppi_roc'][i] = self._calculate_roc_point(
                puppi_sig_pass,
                puppi_bg_pass,
                signal_total,
                background_total,
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
            np.ones(data.signal_ml.shape[0]),
            np.zeros(data.background_ml.shape[0])
        ])

        y_score_ml = np.concatenate([data.signal_ml[:, 0], data.background_ml[:, 0]])
        y_score_puppi = np.concatenate([data.signal_puppi[:, 0], data.background_puppi[:, 0]])

        ml_auc = roc_auc_score(y_true, y_score_ml)
        puppi_auc = roc_auc_score(y_true, y_score_puppi)

        return ml_auc, puppi_auc

class PlotGenerator:
    """Generates various plots for rate/ROC analysis."""

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.signal_sample = self.config.signal_dataset.value
        self.background_sample = self.config.background_dataset.value

    def plot_roc_curve(self, rates: Dict[str, np.ndarray], auc_scores: Tuple[float, float]) -> None:
        """Generate ROC curve plot."""
        ml_auc, puppi_auc = auc_scores

        plt.figure(figsize=(8, 6))
        plt.plot(rates['ml_roc'][:,0], rates['ml_roc'][:,1],
                 label=f'ML ROC, AUC = {ml_auc:.3f}', linewidth=2)
        plt.plot(rates['puppi_roc'][:,0], rates['puppi_roc'][:,1],
                 label=f'PUPPI ROC, AUC = {puppi_auc:.3f}', linewidth=2, color='red')
        
        self._setup_plot_style()
        plt.xlabel('Signal Efficiency (TPR)')
        plt.ylabel('Background Efficiency (FPR)')
        plt.title('ROC Curve: ML vs PUPPI MET')
        plt.xlim(0., 1.)
        plt.yscale("log")
        plt.ylim(1e-6, 1.1)
        plt.legend()

        output_path = self.output_dir / f'ROC_curve_{self.signal_sample}_{self.background_sample}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"ROC curve saved to {output_path}")
    
    def plot_trigger_rates(self, rates: Dict[str, np.ndarray]) -> None:
        """Generate trigger rate plot."""
        thresholds = np.arange(0, self.config.step * self.config.bin_number, self.config.step)

        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, rates['ml_signal'], 'bo', label = 'ML', markersize=2)
        plt.plot(thresholds, rates['puppi_signal'], 'ro', label = 'PUPPI', markersize=2)

        self._setup_plot_style()
        plt.xlim(0, self.config.max_threshold_gev)
        plt.xlabel('MET Threshold (GeV)')
        plt.ylabel(f'{self.signal_sample} Efficiency')
        plt.title('Trigger Rates: ML vs PUPPI MET')
        plt.legend()

        output_path = self.output_dir / f'trigger_rates_{self.signal_sample}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Trigger rates plot saved to {output_path}")

    # refactor of old combined rates, may be outdated (replaced by ROC curve)
    def plot_combined_rates(self, rates: Dict[str, np.ndarray]) -> None:
        """Generate combined rates plot for ML and PUPPI."""
        plt.figure(figsize=(10, 6))

        ml_bg_rate = rates['ml_background'] * self.config.trigger_rate_khz
        puppi_bg_rate = rates['puppi_background'] * self.config.trigger_rate_khz

        plt.plot(rates['ml_signal'], ml_bg_rate, 'bo', label='ML', markersize=2)
        plt.plot(rates['puppi_signal'], puppi_bg_rate, 'ro', label='PUPPI', markersize=2)

        self._setup_plot_style()
        plt.yscale("log")
        plt.xlabel(f'{self.signal_sample} Efficiency')
        plt.ylabel(f'{self.background_sample} (kHz)')
        plt.title('Combined Rates: ML vs PUPPI MET')
        plt.legend()

        output_path = self.output_dir / 'combined_rates.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Combined rates plot saved to {output_path}")
    
    def _setup_plot_style(self) -> None:
        """Apply consistent style to plots."""
        plt.grid(True, color='gray', alpha=0.5, linestyle='--')
        plt.tight_layout()


def main(args: argparse.Namespace) -> None:

    try:
        signal_dataset = Dataset(args.signal)
    except ValueError:
        logger.error(f"Invalid dataset.")
        raise ValueError(f"Invalid dataset.")

    config = AnalysisConfig(signal_dataset=signal_dataset)

    # Load data 
    loader = DataLoader(args.input, config)
    data = loader.load_arrays()

    # Perform analysis
    analyzer = RateAnalyzer(config)
    rates = analyzer.calculate_rates(data)

    # Generate plots
    plotter = PlotGenerator(config)

    if args.plot == "ROC":
        auc_scores = analyzer.calculate_auc(data)
        logger.info(f"ML AUC: {auc_scores[0]:.4f}")
        logger.info(f"PUPPI AUC: {auc_scores[1]:.4f}")
        plotter.plot_roc_curve(rates, auc_scores)

    elif args.plot == "rate":
        plotter.plot_trigger_rates(rates)

    elif args.plot == "rate_com":
        plotter.plot_combined_rates(rates)
    
    else:
        raise ValueError(f"Unknown plot type: {args.plot}")
    
    logger.info("Analysis complete. Plots saved in output directory.")
        
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
        '--signal',
        type=str,
        required=True,
        choices=['TTbar', 'HtoInvisible', 'SUSY'],
        help='Signal dataset'
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
