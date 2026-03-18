"""
evaluate.py — model.h5로 test set에 대한 성능 플롯 생성

사용법:
    TF_USE_LEGACY_KERAS=1 python evaluate.py \
        --model results/eta_puppi_pdgid/model.h5 \
        --input inputs/ \
        --output results/eta_puppi_pdgid/eval_plots/ \
        --config configs/eta_puppi_pdgid.yaml

    # 또는 수동 설정
    TF_USE_LEGACY_KERAS=1 python evaluate.py \
        --model results/eta_puppi_pdgid/model.h5 \
        --input inputs/ \
        --output results/eta_puppi_pdgid/eval_plots/ \
        --feature-mode eta_puppi_pdgid \
        --n-features-pf 4 \
        --n-features-pf-cat 1 \
        --normFac 100

생성 플롯:
    - MET x, y, pt 분포 (ML / PUPPI / Truth)
    - Response vs Gen MET
    - Resolution (XY, pt, phi) vs Gen MET
    - Relative pt error
    - Absolute pt / phi error
    - 2D scatter: predicted vs gen MET
    - Loss history curve (있는 경우)
"""

import argparse
import os
import random
from glob import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import tqdm

os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')

import tf_keras as tfk
import tensorflow as tf

_gpus = tf.config.list_physical_devices('GPU')
for _gpu in _gpus:
    tf.config.experimental.set_memory_growth(_gpu, True)
print(f"[GPU] {[g.name for g in _gpus] if _gpus else 'No GPU, using CPU'}")

from DataGenerator import DataGenerator
from utils import MakePlots, convertXY2PtPhi, preProcessing


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _resolqt(y):
    return (np.percentile(y, 84) - np.percentile(y, 16)) / 2.0


def plot_loss_history(log_path, output_dir):
    """loss_history.log 가 있으면 학습 곡선 플롯."""
    if not os.path.isfile(log_path):
        return
    import csv
    epochs, train_loss, val_loss = [], [], []
    with open(log_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']) + 1)
            train_loss.append(float(row['loss']))
            val_loss.append(float(row['val_loss']))

    plt.style.use(hep.style.CMS)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_loss, label='Train loss')
    ax.plot(epochs, val_loss, label='Val loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.set_title('Training history')
    fig.savefig(os.path.join(output_dir, 'loss_history.png'), bbox_inches='tight')
    plt.close(fig)
    print(f"  saved: loss_history.png  (best val_loss={min(val_loss):.4f} @ epoch {epochs[val_loss.index(min(val_loss))]})")


def plot_2d_scatter(predict_pt, gen_pt, puppi_pt, output_dir):
    """2D scatter: predicted MET vs gen MET."""
    plt.style.use(hep.style.CMS)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    lim = 500
    bins = np.linspace(0, lim, 100)

    for ax, pred, label, color in zip(
        axes,
        [predict_pt, puppi_pt],
        ['ML', 'PUPPI'],
        ['Reds', 'Greens'],
    ):
        h = ax.hist2d(gen_pt, pred, bins=[bins, bins], cmap=color)
        fig.colorbar(h[3], ax=ax)
        ax.plot([0, lim], [0, lim], 'k--', lw=1, label='y=x')
        ax.set_xlabel('Gen MET [GeV]')
        ax.set_ylabel(f'{label} MET [GeV]')
        ax.set_title(f'{label} vs Gen MET')
        ax.legend()

    fig.savefig(os.path.join(output_dir, '2D_MET.png'), bbox_inches='tight')
    plt.close(fig)
    print('  saved: 2D_MET.png')


def plot_response_resolution(predict_ptPhi, puppi_ptPhi, true_ptPhi, output_dir):
    """Response + binned resolution (pt) in a single figure."""
    plt.style.use(hep.style.CMS)
    nbins = 20
    binnings = np.linspace(0, 400, nbins + 1)

    from scipy.stats import binned_statistic
    truth_means, bin_edges, binnumber = binned_statistic(
        true_ptPhi[:, 0], true_ptPhi[:, 0], statistic='mean',
        bins=binnings, range=(0, 400))
    ml_means, _, _    = binned_statistic(
        true_ptPhi[:, 0], predict_ptPhi[:, 0], statistic='mean',
        bins=binnings, range=(0, 400))
    puppi_means, _, _ = binned_statistic(
        true_ptPhi[:, 0], puppi_ptPhi[:, 0], statistic='mean',
        bins=binnings, range=(0, 400))

    # ----- response -----
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.hlines(truth_means / truth_means, bin_edges[:-1], bin_edges[1:],
              colors='k', lw=4, label='Truth')
    ax.hlines(ml_means / truth_means,    bin_edges[:-1], bin_edges[1:],
              colors='r', lw=4, label='ML')
    ax.hlines(puppi_means / truth_means, bin_edges[:-1], bin_edges[1:],
              colors='g', lw=4, label='PUPPI')
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 1.3)
    ax.set_xlabel('Truth MET [GeV]')
    ax.set_ylabel('<MET Estimation> / <MET Truth>')
    ax.legend(loc='lower right')
    fig.savefig(os.path.join(output_dir, 'response.png'), bbox_inches='tight')
    plt.close(fig)
    print('  saved: response.png')

    # ----- resolution -----
    respCorr_ml    = np.take(truth_means / ml_means,
                             np.digitize(true_ptPhi[:, 0], binnings) - 1, mode='clip')
    respCorr_puppi = np.take(truth_means / puppi_means,
                             np.digitize(true_ptPhi[:, 0], binnings) - 1, mode='clip')

    predict_xy = np.stack([predict_ptPhi[:, 0] * np.cos(predict_ptPhi[:, 1]),
                            predict_ptPhi[:, 0] * np.sin(predict_ptPhi[:, 1])], axis=-1)
    puppi_xy   = np.stack([puppi_ptPhi[:, 0] * np.cos(puppi_ptPhi[:, 1]),
                            puppi_ptPhi[:, 0] * np.sin(puppi_ptPhi[:, 1])], axis=-1)
    true_xy    = np.stack([true_ptPhi[:, 0] * np.cos(true_ptPhi[:, 1]),
                            true_ptPhi[:, 0] * np.sin(true_ptPhi[:, 1])], axis=-1)

    res_x_ml, _, _ = binned_statistic(
        true_ptPhi[:, 0],
        true_xy[:, 0] - predict_xy[:, 0] * respCorr_ml,
        statistic=_resolqt, bins=binnings, range=(0, 400))
    res_x_puppi, _, _ = binned_statistic(
        true_ptPhi[:, 0],
        true_xy[:, 0] - puppi_xy[:, 0] * respCorr_puppi,
        statistic=_resolqt, bins=binnings, range=(0, 400))
    res_y_ml, _, _ = binned_statistic(
        true_ptPhi[:, 0],
        true_xy[:, 1] - predict_xy[:, 1] * respCorr_ml,
        statistic=_resolqt, bins=binnings, range=(0, 400))
    res_y_puppi, _, _ = binned_statistic(
        true_ptPhi[:, 0],
        true_xy[:, 1] - puppi_xy[:, 1] * respCorr_puppi,
        statistic=_resolqt, bins=binnings, range=(0, 400))

    nEvt, _ = np.histogram(binnumber, bins=nbins, range=(1, nbins))
    rootN   = np.sqrt(np.maximum(nEvt, 1))
    bc      = (bin_edges[:-1] + bin_edges[1:]) / 2

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    for ax, res_ml, res_puppi, title in zip(
        axes,
        [res_x_ml, res_y_ml],
        [res_x_puppi, res_y_puppi],
        ['MET-x Resolution', 'MET-y Resolution'],
    ):
        ax.hlines(res_ml,    bin_edges[:-1], bin_edges[1:], colors='r', lw=3, label='ML')
        ax.hlines(res_puppi, bin_edges[:-1], bin_edges[1:], colors='g', lw=3, label='PUPPI')
        ax.errorbar(bc, res_ml,    yerr=res_ml / rootN,    fmt='none', color='r')
        ax.errorbar(bc, res_puppi, yerr=res_puppi / rootN, fmt='none', color='g')
        ax.set_xlabel('Truth MET [GeV]')
        ax.set_ylabel(r'$\sigma$(MET) [GeV]')
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
    fig.savefig(os.path.join(output_dir, 'resolution_XY.png'), bbox_inches='tight')
    plt.close(fig)
    print('  saved: resolution_XY.png')


def plot_rel_error(predict_pt, puppi_pt, gen_pt, output_dir):
    """Relative pt error: (pred - gen) / gen"""
    plt.style.use(hep.style.CMS)
    fig, ax = plt.subplots(figsize=(10, 7))
    for pred, label, color in [(predict_pt, 'ML', 'red'), (puppi_pt, 'PUPPI', 'green')]:
        rel = (pred - gen_pt) / np.maximum(gen_pt, 1e-6)
        rel = rel[rel < 3]
        ax.hist(rel, bins=np.linspace(-3, 3, 61), alpha=0.6,
                label=f'{label}  μ={np.mean(rel):.3f}  σ={np.std(rel):.3f}',
                color=color, histtype='stepfilled')
    ax.set_xlabel('(pred - gen) / gen')
    ax.set_ylabel('Events')
    ax.legend()
    ax.set_title('Relative MET pt error')
    fig.savefig(os.path.join(output_dir, 'rel_pt_error.png'), bbox_inches='tight')
    plt.close(fig)
    print('  saved: rel_pt_error.png')


def plot_abs_pt_error(predict_pt, puppi_pt, gen_pt, output_dir):
    """Absolute pt error: pred - gen"""
    plt.style.use(hep.style.CMS)
    fig, ax = plt.subplots(figsize=(10, 7))
    for pred, label, color in [(puppi_pt, 'PUPPI', 'green'), (predict_pt, 'ML', 'red')]:
        err = pred - gen_pt
        ax.hist(err, bins=np.linspace(-250, 250, 61), alpha=0.6,
                label=f'{label}  μ={np.mean(err):.2f}  σ={np.std(err):.2f}',
                color=color, histtype='stepfilled')
    ax.set_xlabel('pred - gen  [GeV]')
    ax.set_ylabel('Events')
    ax.legend()
    ax.set_title('Absolute MET pt error')
    fig.savefig(os.path.join(output_dir, 'abs_pt_error.png'), bbox_inches='tight')
    plt.close(fig)
    print('  saved: abs_pt_error.png')


def plot_abs_phi_error(predict_phi, puppi_phi, gen_phi, output_dir):
    """Absolute phi error."""
    plt.style.use(hep.style.CMS)
    fig, ax = plt.subplots(figsize=(10, 7))
    for pred, label, color in [(puppi_phi, 'PUPPI', 'green'), (predict_phi, 'ML', 'red')]:
        err = pred - gen_phi
        err = np.where(err > np.pi,  err - 2*np.pi, err)
        err = np.where(err < -np.pi, err + 2*np.pi, err)
        ax.hist(err, bins=np.linspace(-3.5, 3.5, 61), alpha=0.6,
                label=f'{label}  μ={np.mean(err):.4f}  σ={np.std(err):.4f}',
                color=color, histtype='stepfilled')
    ax.set_xlabel(r'$\Delta\phi$ (pred - gen)  [rad]')
    ax.set_ylabel('Events')
    ax.legend()
    ax.set_title(r'Absolute MET $\phi$ error')
    fig.savefig(os.path.join(output_dir, 'abs_phi_error.png'), bbox_inches='tight')
    plt.close(fig)
    print('  saved: abs_phi_error.png')


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Evaluate model.h5 on test set and generate plots')
    parser.add_argument('--model',  required=True, help='Path to model.h5')
    parser.add_argument('--input',  required=True, help='Directory containing .h5 input files')
    parser.add_argument('--output', required=True, help='Directory to save plots')
    parser.add_argument('--config', default=None,  help='(Optional) YAML config file')
    parser.add_argument('--feature-mode',     default='eta_puppi_pdgid',
                        choices=['full', 'eta_puppi_pdgid'], help='Feature mode')
    parser.add_argument('--n-features-pf',     type=int, default=4)
    parser.add_argument('--n-features-pf-cat', type=int, default=1)
    parser.add_argument('--normFac',           type=float, default=100.0)
    parser.add_argument('--maxNPF',            type=int, default=128)
    parser.add_argument('--batch-size',        type=int, default=512)
    parser.add_argument('--test-fraction',     type=float, default=0.1,
                        help='Fraction of files to use as test set (default 0.1)')
    parser.add_argument('--all-files',         action='store_true',
                        help='Use ALL files for evaluation (ignores --test-fraction)')
    parser.add_argument('--loss-log',          default=None,
                        help='Path to loss_history.log for training curve (auto-detected if omitted)')
    args = parser.parse_args()

    # ---- config override ----
    if args.config:
        from config import load_config
        cfg = load_config(args.config)
        args.feature_mode      = cfg.get('data.feature_mode',    args.feature_mode)
        args.n_features_pf     = cfg.get('data.n_features_pf',   args.n_features_pf)
        args.n_features_pf_cat = cfg.get('data.n_features_pf_cat', args.n_features_pf_cat)
        args.normFac           = cfg.get('training.normFac',     args.normFac)
        args.maxNPF            = cfg.get('data.maxNPF',          args.maxNPF)
        print(f'Config loaded from {args.config}')

    os.makedirs(args.output, exist_ok=True)
    normFac = args.normFac

    # ---- file split ----
    all_files = sorted(glob(os.path.join(args.input, '*.h5')))
    assert len(all_files) >= 1, f'No .h5 files found in {args.input}'

    if args.all_files:
        test_files = all_files
    else:
        n_test = max(1, int(args.test_fraction * len(all_files)))
        # mirror the same split logic as train.py (last N files after sort+shuffle)
        valid_n = max(1, int(0.1 * len(all_files)))
        train_n = len(all_files) - 2 * valid_n
        test_files = all_files[train_n + valid_n : train_n + valid_n + valid_n]
        if not test_files:
            test_files = all_files[-n_test:]

    print(f'Test files ({len(test_files)}):')
    for f in test_files:
        print(f'  {os.path.basename(f)}')

    # ---- data generator ----
    gen = DataGenerator(
        list_files=test_files,
        batch_size=args.batch_size,
        maxNPF=args.maxNPF,
        n_features_pf_cat=args.n_features_pf_cat,
        feature_mode=args.feature_mode,
    )

    # ---- load model ----
    print(f'\nLoading model: {args.model}')
    import tf_keras.backend as K
    keras_model = tfk.models.load_model(
        args.model,
        compile=False,
        custom_objects={'K': K},
    )
    keras_model.summary()

    # ---- inference ----
    print('\nRunning inference...')
    predict_raw = keras_model.predict(gen, verbose=1)   # shape (N, 2) [px, py], in GeV

    # ---- collect ground truth and PUPPI ----
    # DataGenerator.normFac = 1.0 (hardcoded), so:
    #   Yr  = y_h5 as-is (already in GeV)
    #   Xp (pxpy) = px/normFac, py/normFac  (normalized by training normFac)
    # The model was trained directly against Yr (in GeV), so predict_raw is also in GeV.
    # PUPPI baseline = normFac * sum(pxpy) = sum(px_GeV) over all particles.
    all_puppi, all_yr = [], []
    for Xr, Yr in tqdm.tqdm(gen, desc='Collecting truth'):
        puppi_xy = np.sum(Xr[1], axis=1)   # shape (batch, 2), in normalized units
        all_puppi.append(puppi_xy)
        all_yr.append(Yr)

    PUPPI_pt = normFac * np.concatenate(all_puppi)   # (N, 2) — scale back to GeV
    Yr_test  = np.concatenate(all_yr)                 # (N, 2) — already in GeV
    predict_test = predict_raw                         # already in GeV (no extra scaling)

    # ---- convert XY → pt/phi ----
    pred_ptPhi  = convertXY2PtPhi(predict_test)
    puppi_ptPhi = convertXY2PtPhi(PUPPI_pt)
    true_ptPhi  = convertXY2PtPhi(Yr_test)

    # ---- generate plots ----
    print(f'\nSaving plots to: {args.output}')

    # 1. MET x, y, pt 1D distributions  (from utils.MakePlots)
    MakePlots(Yr_test, predict_test, PUPPI_pt, path_out=args.output + '/')
    print('  saved: MET_x.png  MET_y.png  MET_pt.png')
    print('         MET_response.png  XY_resolution_plots.png  pt_resolution_plots.png')

    # 2. Response + resolution (independent implementation for clarity)
    plot_response_resolution(pred_ptPhi, puppi_ptPhi, true_ptPhi, args.output)

    # 3. Relative pt error
    plot_rel_error(pred_ptPhi[:, 0], puppi_ptPhi[:, 0], true_ptPhi[:, 0], args.output)

    # 4. Absolute pt error
    plot_abs_pt_error(pred_ptPhi[:, 0], puppi_ptPhi[:, 0], true_ptPhi[:, 0], args.output)

    # 5. Absolute phi error
    plot_abs_phi_error(pred_ptPhi[:, 1], puppi_ptPhi[:, 1], true_ptPhi[:, 1], args.output)

    # 6. 2D scatter
    plot_2d_scatter(pred_ptPhi[:, 0], true_ptPhi[:, 0], puppi_ptPhi[:, 0], args.output)

    # 7. Loss history (auto-detect or user-specified)
    log_path = args.loss_log
    if log_path is None:
        model_dir = os.path.dirname(args.model)
        log_path  = os.path.join(model_dir, 'loss_history.log')
    plot_loss_history(log_path, args.output)

    # ---- summary stats ----
    print('\n=== Summary ===')
    print(f'  N events: {len(Yr_test)}')
    print(f'  ML   MET pt  mean±std: {pred_ptPhi[:,0].mean():.2f} ± {pred_ptPhi[:,0].std():.2f} GeV')
    print(f'  PUPPI MET pt mean±std: {puppi_ptPhi[:,0].mean():.2f} ± {puppi_ptPhi[:,0].std():.2f} GeV')
    print(f'  Truth MET pt mean±std: {true_ptPhi[:,0].mean():.2f} ± {true_ptPhi[:,0].std():.2f} GeV')
    print(f'  ML   resolution (overall): {_resolqt(true_ptPhi[:,0] - pred_ptPhi[:,0]):.3f} GeV')
    print(f'  PUPPI resolution (overall): {_resolqt(true_ptPhi[:,0] - puppi_ptPhi[:,0]):.3f} GeV')
    print(f'\nDone. All plots saved to: {args.output}')


if __name__ == '__main__':
    main()
