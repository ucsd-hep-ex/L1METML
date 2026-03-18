"""
Evaluation script that rebuilds the model from config and loads weights.
Workaround for Lambda layer deserialization issue in Keras 3.
"""
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # CPU only (GPU busy or use -1 to free)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import tqdm
from glob import glob
import csv

import tensorflow as tf
from config import load_config
from DataGenerator import DataGenerator
from models import dense_embedding
from utils import MakePlots, convertXY2PtPhi


def _resolqt(y):
    return (np.percentile(y, 84) - np.percentile(y, 16)) / 2.0


CONFIG   = 'configs/eta_puppi_pdgid.yaml'
MODEL_H5 = 'results/eta_puppi_pdgid/model.h5'
INPUT_DIR= 'inputs/'
OUT_DIR  = 'results/eta_puppi_pdgid/eval_plots/'
os.makedirs(OUT_DIR, exist_ok=True)

# ── load config ──────────────────────────────────────────────────────────────
cfg = load_config(CONFIG)
normFac        = cfg.get('training.normFac', 100)
maxNPF         = cfg.get('data.maxNPF', 128)
n_features_pf  = cfg.get('data.n_features_pf', 6)
n_features_cat = cfg.get('data.n_features_pf_cat', 2)
feature_mode   = cfg.get('data.feature_mode', 'full')
batch_size     = 512

# ── test files (same split logic as train.py) ─────────────────────────────
all_files = sorted(glob(os.path.join(INPUT_DIR, '*.h5')))
valid_n   = max(1, int(0.1 * len(all_files)))
train_n   = len(all_files) - 2 * valid_n
test_files = all_files[train_n + valid_n : train_n + valid_n + valid_n]
if not test_files:
    test_files = all_files[-valid_n:]
print(f'Test files ({len(test_files)}):')
for f in test_files: print(f'  {os.path.basename(f)}')

# ── data generator ────────────────────────────────────────────────────────
gen = DataGenerator(
    list_files=test_files,
    batch_size=batch_size,
    maxNPF=maxNPF,
    n_features_pf_cat=n_features_cat,
    feature_mode=feature_mode,
)

# ── rebuild model & load weights ──────────────────────────────────────────
Xr0, _ = gen[0]
emb_input_dim = {i: int(np.max(Xr0[2 + i])) + 1 for i in range(n_features_cat)}
print(f'Embedding dims: {emb_input_dim}')

model = dense_embedding(
    n_features       = n_features_pf,
    n_features_cat   = n_features_cat,
    activation       = cfg.get('model.activation', 'tanh'),
    number_of_pupcandis = maxNPF,
    embedding_input_dim = emb_input_dim,
    emb_out_dim      = cfg.get('model.emb_out_dim', 16),
    with_bias        = cfg.get('model.with_bias', True),
    t_mode           = cfg.get('training.mode', 1),
    units            = cfg.get('model.units', [128, 64, 32]),
)
model.load_weights(MODEL_H5)
print('Weights loaded.')
model.summary()

# ── collect data & run inference ─────────────────────────────────────────
print('\nCollecting batches and running inference...')
all_preds, all_puppi, all_yr = [], [], []
for i in tqdm.tqdm(range(len(gen))):
    Xr, Yr = gen[i]
    pred = model.predict(Xr, verbose=0)
    all_preds.append(pred)
    # PUPPI baseline: MET = -sum(puppi_weight * pxpy) per event
    # Truth Y is defined as the negative of the hadronic recoil (Y = -sum visible pT)
    # eta_puppi_pdgid mode: Xi=[eta(0), puppi_weight(1)]
    # full mode:            Xi=[pt(0), eta(1), phi(2), puppi_weight(3)]
    puppi_idx = 1 if feature_mode == 'eta_puppi_pdgid' else 3
    puppi_w = Xr[0][:, :, puppi_idx:puppi_idx+1]   # (batch, 128, 1)
    puppi_pxpy = puppi_w * Xr[1]                     # (batch, 128, 2)
    all_puppi.append(-np.sum(puppi_pxpy, axis=1))
    all_yr.append(Yr)

predict_raw = np.concatenate(all_preds)
PUPPI_xy = np.concatenate(all_puppi)   # (N,2) GeV  (DataGenerator.normFac=1.0, already in GeV)
Yr_test  = np.concatenate(all_yr)                 # (N,2) GeV
N = len(Yr_test)
predict_test = predict_raw[:N]
PUPPI_xy     = PUPPI_xy[:N]
Yr_test      = Yr_test[:N]

pred_ptPhi  = convertXY2PtPhi(predict_test)
puppi_ptPhi = convertXY2PtPhi(PUPPI_xy)
true_ptPhi  = convertXY2PtPhi(Yr_test)

# ── 1. MakePlots (from utils) ────────────────────────────────────────────
MakePlots(Yr_test, predict_test, PUPPI_xy, path_out=OUT_DIR)
print('  saved: MET_x/y/pt, MET_response, XY_resolution, pt_resolution')

# ── 2. Response ───────────────────────────────────────────────────────────
from scipy.stats import binned_statistic
nbins    = 20
binnings = np.linspace(0, 400, nbins + 1)
bc       = (binnings[:-1] + binnings[1:]) / 2

truth_means, _, _  = binned_statistic(true_ptPhi[:,0], true_ptPhi[:,0],  statistic='mean', bins=binnings)
ml_means,    _, _  = binned_statistic(true_ptPhi[:,0], pred_ptPhi[:,0],  statistic='mean', bins=binnings)
puppi_means, _, bn = binned_statistic(true_ptPhi[:,0], puppi_ptPhi[:,0], statistic='mean', bins=binnings)

plt.style.use(hep.style.CMS)
fig, ax = plt.subplots(figsize=(10,7))
ax.hlines(ml_means/truth_means,    binnings[:-1], binnings[1:], colors='r', lw=4, label='ML')
ax.hlines(puppi_means/truth_means, binnings[:-1], binnings[1:], colors='g', lw=4, label='PUPPI')
ax.axhline(1, color='k', lw=2, ls='--', label='Truth')
ax.set_xlim(0,400); ax.set_ylim(0,1.4)
ax.set_xlabel('Truth MET [GeV]'); ax.set_ylabel('<MET pred> / <MET truth>')
ax.set_title('MET Response'); ax.legend(loc='lower right')
fig.savefig(OUT_DIR+'response.png', bbox_inches='tight'); plt.close(fig)
print('  saved: response.png')

# ── 3. Resolution (XY) ────────────────────────────────────────────────────
respCorr_ml    = np.take(truth_means/np.where(ml_means==0,1,ml_means),
                         np.clip(np.digitize(true_ptPhi[:,0], binnings)-1, 0, nbins-1))
respCorr_puppi = np.take(truth_means/np.where(puppi_means==0,1,puppi_means),
                         np.clip(np.digitize(true_ptPhi[:,0], binnings)-1, 0, nbins-1))

pred_xy  = np.stack([pred_ptPhi[:,0]*np.cos(pred_ptPhi[:,1]),
                     pred_ptPhi[:,0]*np.sin(pred_ptPhi[:,1])], axis=-1)
puppi_xy2= np.stack([puppi_ptPhi[:,0]*np.cos(puppi_ptPhi[:,1]),
                     puppi_ptPhi[:,0]*np.sin(puppi_ptPhi[:,1])], axis=-1)
true_xy  = np.stack([true_ptPhi[:,0]*np.cos(true_ptPhi[:,1]),
                     true_ptPhi[:,0]*np.sin(true_ptPhi[:,1])], axis=-1)

nEvt, _  = np.histogram(np.digitize(true_ptPhi[:,0], binnings), bins=nbins, range=(1,nbins))
rootN    = np.sqrt(np.maximum(nEvt,1))

fig, axes = plt.subplots(1,2,figsize=(16,7),sharey=True)
for ax, dim, label in zip(axes, [0,1], ['MET-x Resolution','MET-y Resolution']):
    rx_ml,    _, _ = binned_statistic(true_ptPhi[:,0], true_xy[:,dim] - pred_xy[:,dim]*respCorr_ml,    statistic=_resolqt, bins=binnings)
    rx_puppi, _, _ = binned_statistic(true_ptPhi[:,0], true_xy[:,dim] - puppi_xy2[:,dim]*respCorr_puppi, statistic=_resolqt, bins=binnings)
    ax.hlines(rx_ml,    binnings[:-1], binnings[1:], colors='r', lw=3, label='ML')
    ax.hlines(rx_puppi, binnings[:-1], binnings[1:], colors='g', lw=3, label='PUPPI')
    ax.errorbar(bc, rx_ml,    yerr=rx_ml/rootN,    fmt='none', color='r')
    ax.errorbar(bc, rx_puppi, yerr=rx_puppi/rootN, fmt='none', color='g')
    ax.set_xlabel('Truth MET [GeV]'); ax.set_ylabel(r'$\sigma$(MET) [GeV]')
    ax.set_title(label); ax.legend(); ax.grid(True)
fig.savefig(OUT_DIR+'resolution_XY.png', bbox_inches='tight'); plt.close(fig)
print('  saved: resolution_XY.png')

# ── 4. Abs pt error ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10,7))
for pred, label, color in [(puppi_ptPhi[:,0],'PUPPI','green'),(pred_ptPhi[:,0],'ML','red')]:
    err = pred - true_ptPhi[:,0]
    ax.hist(err, bins=np.linspace(-250,250,61), alpha=0.6,
            label=f'{label}  μ={np.mean(err):.2f}  σ={np.std(err):.2f}',
            color=color, histtype='stepfilled')
ax.set_xlabel('pred - gen [GeV]'); ax.set_ylabel('Events')
ax.legend(); ax.set_title('Absolute MET pt error')
fig.savefig(OUT_DIR+'abs_pt_error.png', bbox_inches='tight'); plt.close(fig)
print('  saved: abs_pt_error.png')

# ── 5. Rel pt error ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10,7))
for pred, label, color in [(pred_ptPhi[:,0],'ML','red'),(puppi_ptPhi[:,0],'PUPPI','green')]:
    rel = (pred - true_ptPhi[:,0]) / np.maximum(true_ptPhi[:,0], 1e-6)
    rel = rel[np.abs(rel) < 3]
    ax.hist(rel, bins=np.linspace(-3,3,61), alpha=0.6,
            label=f'{label}  μ={np.mean(rel):.3f}  σ={np.std(rel):.3f}',
            color=color, histtype='stepfilled')
ax.set_xlabel('(pred-gen)/gen'); ax.set_ylabel('Events')
ax.legend(); ax.set_title('Relative MET pt error')
fig.savefig(OUT_DIR+'rel_pt_error.png', bbox_inches='tight'); plt.close(fig)
print('  saved: rel_pt_error.png')

# ── 6. 2D scatter ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1,2,figsize=(16,7))
lim  = 500
bins = np.linspace(0, lim, 100)
for ax, pred, label, cmap in zip(axes,
        [pred_ptPhi[:,0], puppi_ptPhi[:,0]], ['ML','PUPPI'], ['Reds','Greens']):
    h = ax.hist2d(true_ptPhi[:,0], pred, bins=[bins,bins], cmap=cmap)
    fig.colorbar(h[3], ax=ax)
    ax.plot([0,lim],[0,lim],'k--',lw=1,label='y=x')
    ax.set_xlabel('Gen MET [GeV]'); ax.set_ylabel(f'{label} MET [GeV]')
    ax.set_title(f'{label} vs Gen MET'); ax.legend()
fig.savefig(OUT_DIR+'2D_MET.png', bbox_inches='tight'); plt.close(fig)
print('  saved: 2D_MET.png')

# ── 7. Loss history ───────────────────────────────────────────────────────
log_path = os.path.join(os.path.dirname(MODEL_H5), 'loss_history.log')
if os.path.isfile(log_path):
    epochs, tr, vl = [], [], []
    with open(log_path) as f:
        for row in csv.DictReader(f):
            epochs.append(int(row['epoch'])+1)
            tr.append(float(row['loss']))
            vl.append(float(row['val_loss']))
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(epochs, tr, label='Train loss')
    ax.plot(epochs, vl, label='Val loss')
    best_e = epochs[vl.index(min(vl))]
    ax.axvline(best_e, color='gray', ls='--', label=f'Best val (epoch {best_e})')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.legend(); ax.set_title('Training history')
    fig.savefig(OUT_DIR+'loss_history.png', bbox_inches='tight'); plt.close(fig)
    print(f'  saved: loss_history.png  (best val_loss={min(vl):.4f} @ epoch {best_e})')

# ── summary ───────────────────────────────────────────────────────────────
print('\n=== Summary ===')
print(f'  N events : {N}')
print(f'  ML   MET pt  mean±std: {pred_ptPhi[:,0].mean():.2f} ± {pred_ptPhi[:,0].std():.2f} GeV')
print(f'  PUPPI MET pt mean±std: {puppi_ptPhi[:,0].mean():.2f} ± {puppi_ptPhi[:,0].std():.2f} GeV')
print(f'  Truth MET pt mean±std: {true_ptPhi[:,0].mean():.2f} ± {true_ptPhi[:,0].std():.2f} GeV')
print(f'  ML    resolution: {_resolqt(true_ptPhi[:,0]-pred_ptPhi[:,0]):.3f} GeV')
print(f'  PUPPI resolution: {_resolqt(true_ptPhi[:,0]-puppi_ptPhi[:,0]):.3f} GeV')
print(f'  ML   response (mean): {(pred_ptPhi[:,0]/np.maximum(true_ptPhi[:,0],1)).mean():.3f}')
print(f'\nAll plots saved to: {OUT_DIR}')
