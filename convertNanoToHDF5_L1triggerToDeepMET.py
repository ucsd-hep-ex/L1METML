#!/usr/bin/env python

import optparse
import sys
import uproot
import numpy as np
import awkward as ak
import h5py
#import progressbar
from tqdm import tqdm
import os

from utils import to_np_array

'''
widgets=[
    progressbar.SimpleProgress(), ' - ', progressbar.Timer(), ' - ', progressbar.Bar(), ' - ', progressbar.AbsoluteETA()
]
'''


def deltaR(eta1, phi1, eta2, phi2):
    """ calculate deltaR """
    dphi = (phi1-phi2)
    while dphi > np.pi:
        dphi -= 2*np.pi
    while dphi < -np.pi:
        dphi += 2*np.pi
    deta = eta1-eta2
    return np.hypot(deta, dphi)


# configuration
usage = 'usage: %prog [options]'
parser = optparse.OptionParser(usage)
parser.add_option('-i', '--input', dest='input', help='input file', default='', type='string')
parser.add_option('-o', '--output', dest='output', help='output file', default='', type='string')
parser.add_option("-N", "--maxevents", dest='maxevents', help='max number of events', default=-1, type='int')
parser.add_option("--data", dest="data", action="store_true", default=False, help="input is data. The default is MC")
(opt, args) = parser.parse_args()

if opt.input == '' or opt.output == '':
    sys.exit('Need to specify input and output files!')

##
varList = [
        'nL1PuppiCands', 'L1PuppiCands_pt', 'L1PuppiCands_eta', 'L1PuppiCands_phi',
        'L1PuppiCands_charge', 'L1PuppiCands_pdgId', 'L1PuppiCands_puppiWeight'
]

# event-level variables

varList_mc = [
    'genMet_pt', 'genMet_phi',
]

d_encoding = {
    'L1PuppiCands_charge': {-999.0: 0,
                            -1.0: 1,
                            0.0: 2,
                            1.0: 3},
    'L1PuppiCands_pdgId': {-999.0: 0,
                           -211.0: 1,
                           -130.0: 2,
                           -22.0: 3,
                           -13.0: 4,
                           -11.0: 5,
                           11.0: 5,
                           13.0: 4,
                           22.0: 3,
                           130.0: 2,
                           211.0: 1}
}

if not opt.data:
    varList = varList + varList_mc

upfile = uproot.open(opt.input)
tree = upfile['Events'].arrays(varList, entry_stop=opt.maxevents)
# general setup
maxNPuppi = 128
nFeatures = 8
if opt.maxevents == -1:
    maxEntries = len(tree['nL1PuppiCands'])
else:
    maxEntries = opt.maxevents 
# input Puppi candidates
X = np.zeros(shape=(maxEntries, maxNPuppi, nFeatures), dtype=float, order='F')
# recoil estimators
Y = np.zeros(shape=(maxEntries, 2), dtype=float, order='F')

pt = to_np_array(tree['L1PuppiCands_pt'], maxN=maxNPuppi)
eta = to_np_array(tree['L1PuppiCands_eta'], maxN=maxNPuppi)
phi = to_np_array(tree['L1PuppiCands_phi'], maxN=maxNPuppi)
pdgid = to_np_array(tree['L1PuppiCands_pdgId'], maxN=maxNPuppi, pad=-999)
charge = to_np_array(tree['L1PuppiCands_charge'], maxN=maxNPuppi, pad=-999)
puppiw = to_np_array(tree['L1PuppiCands_puppiWeight'], maxN=maxNPuppi)

X[:, :, 0] = pt
X[:, :, 1] = pt * np.cos(phi)
X[:, :, 2] = pt * np.sin(phi)
X[:, :, 3] = eta
X[:, :, 4] = phi
X[:, :, 5] = puppiw

# encoding
X[:, :, 6] = np.vectorize(d_encoding['L1PuppiCands_pdgId'].__getitem__)(pdgid.astype(float))
X[:, :, 7] = np.vectorize(d_encoding['L1PuppiCands_charge'].__getitem__)(charge.astype(float))

# truth info
if not opt.data:
    Y[:, 0] += tree['genMet_pt'].to_numpy() * np.cos(tree['genMet_phi'].to_numpy())
    Y[:, 1] += tree['genMet_pt'].to_numpy() * np.sin(tree['genMet_phi'].to_numpy())

with h5py.File(opt.output, 'w') as h5f:
    h5f.create_dataset('X',    data=X,   compression='lzf')
    h5f.create_dataset('Y',    data=Y,   compression='lzf')
