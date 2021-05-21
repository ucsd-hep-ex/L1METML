#!/usr/bin/env python

import sys
import uproot
import numpy as np
import h5py
#import progressbar
from tqdm import tqdm
import os

'''
widgets=[
    progressbar.SimpleProgress(), ' - ', progressbar.Timer(), ' - ', progressbar.Bar(), ' - ', progressbar.AbsoluteETA()
]
'''

def deltaR(eta1, phi1, eta2, phi2):
    """ calculate deltaR """
    dphi = (phi1-phi2)
    while dphi >  np.pi: dphi -= 2*np.pi
    while dphi < -np.pi: dphi += 2*np.pi
    deta = eta1-eta2
    return np.hypot(deta, dphi)


import optparse

#configuration
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
	'nL1PuppiCands', 'L1PuppiCands_pt','L1PuppiCands_eta','L1PuppiCands_phi',
	'L1PuppiCands_charge','L1PuppiCands_pdgId','L1PuppiCands_puppiWeight'
]

# event-level variables

varList_mc = [
    'genMet_pt', 'genMet_phi',
]

d_encoding = {
    'L1PuppiCands_charge':{-1.0: 0, 0.0: 1, 1.0: 2},
    'L1PuppiCands_pdgId':{-211.0: 0, -130.0: 1, -22.0: 2, -13.0: 3, -11.0: 4, 0.0: 5, 1.0: 6, 2.0: 7, 11.0: 8, 13.0: 9, 22.0: 10, 130.0: 11, 211.0: 12}#,
}

if not opt.data:
    varList = varList + varList_mc
    
upfile = uproot.open(opt.input)
tree = upfile['Events'].arrays( varList, entry_stop =opt.maxevents)
# general setup
maxNPuppi = 100
nFeatures = 8
maxEntries = len(tree['nL1PuppiCands']) 
# input Puppi candidates
X = np.zeros(shape=(maxEntries,maxNPuppi,nFeatures), dtype=float, order='F')
# recoil estimators
Y = np.zeros(shape=(maxEntries,2), dtype=float, order='F')

print(X.shape)

# loop over events
#for e in progressbar.progressbar(range(maxEntries), widgets=widgets):
for e in tqdm(range(maxEntries)):

    # get momenta
    ipuppi = 0
    ilep = 0
    for j in range(tree['nL1PuppiCands'][e]):
        if ipuppi == maxNPuppi:
            break

        pt = tree['L1PuppiCands_pt'][e][j]
        #if pt < 0.5:
        #    continue
        eta = tree['L1PuppiCands_eta'][e][j]
        phi = tree['L1PuppiCands_phi'][e][j]
       
        puppi = X[e][ipuppi]

        isLep = False
        if not isLep:
            ipuppi += 1
                 
        # 4-momentum
        puppi[0] = pt
        puppi[1] = pt * np.cos(phi)
        puppi[2] = pt * np.sin(phi)
        puppi[3] = eta
        puppi[4] = phi
        puppi[5] = tree['L1PuppiCands_puppiWeight'][e][j]
        # encoding
        puppi[6] = d_encoding['L1PuppiCands_pdgId' ][float(tree['L1PuppiCands_pdgId' ][e][j])]
        puppi[7] = d_encoding['L1PuppiCands_charge'][float(tree['L1PuppiCands_charge'][e][j])]

    # truth info
    if not opt.data:
        Y[e][0] += tree['genMet_pt'][e] * np.cos(tree['genMet_phi'][e])
        Y[e][1] += tree['genMet_pt'][e] * np.sin(tree['genMet_phi'][e])


with h5py.File(opt.output, 'w') as h5f:
    h5f.create_dataset('X',    data=X,   compression='lzf')
    h5f.create_dataset('Y',    data=Y,   compression='lzf')
