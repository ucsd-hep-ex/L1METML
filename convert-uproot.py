import uproot
import pandas
import numpy as np
import pandas as pd
import h5py
import tables
import sys
filters = tables.Filters(complevel=7, complib='blosc')

path = '/afs/cern.ch/work/d/daekwon/public/l1samples/'

infiles = [path+'perfNano_SingleNeutrino_PU200.v0.root',
           path+'perfNano_TTbar_PU200.v0.root',
           path+'perfNano_VBF_HToInvisible_PU200.v0.root',
]

outfile = 'input_MET.h5'
#entrystop = 10000
entrystop = None

other_branches = ['run', 'luminosityBlock', 'event', 'nL1PuppiJets']

met_branches = [# barrel
                'L1CHSMetBarrel_pt', 'L1CHSMetBarrel_phi',
                'L1CaloMetBarrel_pt', 'L1CaloMetBarrel_phi',
                'L1PFMetBarrel_pt', 'L1PFMetBarrel_phi',
                'L1PuppiMetBarrel_pt', 'L1PuppiMetBarrel_phi',
                'L1TKMetBarrel_pt', 'L1TKMetBarrel_phi',
                'L1TKV5MetBarrel_pt', 'L1TKV5MetBarrel_phi',
                'L1TKV6MetBarrel_pt', 'L1TKV6MetBarrel_phi',
                'L1TKVMetBarrel_pt', 'L1TKVMetBarrel_phi',
                'genMetBarrel_pt', 'genMetBarrel_phi',
                # central
                'L1CHSMetCentral_pt', 'L1CHSMetCentral_phi',
                'L1CaloMetCentral_pt', 'L1CaloMetCentral_phi',
                'L1PFMetCentral_pt', 'L1PFMetCentral_phi',
                'L1PuppiMetCentral_pt', 'L1PuppiMetCentral_phi',
                'L1TKMetCentral_pt', 'L1TKMetCentral_phi',
                'L1TKV5MetCentral_pt', 'L1TKV5MetCentral_phi',
                'L1TKV6MetCentral_pt', 'L1TKV6MetCentral_phi',
                'L1TKVMetCentral_pt', 'L1TKVMetCentral_phi',
                'genMetCentral_pt', 'genMetCentral_phi',
                # total
                'L1CHSMet_pt', 'L1CHSMet_phi',
                'L1CaloMet_pt', 'L1CaloMet_phi',
                'L1PFMet_pt', 'L1PFMet_phi',
                'L1PuppiMet_pt', 'L1PuppiMet_phi',
                'L1TKMet_pt', 'L1TKMet_phi',
                'L1TKV5Met_pt', 'L1TKV5Met_phi',
                'L1TKV6Met_pt', 'L1TKV6Met_phi',
                'L1TKVMet_pt', 'L1TKVMet_phi',
                'genMet_pt', 'genMet_phi'
]


jet_branches =['L1PuppiJets_pt','L1PuppiJets_eta','L1PuppiJets_phi', 
               'L1PuppiJets_mass']

print("Reading branches", other_branches)
print("Reading branches:", met_branches)
print("Reading branches:", jet_branches)

def _write_carray(a, h5file, name, group_path='/', **kwargs):
    h5file.create_carray(group_path, name, obj=a, filters=filters, createparents=True, **kwargs)
    
def _transform(dataframe, max_particles=100, start=0, stop=-1):    
    return dataframe[dataframe.index.get_level_values(-1)<max_particles].unstack().fillna(0)

df_others = []
df_mets = []
df_jets = []

currententry = 0
for infile in infiles:
    print('Opening file: %s'%infile)

    upfile = uproot.open(infile)
    tree = upfile['Events']
    
    df_other = tree.pandas.df(branches=other_branches, entrystart=0, entrystop = entrystop)
    df_met = tree.pandas.df(branches=met_branches, entrystart=0, entrystop = entrystop)
    df_jet = tree.pandas.df(branches=jet_branches, entrystart=0, entrystop = entrystop)

    df_other.index = df_other.index+currententry
    df_met.index = df_met.index+currententry
    df_jet.index = df_jet.index.set_levels(df_jet.index.levels[0]+currententry, level=0)
    currententry += len(df_other)

    df_others.append(df_other)
    df_mets.append(df_met)
    df_jets.append(df_jet)
    
df_other = pd.concat(df_others)
df_met = pd.concat(df_mets)
df_jet = pd.concat(df_jets)

# shuffle
df_other = df_other.sample(frac=1)
# apply new ordering to other dataframes
df_met = df_met.reindex(df_other.index.values)
df_jet = df_jet.reindex(df_other.index.values,level=0)

with tables.open_file(outfile, mode='w') as h5file:
    
    #max_jet = len(df_jet.index.get_level_values(-1).unique())
    # cap at 20 jets
    max_jet = 20
    print("Max number of jets",max_jet)
    
    v_jet = _transform(df_jet, max_particles = max_jet)
    for k in jet_branches:
        v = np.stack([v_jet[(k, i)].values for i in range(max_jet)], axis=-1)
        _write_carray(v, h5file, name=k)
        
    for k in df_other.columns:
        _write_carray(df_other[k].values, h5file, name=k.replace('[','').replace(']',''))

    for k in df_met.columns:
        _write_carray(df_met[k].values, h5file, name=k.replace('[','').replace(']',''))


f = tables.open_file(outfile)
print("Output file: %s"%outfile)
print(f)
f.close()
