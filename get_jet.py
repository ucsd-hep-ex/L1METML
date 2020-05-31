import tables
import numpy as np


def get_features_targets(file_name, features, targets):
    # load file
    h5file = tables.open_file(file_name, 'r')
    nevents = getattr(h5file.root,features[0]).shape[0]
    ntargets = len(targets)
    nfeatures = len(features)
    print(nevents)

    # allocate arrays
    feature_array = np.zeros((nevents,nfeatures))
    target_array = np.zeros((nevents,ntargets))


    # load feature arrays
    for (i, feat) in enumerate(features):
        feature_array[:,i] = getattr(h5file.root,feat)[:]
    # load target arrays
    for (i, targ) in enumerate(targets):
        target_array[:,i] = getattr(h5file.root,targ)[:]

    h5file.close()
    return feature_array,target_array

def get_jet_features_targets(file_name, features, number_of_jets):
    # load file
    h5file = tables.open_file(file_name, 'r')
    nevents = getattr(h5file.root,features[0]).shape[0]
    nfeatures = len(features)
    print(nevents)

    # allocate arrays
    feature_array = np.zeros((nevents,nfeatures*number_of_jets))

    # load feature arrays
    for (i, feat) in enumerate(features):
        for j in range(number_of_jets):
            feature_array[:,(j*number_of_jets)+i] = getattr(h5file.root, feat)[:,j]
    h5file.close()
    return feature_array
