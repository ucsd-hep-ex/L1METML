
import tensorflow
import tensorflow.keras as keras
import numpy as np
import uproot
import awkward as ak
from utils import convertXY2PtPhi, preProcessing, to_np_array
import h5py
import os
import itertools


class DataGenerator(tensorflow.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_files, batch_size=1024, n_dim=100, maxNPF=100, compute_ef=0,
                 max_entry=100000000, edge_list=[]):
        'Initialization'
        self.n_features_pf = 6
        self.n_features_pf_cat = 2
        self.normFac = 1.
        self.batch_size = batch_size
        self.n_dim = n_dim
        self.n_channels = 8
        self.global_IDs = []
        self.local_IDs = []
        self.file_mapping = []
        self.max_entry = max_entry
        self.open_files = [None]*len(list_files)
        self.maxNPF = maxNPF
        self.compute_ef = compute_ef
        self.edge_list = edge_list
        running_total = 0

        self.h5files = []
        for ifile in list_files:
            if 'singleneutrino' in ifile.lower():
                h5file_path = ifile
            else:
                h5file_path = ifile.replace('.root', '.h5')
            if not os.path.isfile(h5file_path):
                if 'singleneutrino' in h5file_path.lower():
                    raise ValueError("No single neutrino file found")
                os.system(f'python convertNanoToHDF5_L1triggerToDeepMET.py -i {ifile} -o {h5file_path}')
            self.h5files.append(h5file_path)
        print(self.h5files)
        for i, file_name in enumerate(self.h5files):
            with h5py.File(file_name, "r") as h5_file:
                self.open_files.append(h5_file)
                nEntries = len(h5_file['X'])
                self.global_IDs.append(np.arange(running_total, running_total+nEntries))
                self.local_IDs.append(np.arange(0, nEntries))
                self.file_mapping.append(np.repeat([i], nEntries))
                running_total += nEntries
                h5_file.close()
        self.global_IDs = np.concatenate(self.global_IDs)
        self.local_IDs = np.concatenate(self.local_IDs)
        self.file_mapping = np.concatenate(self.file_mapping)
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.global_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        files = self.file_mapping[index*self.batch_size:(index+1)*self.batch_size]
        unique_files = np.unique(files)
        starts = np.array([min(indexes[files == i]) for i in unique_files])
        stops = np.array([max(indexes[files == i]) for i in unique_files])

        # Check if files needed open (if not open them)
        # Also if file is not needed, close it
        for ifile, file_name in enumerate(self.h5files):
            if ifile in unique_files:
                if self.open_files[ifile] is None:
                    self.open_files[ifile] = h5py.File(file_name, "r")
            else:
                if self.open_files[ifile] is not None:
                    self.open_files[ifile].close()
                    self.open_files[ifile] = None

        # Generate data
        return self.__data_generation(unique_files, starts, stops)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = self.local_IDs

    def __data_generation(self, unique_files, starts, stops):
        'Generates data containing batch_size samples'
        # X : (n_samples, n_dim, n_channels)
        # y : (n_samples, 2)
        Xs = []
        ys = []
        e = []

        # Generate data
        if self.compute_ef == 1:
            for ifile, start, stop in zip(unique_files, starts, stops):
                self.X, self.y, self.e = self.__get_features_labels(ifile, start, stop)
                Xs.append(self.X)
                ys.append(self.y)
                e.append(self.e)

            # Stack data if going over multiple files
            if len(unique_files) > 1:
                self.X = np.concatenate(Xs, axis=0)
                self.y = np.concatenate(ys, axis=0)
                self.e = np.concatenate(e, axis=0)

            # process inputs
            Y = self.y / (-self.normFac)
            Xi, Xp, Xc1, Xc2 = preProcessing(self.X, self.normFac)

            edge_dict = {'dR':0, 'kT':1, 'z':2, 'm2':3}
            ef_list = []
            for edge in self.edge_list:
                edge_idx = edge_dict[edge]
                ef_list.append(self.e[:,:,edge_idx])
            ef = np.stack(ef_list, axis=-1)

            Xc = [Xc1, Xc2]
            # dimension parameter for keras model
            self.emb_input_dim = {i: int(np.max(Xc[i][0:1000])) + 1 for i in range(self.n_features_pf_cat)}
            # Prepare training/val data
            Yr = Y
            Xr = [Xi, Xp] + Xc + [ef]
            return Xr, Yr

        else:
            for ifile, start, stop in zip(unique_files, starts, stops):
                self.X, self.y = self.__get_features_labels(ifile, start, stop)
                Xs.append(self.X)
                ys.append(self.y)

            if len(unique_files) > 1:
                self.X = np.concatenate(Xs, axis=0)
                self.y = np.concatenate(ys, axis=0)
    
            Y = self.y / (-self.normFac)
            Xi, Xp, Xc1, Xc2 = preProcessing(self.X, self.normFac)
            Xc = [Xc1, Xc2]
            # dimension parameter for keras model
            self.emb_input_dim = {i: int(np.max(Xc[i][0:1000])) + 1 for i in range(self.n_features_pf_cat)}

            # Prepare training/val data
            Yr = Y
            Xr = [Xi, Xp] + Xc
            return Xr, Yr

    def __get_features_labels(self, ifile, entry_start, entry_stop):
        'Loads data from one file'

        # Double check that file is open
        if self.open_files[ifile] is None:
            h5_file = h5py.File(file_name, "r")
        else:
            h5_file = self.open_files[ifile]

        X = h5_file['X'][entry_start:entry_stop+1]
        y = h5_file['Y'][entry_start:entry_stop+1]

        if self.maxNPF < 100:
            order = X[:, :, 0].argsort(axis=1)[:, ::-1]
            shape = np.shape(X)
            for x in range(shape[0]):
                X[x, :, :] = X[x, order[x], :]
            X = X[:, 0:self.maxNPF, :]

        if self.compute_ef == 1:
            ef_array = 'ef_' + str(self.maxNPF) + 'cand'
            ef = h5_file['ef_100cand'][entry_start:entry_stop+1]
            return X, y, ef

        else: 
            return X, y
