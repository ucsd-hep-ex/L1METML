
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
                 max_entry=100000000):
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
        running_total = 0

        self.h5files = []
        for ifile in list_files:
            h5file_path = ifile.replace('.root', '.h5')
            if not os.path.isfile(h5file_path):
                os.system(f'python convertNanoToHDF5_L1triggerToDeepMET.py -i {ifile} -o {h5file_path}')
            self.h5files.append(h5file_path)
        for i, file_name in enumerate(self.h5files):
            with h5py.File(file_name, "r") as h5_file:
                self.open_files.append(h5_file)
                h5_file['X'] = h5_file['X'][:,0:50]
                h5_file['Y'] = h5_file['Y'][:,0:50]
                nEntries = len(h5_file['X'])
                print('nEntries:  ', nEntries)
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
        
        print('indexes:  ', indexes)
        print('files:  ', files)

        unique_files = np.unique(files)
        starts = np.array([min(indexes[files == i]) for i in unique_files])
        stops = np.array([max(indexes[files == i]) for i in unique_files])
        print('starts:  ', starts, np.shape(starts))
        print('stops:  ', stops, np.shape(stops))

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

    def deltaR(self, eta1, phi1, eta2, phi2):
        """ calculate deltaR """
        dphi = (phi1-phi2)
        gt_pi_idx = (dphi > np.pi)
        lt_pi_idx = (dphi < -np.pi)
        dphi[gt_pi_idx] -= 2*np.pi
        dphi[lt_pi_idx] += 2*np.pi
        deta = eta1-eta2
        return np.hypot(deta, dphi)
    
    def kT(self,pti,ptj,dR):
        min_pt = np.minimum(pti[:,0:1],ptj[:,0:1])
        kT = min_pt * dR
        #kT = np.log10(kT)
        #kT[np.isneginf(kT)] = 0
        return kT

    def z(self, pti, ptj):
        epsilon = 1.0e-12
        min_pt = np.minimum(pti[:,0:1],ptj[:,0:1])
        z = min_pt/(pti + ptj + epsilon)
        #z[np.isnan(z)] = 0
        #z[np.isinf(z)] = 0
        #z[z==0] = 1
        #z = np.log(z) / 5
        return z
    
    def m2(self, pi, pj):
        m2 = np.linalg.norm((pi+pj),axis=-1,keepdims=True) ** 2
        return m2
    
    def __data_generation(self, unique_files, starts, stops):
        'Generates data containing batch_size samples'
        # X : (n_samples, n_dim, n_channels)
        # y : (n_samples, 2)
        Xs = []
        ys = []

        # Generate data
        for ifile, start, stop in zip(unique_files, starts, stops):
            self.X, self.y = self.__get_features_labels(ifile, start, stop)
            #print('X.shape:  ', len(self.X))
            Xs.append(self.X)
            print('X:  ', self.X, np.shape(self.X))
            ys.append(self.y)
        print('stop:  ', stop)

        # Stack data if going over multiple files
        if len(unique_files) > 1:
            self.X = np.concatenate(Xs, axis=0)
            self.y = np.concatenate(ys, axis=0)

        # process inputs
        Y = self.y / (-self.normFac)
        Xi, Xp, Xc1, Xc2 = preProcessing(self.X, self.normFac)
        
        N = self.maxNPF
        Nr = N*(N-1)

        if self.compute_ef == 1:
            eta = Xi[:,:,1:2]
            phi = Xi[:,:,2:3]
            pt = Xi[:,:,0:1]
            #px = Xp[:,:,0:1]
            #py = Xp[:,:,1:2]
            #pz = pt*np.sinh(eta)
            #p_vector = np.concatenate((px,py,pz), axis=-1)
            receiver_sender_list = [i for i in itertools.product(range(N), range(N)) if i[0] != i[1]]
            set_size = Xi.shape[0]
            ef = np.zeros([set_size, Nr, 1])
            for count, edge in enumerate(receiver_sender_list):
                receiver = edge[0]
                sender = edge[1]
                eta1 = eta[:, receiver, :]
                phi1 = phi[:, receiver, :]
                eta2 = eta[:, sender, :]
                phi2 = phi[:, sender, :]
                pt1 = pt[:, receiver, :]
                pt2 = pt[:, sender, :]
                #p1 = p_vector[:, receiver, :]
                #p2 = p_vector[:, sender, :]
                dR = self.deltaR(eta1, phi1, eta2, phi2)
                #m2 = self.m2(p1,p2)
                #kT = self.kT(pt1,pt2,dR)
                #z = self.z(pt1,pt2)
                ef[:,count,0:1] = dR
                #ef[:,count,1:2] = m2
                #ef[:,count,1:2] = kT
                #ef[:,count,2:3] = z
                
                '''print('dR shape')
                print(dR.shape)
                print('-----')
                print('kT shape')
                print(kT.shape)
                print('-----')
                print('ef shape')
                print(ef.shape)
                print('-----')
                print('Xi shape')
                print(Xi.shape)
                ef[:,count,2:3] = z'''

            Xc = [Xc1, Xc2]
            # dimension parameter for keras model
            self.emb_input_dim = {i: int(np.max(Xc[i][0:1000])) + 1 for i in range(self.n_features_pf_cat)}
            print('emb_input_dim:  ', self.emb_input_dim)

            # Prepare training/val data
            Yr = Y
            Xr = [Xi, Xp] + Xc + [ef]
            return Xr, Yr
        
        else:
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

        return X, y
