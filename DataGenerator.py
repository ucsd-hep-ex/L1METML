import tensorflow
import tensorflow.keras as keras
import numpy as np
import uproot
import awkward as ak
from models import assign_matrices
from utils import convertXY2PtPhi, preProcessing, to_np_array
import h5py
import os
import itertools
import time


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
            h5file_path = ifile.replace('.root', '.h5')
            if not os.path.isfile(h5file_path):
                os.system(f'python convertNanoToHDF5_L1triggerToDeepMET.py -i {ifile} -o {h5file_path}')
            self.h5files.append(h5file_path)
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

    def deltaR_calc(self, eta1, phi1, eta2, phi2):
        """ calculate deltaR """
        dphi = (phi1-phi2)
        gt_pi_idx = (dphi > np.pi)
        lt_pi_idx = (dphi < -np.pi)
        dphi[gt_pi_idx] -= 2*np.pi
        dphi[lt_pi_idx] += 2*np.pi
        deta = eta1-eta2
        return np.hypot(deta, dphi)

    def kT_calc(self, pti, ptj, dR):
        min_pt = np.minimum(pti[:, 0:1], ptj[:, 0:1])
        kT = min_pt * dR
        return kT

    def z_calc(self, pti, ptj):
        epsilon = 1.0e-12
        min_pt = np.minimum(pti[:, 0:1], ptj[:, 0:1])
        z = min_pt/(pti + ptj + epsilon)
        return z

    def mass2_calc(self, pi, pj):
        pij = pi + pj
        m2 = pij[:, 0:1]**2 - pij[:, 1:2]**2 - pij[:, 2:3]**2 - pij[:, 3:4]**2
        return m2

    def assign_matrices(N, Nr):
        Rs = np.zeros([N, Nr], dtype=np.float32)
        Rr = np.zeros([N, Nr], dtype=np.float32)
        receiver_sender_list = [i for i in itertools.product(range(N), range(N)) if i[0] != i[1]]
        for i, (r, s) in enumerate(receiver_sender_list):
            Rr[r, i] = 1
            Rs[s, i] = 1
        return Rs, Rr

    def MakeEdgeHist(edge_feat, xname, outputname, nbins=1000, density=False, yname="# of edges"):
        plt.style.use(hep.style.CMS)
        plt.figure(figsize=(10, 8))
        plt.hist(edge_feat, bins=nbins, density=density, histtype='step', facecolor='k', label='Truth')
        plt.xlabel(xname)
        plt.ylabel(yname)
        plt.savefig(outputname)
        plt.close()

    def __data_generation(self, unique_files, starts, stops):
        'Generates data containing batch_size samples'
        # X : (n_samples, n_dim, n_channels)
        # y : (n_samples, 2)
        Xs = []
        ys = []
        ys_gencands = []

        # Generate data
        for ifile, start, stop in zip(unique_files, starts, stops):
            self.X, self.y, self.y_gencands = self.__get_features_labels(ifile, start, stop)

            Xs.append(self.X)
            ys.append(self.y)
            ys_gencands.append(self.y_gencands)

        # Stack data if going over multiple files
        if len(unique_files) > 1:
            self.X = np.concatenate(Xs, axis=0)
            self.y = np.concatenate(ys, axis=0)
            self.y_gencands = np.concatenate(ys_gencands, axis=0)

        quad = True
        if quad==True:
            bins = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
            phi_mod = np.mod(self.X[:,:,4:5], 2*np.pi)    # ensures all phi values are within 0-2pi
            idx = np.digitize(phi_mod,bins)     # Bins phi column into 4 quadrants
            X_quad2 = np.where(idx==2,self.X,0)     # uses 'idx' as index to bin input data
            X_quad1 = np.where(idx==1,self.X,0)     # binned data preserves shape, self.X = [256, 100, 8] -> X_quad = [256, 100, 8], padded with zeros
            X_quad3 = np.where(idx==3,self.X,0)
            X_quad4 = np.where(idx==4,self.X,0)
            self.X = np.concatenate((X_quad1,X_quad2,X_quad3,X_quad4),axis=0)    # concatenates all quadrants together sefl.X=[1024,100,8]


            order = self.X[:, :, 0].argsort(axis=1)[:, ::-1]    # Order self.X by pt so when we truncate, we take most important particles
            order = order[:, :, np.newaxis]
            order = np.repeat(order, repeats=8, axis=2)    # repeats=8 for the 8 node features. Creates index to sort array
            self.X = np.take_along_axis(self.X, order, axis=1)    # sorts self.X from greatest to least using 'order' variable as index
            self.X = self.X[:, 0:25, :]    # truncates events to only keep 25 particles per event per quadrant


        # process inputs
        Y = self.y / (-self.normFac)
        Xi, Xp, Xc1, Xc2 = preProcessing(self.X, self.normFac)
        Y_gencand = self.y_gencands / (self.normFac)

        N = int(self.maxNPF/4)
        Nr = N*(N-1)

        if quad==True:
            bins = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]    # for truth tensor, we bin using same process as self.X
            pt_phi = Y_gencand[:, :, [0,2]]
            y_phi_mod = np.mod(pt_phi[:,:,1:2], 2*np.pi)
            idx = np.digitize(y_phi_mod,bins)
            Y_quad1 = np.where(idx==1,pt_phi,0)
            Y_quad2 = np.where(idx==2,pt_phi,0)
            Y_quad3 = np.where(idx==3,pt_phi,0)
            Y_quad4 = np.where(idx==4,pt_phi,0)

            Y_1px = Y_quad1[:,:,0] * np.cos(Y_quad1[:,:,1])    # px = pt * cos(phi)
            Y_1py = Y_quad1[:,:,0] * np.sin(Y_quad1[:,:,1])    # py = pt * cos(phi)
            Y_1px = np.sum(Y_1px, axis=1)    # sum px of all particles to get total px
            Y_1py = np.sum(Y_1py, axis=1)    # sum py of all particles ot get total py

            Y_2px = Y_quad2[:,:,0] * np.cos(Y_quad2[:,:,1])
            Y_2py = Y_quad2[:,:,0] * np.sin(Y_quad2[:,:,1])
            Y_2px = np.sum(Y_2px, axis=1)
            Y_2py = np.sum(Y_2py, axis=1)

            Y_3px = Y_quad3[:,:,0] * np.cos(Y_quad3[:,:,1])
            Y_3py = Y_quad3[:,:,0] * np.sin(Y_quad3[:,:,1])
            Y_3px = np.sum(Y_3px, axis=1)
            Y_3py = np.sum(Y_3py, axis=1)

            Y_4px = Y_quad4[:,:,0] * np.cos(Y_quad4[:,:,1])
            Y_4py = Y_quad4[:,:,0] * np.sin(Y_quad4[:,:,1])
            Y_4px = np.sum(Y_4px, axis=1)
            Y_4py = np.sum(Y_4py, axis=1)

            #sum px distribution in quadrant 1 for gencand
            # compare to reco px
            # do they look like each other? if not are we matching them up wrong

            # add sorting for default (311, utils.py)
            
            Yx = np.concatenate((Y_1px,Y_2px,Y_3px,Y_4px))    # concatenate px of all quadrants together
            Yy = np.concatenate((Y_1py,Y_2py,Y_3py,Y_4py))    # concatenate py of all quadrants together
            Y = np.stack((Yx,Yy),axis=1)    # stack px, and py to get truth tensor with shape (1024, 2)



        if self.compute_ef == 1:
            eta = Xi[:, :, 1:2]
            phi = Xi[:, :, 2:3]
            pt = Xi[:, :, 0:1]
            if ('m2' in self.edge_list):
                px = Xp[:, :, 0:1]
                py = Xp[:, :, 1:2]
                pz = pt*np.sinh(eta)
                energy = np.sqrt(px**2 + py**2 + pz**2)
                p4 = np.concatenate((energy, px, py, pz), axis=-1)
            receiver_sender_list = [i for i in itertools.product(range(N), range(N)) if i[0] != i[1]]
            set_size = Xi.shape[0]
            ef = np.zeros([set_size, Nr, len(self.edge_list)])     # edge features: dimensions of [# of events, # of edges, # of edges]
            for count, edge in enumerate(receiver_sender_list):       # for loop creates edge features
                receiver = edge[0]  # "receiver_sender_list" generates edge and receiving indices
                sender = edge[1]
                if ('dR' in self.edge_list) or ('kT' in self.edge_list):
                    eta1 = eta[:, receiver, :]
                    phi1 = phi[:, receiver, :]
                    eta2 = eta[:, sender, :]
                    phi2 = phi[:, sender, :]
                    dR = self.deltaR_calc(eta1, phi1, eta2, phi2)
                    ef[:, count, 0:1] = dR
                if ('kT' in self.edge_list) or ('z' in self.edge_list):
                    pt1 = pt[:, receiver, :]
                    pt2 = pt[:, sender, :]
                    if ('kT' in self.edge_list):
                        kT = self.kT_calc(pt1, pt2, dR)
                        ef[:, count, 1:2] = kT
                    if ('z' in self.edge_list):
                        z = self.z_calc(pt1, pt2)
                        ef[:, count, 2:3] = z
                if ('m2' in self.edge_list):
                    p1 = p4[:, receiver, :]
                    p2 = p4[:, sender, :]
                    m2 = self.mass2_calc(p1, p2)
                    ef[:, count, 3:4] = m2

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
            #print('emb_input_dim:  ', self.emb_input_dim)
            assert not np.any(np.isnan(Xc1))
            assert not np.any(np.isnan(Xc2))
            assert not np.any(np.isnan(Xi))
            assert not np.any(np.isnan(Xp))
            assert not np.any(np.isnan(ef))
            # Prepare training/val data
            Yr = Y
            Xr = [Xi, Xp] + Xc + [ef]
            print(np.shape(Xi))
            print(np.shape(Xp))
            print(np.shape(Xc1))
            print(np.shape(Xc2))
            print(np.shape(ef))
            print(np.shape(Yr))
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
        y_gencands = h5_file['Y_GenCand'][entry_start:entry_stop+1]


        if self.maxNPF < 100:
            order = X[:, :, 0].argsort(axis=1)[:, ::-1]    # gets axis 1 index from greatest to least
            order = order[:, :, np.newaxis]
            order = np.repeat(order, repeats=8, axis=2)  # repeats=8 for the 8 node features. Creates index to sort array
            X = np.take_along_axis(X, order, axis=1)     # sorts data from greatest to least using 'order' variable as index
            X = X[:, 0:self.maxNPF, :]

        return X, y, y_gencands
