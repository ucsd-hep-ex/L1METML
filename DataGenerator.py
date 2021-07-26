import tensorflow
import tensorflow.keras as keras
import numpy as np
import uproot
import awkward as ak
from utils import convertXY2PtPhi, preProcessing, to_np_array
from sklearn.model_selection import train_test_split

class DataGenerator(tensorflow.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_files, batch_size=1024, n_dim=100,
                 max_entry = 100000000):
        'Initialization'
        self.n_features_pf = 6
        self.n_features_pf_cat = 2
        self.normFac = 1.
        self.batch_size = batch_size
        self.list_files = list_files
        self.n_dim = n_dim
        self.n_channels = 8
        self.features = ['nL1PuppiCands', 'L1PuppiCands_pt','L1PuppiCands_eta','L1PuppiCands_phi',
                         'L1PuppiCands_charge','L1PuppiCands_pdgId','L1PuppiCands_puppiWeight']
        self.labels = ['genMet_pt', 'genMet_phi']
        self.d_encoding = {
            'L1PuppiCands_charge':{-999.0: 0,
                           -1.0: 1,
                           0.0: 2,
                           1.0: 3},
            'L1PuppiCands_pdgId':{-999.0: 0,
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
        self.global_IDs = []
        self.local_IDs = []
        self.file_mapping = []
        self.max_entry = max_entry
        self.open_files = [None]*len(self.list_files)
        running_total = 0
        for i, file_name in enumerate(self.list_files):
            root_file = uproot.open(file_name)
            self.open_files.append(root_file)
            tree = root_file['Events']
            tree_length = min(tree.num_entries,self.max_entry)
            self.global_IDs.append(np.arange(running_total,running_total+tree_length))
            self.local_IDs.append(np.arange(0,tree_length))
            self.file_mapping.append(np.repeat([i],tree_length))
            running_total += tree_length
            root_file.close()
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
        starts = np.array([min(indexes[files==i]) for i in unique_files])
        stops = np.array([max(indexes[files==i]) for i in unique_files])

        # Check if files needed open (if not open them)
        # Also if file is not needed, close it
        for ifile, file_name in enumerate(self.list_files):
            if ifile in unique_files:
                if self.open_files[ifile] is None: 
                    self.open_files[ifile] = uproot.open(file_name)
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
        
        # Generate data
        for ifile, start, stop in zip(unique_files, starts, stops):
            self.X, self.y = self.__get_features_labels(ifile, start, stop)
            Xs.append(self.X)
            ys.append(self.y)
            
        # Stack data if going over multiple files
        if len(unique_files)>1:
            self.X = np.concatenate(Xs,axis=0)
            self.y = np.concatenate(ys,axis=0)
	
        #process inputs
        Y = self.y /(-self.normFac)
        Xi, Xc1, Xc2 = preProcessing_root(self.X, self.normFac)
        
        Xc = [Xc1, Xc2]
	# dimension parameter for keras model
        self.emb_input_dim = {i:int(np.max(Xc[i][0:1000])) + 1 for i in range(self.n_features_pf_cat)}

    	# Prepare training/val data
        Yr = Y
        Xr = [Xi] + Xc
        
        return Xr, Yr
   
    def __get_features_labels(self, ifile, entry_start, entry_stop):
        'Loads data from one file'
        
        # Double check that file is open
        if self.open_files[ifile] is None:
            root_file = uproot.open(self.list_file[ifile])
        else:
            root_file = self.open_files[ifile]
            
        tree = root_file['Events'].arrays(self.features+self.labels,
                                          entry_start=entry_start,
                                          entry_stop=entry_stop+1)
        
        n_samples = len(tree[self.labels[0]])

        X = np.zeros(shape=(n_samples,self.n_dim,self.n_channels), dtype=float, order='F')
        y = np.zeros(shape=(n_samples,2), dtype=float, order='F')
        
        pt = to_np_array(tree['L1PuppiCands_pt'],maxN=self.n_dim)
        eta = to_np_array(tree['L1PuppiCands_eta'],maxN=self.n_dim)
        phi = to_np_array(tree['L1PuppiCands_phi'],maxN=self.n_dim)
        pdgid = to_np_array(tree['L1PuppiCands_pdgId'],maxN=self.n_dim,pad=-999)
        charge = to_np_array(tree['L1PuppiCands_charge'],maxN=self.n_dim,pad=-999)
        puppiw = to_np_array(tree['L1PuppiCands_puppiWeight'],maxN=self.n_dim)
        
        #X[:,:,0] = pt
        X[:,:,0] = pt * np.cos(phi)
        X[:,:,1] = pt * np.sin(phi)
        X[:,:,2] = eta
        #X[:,:,4] = phi
        X[:,:,3] = puppiw

        # encoding
        X[:,:,4] = np.vectorize(self.d_encoding['L1PuppiCands_pdgId'].__getitem__)(pdgid.astype(float))
        X[:,:,5] = np.vectorize(self.d_encoding['L1PuppiCands_charge'].__getitem__)(charge.astype(float))
    
        # truth data
        y[:,0] += tree['genMet_pt'].to_numpy() * np.cos(tree['genMet_phi'].to_numpy())
        y[:,1] += tree['genMet_pt'].to_numpy() * np.sin(tree['genMet_phi'].to_numpy())

        return X, y

