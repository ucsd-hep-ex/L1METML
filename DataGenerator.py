import tensorflow
import tensorflow.keras as keras
import numpy as np
import uproot
import awkward as ak

class DataGenerator(tensorflow.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_files, batch_size=1024, n_dim=100, 
                 max_entry = 100000000):
        'Initialization'
        self.batch_size = batch_size
        self.list_files = list_files
        self.n_dim = n_dim
        self.n_channels = 8
        self.features = ['nL1PuppiCands', 'L1PuppiCands_pt','L1PuppiCands_eta','L1PuppiCands_phi',
                         'L1PuppiCands_charge','L1PuppiCands_pdgId','L1PuppiCands_puppiWeight']
        self.labels = ['genMet_pt', 'genMet_phi']
        self.d_encoding = {
            'L1PuppiCands_charge':{-999.0: 0,
                                   -1.0: 0, 
                                   0.0: 1, 
                                   1.0: 2},
            'L1PuppiCands_pdgId':{-999.0: 0,
                                  -211.0: 0, 
                                  -130.0: 1, 
                                  -22.0: 2, 
                                  -13.0: 3, 
                                  -11.0: 4,
                                  0.0: 5, 
                                  1.0: 6, 
                                  2.0: 7, 
                                  11.0: 8, 
                                  13.0: 9, 
                                  22.0: 10, 
                                  130.0: 11, 
                                  211.0: 12}
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
            tree_length = min(len(tree),max_entry)
            self.global_IDs.append(np.arange(running_total,running_total+tree_length))
            self.local_IDs.append(np.arange(tree_length))
            self.file_mapping.append(np.repeat(i,tree_length))
            running_total += tree_length
            root_file.close()
        self.global_IDs = np.concatenate(self.global_IDs)
        self.local_IDs = np.concatenate(self.local_IDs)
        self.file_mapping = np.concatenate(self.file_mapping)
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.global_IDs) / self.batch_size))

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
            X, y = self.__get_features_labels(ifile, start, stop)
            Xs.append(X)
            ys.append(y)
            
        # Stack data if going over multiple files
        if len(unique_files)>1:
            X = np.concatenate(Xs,axis=0)
            y = np.concatenate(ys,axis=0)
            
        return X, y
                         
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

        def to_np_array(ak_array, maxN=100, pad=0):
            return ak.fill_none(ak.pad_none(ak_array,maxN,clip=True,axis=-1),pad).to_numpy()
    
        pt = to_np_array(tree['L1PuppiCands_pt'],maxN=self.n_dim)
        eta = to_np_array(tree['L1PuppiCands_eta'],maxN=self.n_dim)
        phi = to_np_array(tree['L1PuppiCands_phi'],maxN=self.n_dim)
        pdgid = to_np_array(tree['L1PuppiCands_pdgId'],maxN=self.n_dim,pad=-999)
        charge = to_np_array(tree['L1PuppiCands_charge'],maxN=self.n_dim,pad=-999)
        puppiw = to_np_array(tree['L1PuppiCands_puppiWeight'],maxN=self.n_dim)
        
        X[:,:,0] = pt
        X[:,:,1] = pt * np.cos(phi)
        X[:,:,2] = pt * np.sin(phi)
        X[:,:,3] = eta
        X[:,:,4] = phi
        X[:,:,5] = puppiw

        # encoding
        X[:,:,6] = np.vectorize(self.d_encoding['L1PuppiCands_pdgId'].__getitem__)(pdgid.astype(float))
        X[:,:,7] = np.vectorize(self.d_encoding['L1PuppiCands_charge'].__getitem__)(charge.astype(float))
    
        # truth data
        y[:,0] += tree['genMet_pt'].to_numpy() * np.cos(tree['genMet_phi'].to_numpy())
        y[:,1] += tree['genMet_pt'].to_numpy() * np.sin(tree['genMet_phi'].to_numpy())

        return X, y

train_generator = DataGenerator(list_files=['data/perfNano_TTbar_PU200.110X_set0.root','data/perfNano_TTbar_PU200.110X_set1.root'],batch_size=128)
X, y = train_generator[0]
print(X.shape)
print(y.shape)