# Author: Noel Dawe <noel.dawe@gmail.com>
#
# License: BSD 3 clause

# importing necessary libraries
import numpy as np
import tables
import matplotlib.pyplot as plt
import math
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from Write_MET_binned_histogram_forBDT import *

def get_features_targets(file_name, features, targets):
    # load file
    h5file = tables.open_file(file_name, 'r')
    nevents = getattr(h5file.root,features[0]).shape[0]
    ntargets = len(targets)
    nfeatures = len(features)

    # allocate arrays
    feature_array = np.zeros((nevents, nfeatures))
    target_array = np.zeros((nevents, ntargets))

    '''
    # load feature arrays
    for j in range(nevents):
            feature_array[j] = getattr(h5file.root,'L1PuppiMet_pt')[j]
    # load target arrays
    for j in range(nevents):
        target_array[j] = getattr(h5file.root,'genMet_pt')[j]

    h5file.close()
    return feature_array,target_array
    '''
    # load feature arrays
    for (i, feat) in enumerate(features):
        feature_array[:,i] = getattr(h5file.root,feat)[:]
    # load target arrays
    for (i, targ) in enumerate(targets):
        target_array[:,i] = getattr(h5file.root,targ)[:]

    h5file.close()
    return feature_array,target_array



file_path = 'input_MET.h5'
features =  ['L1CHSMet_pt', 'L1CHSMet_phi',
            'L1CaloMet_pt', 'L1CaloMet_phi',
            'L1PFMet_pt', 'L1PFMet_phi',
            'L1PuppiMet_pt', 'L1PuppiMet_phi',
            'L1TKMet_pt', 'L1TKMet_phi',
            'L1TKV5Met_pt', 'L1TKV5Met_phi',
            'L1TKV6Met_pt', 'L1TKV6Met_phi']

targets = ['genMet_pt', 'genMet_phi']
feature, target = get_features_targets(file_path, features, targets)
nevents = feature.shape[0]
nfeatures = feature.shape[1]
ntargets = target.shape[1]


# Exclude Gen met < 100 GeV events
event_zero = 0
skip = 0
for i in range(nevents):
    if (target[i,0] < 100):
        event_zero = event_zero + 1

feature_array_without0 = np.zeros((nevents - event_zero, nfeatures))
target_array_without0 = np.zeros((nevents - event_zero, 2))

for i in range(nevents):
    if (target[i, 0] < 100):
        skip = skip + 1
        continue
    feature_array_without0[i - skip, :] = feature[i, :]
    target_array_without0[i - skip, :] = target[i, :]

nevents = feature_array_without0.shape[0]
feature = np.zeros((nevents, nfeatures)) 
target = np.zeros((nevents, 2)) 
feature = feature_array_without0
target = target_array_without0


# Convert feature from pt, phi to px, py
feature_array_xy = np.zeros((nevents, nfeatures))
for i in range(nfeatures):
    if i%2 == 0:
        for j in range(nevents):
            feature_array_xy[j,i] = feature[j,i] * math.cos(feature[j,i+1])
    if i%2 == 1:
        for j in range(nevents):
            feature_array_xy[j,i] = feature[j,i-1] * math.sin(feature[j,i])

feature = np.zeros((nevents, nfeatures))
feature = feature_array_xy

# Convert target from pt phi to px, py
target_array_x = np.zeros(nevents)

for i in range(nevents):
    target_array_x[i] = target[i,0] * math.cos(target[i,1])

target_array_y = np.zeros(nevents)

for i in range(nevents):
    target_array_y[i] = target[i,0] * math.sin(target[i,1])

target_x = np.zeros(nevents)
target_x = target_array_x
target_y = np.zeros(nevents)
target_y = target_array_y


fulllen = nevents
tv_frac = 0.10
tv_num = math.ceil(fulllen*tv_frac)
splits = np.cumsum([fulllen-2*tv_num,tv_num,tv_num])
splits = [int(s) for s in splits]

feature_array = feature[0:splits[0]]
feat_test = feature[splits[0]:splits[1]]

target_x_array = target_x[0:splits[0]]
targ_x_test = target_x[splits[0]:splits[1]]

target_y_array = target_y[0:splits[0]]
targ_y_test = target_y[splits[0]:splits[1]]

nevents = feature_array.shape[0]



# Fit regression model
regr_x = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=300)

regr_x.fit(feature_array, target_x_array)

regr_y = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=300)

regr_y.fit(feature_array, target_y_array)

# Predict
predict_x = regr_x.predict(feat_test)

predict_y = regr_y.predict(feat_test)

nevents = feat_test.shape[0]
gen_array = np.zeros((nevents, 2))
predict_phi = np.zeros((nevents, 2))

for i in range(nevents):
    predict_phi[i,0] = math.sqrt((predict_x[i]**2 + predict_y[i]**2))
    if 0 < predict_y[i]:
        predict_phi[i,1] = math.acos(predict_x[i]/predict_phi[i,0])
    if predict_y[i] < 0:
        predict_phi[i,1] = -math.acos(predict_x[i]/predict_phi[i,0])

for i in range(nevents):
    gen_array[i,0] = math.sqrt((targ_x_test[i]**2 + targ_y_test[i]**2))
    if 0 < targ_y_test[i]:
        gen_array[i,1] = math.acos(targ_x_test[i]/gen_array[i,0])
    if targ_y_test[i] < 0:
        gen_array[i,1] = -math.acos(targ_x_test[i]/gen_array[i,0])


'''
# Plot the results
plt.figure()
plt.scatter(X, y, c="k", label="training samples")
plt.plot(X, y_1, c="g", label="n_estimators=1", linewidth=2)
plt.plot(X, y_2, c="r", label="n_estimators=300", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Boosted Decision Tree Regression")
plt.legend()
plt.show()

#rel_err = (predict_phi[:,0] - gen_array[:,0])
rel_err = (predict_phi[:,0] - gen_array[:,0])/gen_array[:,0]
plt.figure()
#plt.hist(rel_err, bins=np.linspace(-500., 500., 50+1))
plt.hist(rel_err, bins=np.linspace(-3., 3., 50+1))
#plt.xlabel("abs error (predict - true)")
plt.xlabel("rel error (predict - true)/true")
plt.ylabel("Events")
plt.figtext(0.25, 0.90, 'CMS', fontweight='bold', wrap=True, horizontalalignment='right', fontsize=14)
plt.figtext(0.35, 0.90, 'preliminary', style='italic', wrap=True, horizontalalignment='center', fontsize=14)
#plt.savefig('MET_abs_error_BDT.pdf')
plt.savefig('MET_rel_error_BDT.pdf')
'''


#MET_rel_error(predict_phi[:,0], gen_array[:,0], name='MET_rel_error_all_100cut_BDT_1_nestimator.pdf')
#Phi_abs_error(predict_phi[:,1], gen_array[:,1], name='Phi_abs_error_all_100cut_BDT_1_nestimator.pdf')
Write_MET_binned_histogram(predict_phi, gen_array, 20, 0, 100, 400, name='./rootfiles/(VBF)histogram_all_100cut_BDT_300_nestimator.root')
