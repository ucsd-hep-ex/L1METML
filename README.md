# L1METML

### Setup
```
cmsrel CMSSW_10_6_1_patch2
cd CMSSW_10_6_1_patch2/src
cmsenv
git clone https://github.com/jmduarte/L1METML
cd L1METML
```
Create an anaconda environment with Python 3.6 and install packages needed to run train.py.
```
sh conda_setup.sh
```

### Convert
The TTBar sample used in 'convert-uproot.py' is located in
```
/afs/cern.ch/work/y/yeseo/public/ml4MET/L1METML/data/perfNano_TTbar_PU200.110X_v1.root
```
Convert into hdf5
```
python convert-uproot.py
```

### Train
```
python train.py
```
