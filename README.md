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
```
python convert-uproot.py
```

### Train
```
python train.py
```
