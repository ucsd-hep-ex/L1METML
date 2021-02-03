# L1METML

### Setup
```
cmsrel CMSSW_10_6_1_patch2
cd CMSSW_10_6_1_patch2/src
cmsenv
git clone -b jet_input git@github.com:jmduarte/L1METML.git
cd L1METML
```

#Caution : You need ROOT version over 6.22 to use pyROOT on python 3.

Create an anaconda environment with Python 3.6 and install packages needed to run train.py.
```
sh conda_setup.sh
```

### Convert
The TTBar sample used in 'convert-uproot.py' is located in
```
/afs/cern.ch/work/d/daekwon/public/L1PF_110X/CMSSW_11_1_2/src/FastPUPPI/NtupleProducer/python/TTbar_PU200_110X_1M/

```
Convert into hdf5
```
python convert-uproot.py
```

### Pre-processing
The preprocessing took so long that it was separated from the educational code.
```
python preprocessing.py --input [input .h5 file] --output [output path]
```
For example,
```
python preprocessing.py --input ./input.h5 --output ./preprocessed/
```

### Train
```
python train.py --input [input files path] --output [output path]
```
For example,
```
python train.py --input ./preprocessed/ --output ./result/
```
