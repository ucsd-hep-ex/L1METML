# L1METML

### Setup
```
cmsrel CMSSW_10_6_1_patch2
cd CMSSW_10_6_1_patch2/src
cmsenv
git clone -b DeepMET_merge git@github.com:jmduarte/L1METML.git
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
python convertNanoToHDF5_L1triggerToDeepMET.py -i [input .root file path] -o [output file path]
```

### Train
```
python train.py --input [.txt file with input files list] --output [output path (plot and weight will be stored)] --mode [0 or 1 (0 for L1MET model, 1 for DeepMET model)]
```
For example,
```
python train.py --input ./preprocessed/input.txt --output ./result/ --mode 0
```
