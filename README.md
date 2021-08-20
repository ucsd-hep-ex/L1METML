# L1METML

### Setup
```bash
git clone git@github.com:jmduarte/L1METML.git
cd L1METML
```

Create an anaconda environment with Python 3.6 and install packages needed to run train.py.
```bash
bash conda_setup.sh
```

### Convert
The TTBar sample used in `convert-uproot.py` is located in
```
/afs/cern.ch/work/d/daekwon/public/L1PF_110X/CMSSW_11_1_2/src/FastPUPPI/NtupleProducer/python/TTbar_PU200_110X_1M/
```
or at this CERNBox link: https://cernbox.cern.ch/index.php/s/JK2InUjatHFxFbf

Convert into HDF5
```
python convertNanoToHDF5_L1triggerToDeepMET.py -i [input .root file path] -o [output file path]
```

### Train
```bash
python train.py --workflowType ['h5' or 'root': convert root to h5 files or use data generator on root files]  --input [path to input files] --output [output path (plot and weight will be stored)] --mode [0 or 1 (0 for L1MET model, 1 for DeepMET model)] --epochs [int] --quantized [total bits] [int bits] --units [dense layer 1 units] [dense layer 2 units] [etc.]
```
For example,
```bash
python train.py --workflowType --root --input ./path/to/files/ --output ./path/to/result/ --mode 1 --epochs --quantized 8 4 --units 16 32
```
