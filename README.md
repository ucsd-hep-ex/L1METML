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

### Development Setup
For code quality and consistent formatting, install pre-commit hooks:
```bash
# Install pre-commit
pip install pre-commit

# Install the git hooks
pre-commit install

# Optionally, run on all files once
pre-commit run --all-files
```

This will automatically format your code and check for common issues before each commit.

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
python train.py --workflowType ['dataGenerator' or 'loadAllData': either use a data generator or load all data into memory]  --input [path to input files] --output [output path (plot and weight will be stored)] --mode [0 or 1 (0 for L1MET model, 1 for DeepMET model)] --epochs [int] --quantized [total bits] [int bits] --units [dense layer 1 units] [dense layer 2 units] [etc.]
```
For example,
```bash
python train.py --workflowType dataGenerator --input ./path/to/files/ --output ./path/to/result/ --mode 1 --epochs --quantized 8 2 --units 12 36
```

### Test
You need output results with input TTbar and SingleNeutrino.
When you use SingleNeutrino sample as input please change 'TTbar' to 'SingleNeutrino' in test function in train.py. #L66-67

SingleNeutrino sample is located in : https://cernbox.cern.ch/index.php/s/5inLVZpXreq1vOx

```bash
python rate_test.py --input [path to input files (output path of train.py)] --plot [ROC, rate, rate_com]
```
