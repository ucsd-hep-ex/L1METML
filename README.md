# L1METML

### Setup
```
cmsrel CMSSW_10_6_1_patch2
cd CMSSW_10_6_1_patch2/src
cmsenv
git clone https://github.com/jmduarte/L1METML
cd L1METML
```
Create an anaconda environment with Python 3.5-3.7   
In order to run 'convert-uproot.py', you need to install [uproot](https://github.com/scikit-hep/uproot "uproot link").
```
conda config --add channels conda-forge
conda install uproot
```
And you need to install following packages.
These can be installed by "conda install [package name]"
* [numpy](https://scipy.org/install.html)
* [matplotlib](https://matplotlib.org/)
* [awkward-array](https://github.com/scikit-hep/awkward-array)
* [uproot-methods](https://github.com/scikit-hep/uproot-methods)
* [cachetools](https://pypi.org/project/cachetools/)
* [pandas](https://pandas.pydata.org/)
* [tables](https://pypi.org/project/tables/)
* [tensorflow](https://www.tensorflow.org/install)
* [keras](https://keras.io/#installation)

### Convert
```
python convert-uproot.py
```

### Train
```
python train.py
```
