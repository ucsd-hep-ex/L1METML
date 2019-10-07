# L1METML

Setup:
```
cmsrel CMSSW_10_6_1_patch2
cd CMSSW_10_6_1_patch2/src
cmsenv
git clone https://github.com/jmduarte/L1METML
cd L1METML
```

Convert:
```
python convert-uproot.py
```

Train:
```
python train.py
```