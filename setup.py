from setuptools import setup, find_packages

setup(
    name="L1METML",  
    version="0.1",  
    description="A package for L1 MET machine learning workflows", 
    author="Daniel Primosch",
    author_email="dprimosc@ucsd.edu", 
    url="https://github.com/dprim7/L1METML",  
    packages=find_packages(), 
    install_requires=[
        "uproot",
        "h5py",
        "matplotlib",
        "numpy",
        "mplhep",
        "hist",
    ],  
    python_requires=">=3.6",  
)