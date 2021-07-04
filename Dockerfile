# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
FROM jupyter/tensorflow-notebook:latest

LABEL maintainer="Javier Duarte <jduarte@ucsd.edu>"

USER root

RUN apt-get update && apt-get -y install openssh-client

USER ${NB_UID}

# Install Tensorflow
RUN pip install --quiet --no-cache-dir \
    uproot \
    tables \
    pandas \
    h5py \
    tqdm \
    scikit-learn \
    setGPU \
    mplhep \
    awkward
  
