# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
FROM gitlab-registry.nrp-nautilus.io/prp/jupyter-stack/tensorflow:latest

LABEL maintainer="Javier Duarte <jduarte@ucsd.edu>"

USER root

RUN apt-get update && apt-get -y install openssh-client

USER ${NB_UID}

# Install Tensorflow
RUN pip install --quiet --no-cache-dir \
    coffea \
    uproot \
    awkward \
    uproot \
    tqdm \
    setGPU \
    mplhep \
    autopep8 \
    git+https://github.com/google/qkeras#egg=qkeras \
    git+https://github.com/jmduarte/hls4ml@l1metml#egg=hls4ml[profiling]
  
