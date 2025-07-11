# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
FROM gitlab-registry.nrp-nautilus.io/prp/jupyter-stack/tensorflow:latest

LABEL maintainer="Javier Duarte <jduarte@ucsd.edu>"

USER root

RUN apt-get update \
    && apt-get -yq --no-install-recommends install openssh-client vim emacs \
    && sudo apt-get clean && sudo rm -rf /var/lib/apt/lists/*

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
    git+https://github.com/jmduarte/hls4ml@split_pointwise_conv_by_rf#egg=hls4ml[profiling]
  
