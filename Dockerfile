# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
FROM gitlab-registry.nrp-nautilus.io/prp/jupyter-stack/tensorflow:latest

LABEL maintainer="Javier Duarte <jduarte@ucsd.edu>"

USER root

RUN apt-get update && apt-get -y install openssh-client

USER ${NB_UID}

# Install Tensorflow
RUN pip install --quiet --no-cache-dir \
    uproot \
    awkward \
    uproot \
    tqdm \
    setGPU \
    mplhep \
    git+https://github.com/jmduarte/qkeras#egg=qkeras
  
