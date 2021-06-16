# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
ARG BASE_CONTAINER=prp/jupyter-stack/tensorflow
gitlab-registry.nautilus.optiputer.net/prp/jupyter-stack/tensorflow:latest
FROM $BASE_CONTAINER

LABEL maintainer="Javier Duarte <jduarte@ucsd.edu>"

# Install Tensorflow
RUN pip install --quiet --no-cache-dir \
    pytables \
    pandas \
    h5py \
    tqdm \
    scikit-learn \
    setGPU \
    mplhep \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"