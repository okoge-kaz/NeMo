#!/bin/bash

# python virtualenv
source .env/bin/activate

# pip install
pip install --upgrade pip
pip install -r requirements.txt

pip install ninja wheel packaging

# apt-get update && apt-get install -y libsndfile1 ffmpeg

pip install Cython
pip install nemo_toolkit['all']

# apex install
git clone git@github.com:NVIDIA/apex.git
cd apex

pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

# flash-attention install
git clone git@github.com:Dao-AILab/flash-attention.git
cd flash-attention
python setup.py install

# huggingface install
pip install transformers accelerate zarr tensorstore

# transformer engine
pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable
