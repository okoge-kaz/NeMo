#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=00:01:00:00
#$ -o outputs/install/$JOB_ID.log
#$ -e outputs/install/$JOB_ID.log
#$ -p -3

# Load modules
module use /gs/fs/tga-NII-LLM/modules/modulefiles

module load ylab/cuda/12.4
module load ylab/cudnn/9.1.0
module load ylab/nccl/cuda-12.4/2.21.5
module load ylab/hpcx/2.17.1
module load ninja/1.11.1

source .env/bin/activate

# python packages
pip install --upgrade pip
pip install --upgrade wheel cmake ninja packaging

# install pytorch
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
git checkout v2.5.1
git submodule sync
git submodule update --init --recursive

# pytorch requirements
pip install -r requirements.txt

# nccl, mpi, backend pytorch
export USE_DISTRIBUTED=1
export USE_MPI=1
export USE_CUDA=1
export MAX_JOBS=16

# set tmp dir
export TMPDIR="/gs/bs/tge-gc24sp03/cache"
export TMP="/gs/bs/tge-gc24sp03/cache"

python setup.py develop

pip install triton

# install apex, transformer engine, mcore
# https://github.com/NVIDIA/NeMo/tree/main?tab=readme-ov-file#install-llms-and-mms-dependencies

# apex
git clone https://github.com/NVIDIA/apex.git
cd apex
git checkout $apex_commit
pip install . -v --no-build-isolation --disable-pip-version-check --no-cache-dir --config-settings "--build-option=--cpp_ext --cuda_ext --fast_layer_norm --distributed_adam --deprecated_fused_adam --group_norm"

# transformer engine
pip install git+https://github.com/NVIDIA/TransformerEngine.git@v1.12

# mcore
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_r0.9.0
pip install -e .
cd megatron/core/datasets
make

# nemo
pip install Cython packaging

# mamba-ssm
git clone git@github.com:state-spaces/mamba.git
cd mamba
git checkout v2.2.2
pip install .

# nemo requirements
pip install nemo_toolkit['all']

# pytorch re-install
python setup.py develop

# nemo run
pip install git+https://github.com/NVIDIA/NeMo-Run.git
