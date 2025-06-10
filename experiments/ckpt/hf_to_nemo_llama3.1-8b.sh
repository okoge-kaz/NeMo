#!/bin/sh
#PBS -q rt_HF
#PBS -N convert
#PBS -l select=1:ncpus=192:ngpus=8
#PBS -l walltime=1:00:00
#PBS -j oe
#PBS -koed
#PBS -o outputs/convert/hf_to_nemo
#PBS -P gag51395

cd $PBS_O_WORKDIR
mkdir -p outputs/convert/hf_to_nemo

source /etc/profile.d/modules.sh
module use /home/acf15649kv/modules/modulefiles

module load hpcx/2.21.0

# checkpoint directories
HF_CHECKPOINT_DIR="/groups/gag51395/hf_checkpoints/Llama-3.1-Swallow-8B-v0.5"
NEMO_CHECKPOINT_DIR="/groups/gag51395/fujii/checkpoints/hf-to-nemo"
mkdir -p $NEMO_CHECKPOINT_DIR

# singularity image
SINGULARITY_IMAGE="/groups/gag51395/fujii/container/ngc-pytorch-25.04-te.sif"

# convert
singularity exec \
  --nv \
  --bind /groups/gag51395:/groups/gag51395 \
  --bind /home/acf15649kv:/home/acf15649kv \
  --bind /dev/shm:/dev/shm \
  --bind /tmp:/tmp \
  $SINGULARITY_IMAGE \
  bash -c "\
    PYTHONPATH=/groups/gag51395/fujii/src/NeMo \
    python scripts/checkpoint_converters/convert_llama_hf_to_nemo.py \
    --input_name_or_path $HF_CHECKPOINT_DIR \
    --output_path $NEMO_CHECKPOINT_DIR/llama3.1-8b-swallow-v0.5.nemo \
    --precision bf16 \
    --llama31 True"
