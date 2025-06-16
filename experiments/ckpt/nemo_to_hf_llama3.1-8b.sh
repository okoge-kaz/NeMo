#!/bin/sh
#PBS -q rt_HF
#PBS -N convert
#PBS -l select=1:ncpus=192:ngpus=8
#PBS -l walltime=1:00:00
#PBS -j oe
#PBS -koed
#PBS -o outputs/convert/nemo_to_hf
#PBS -P gag51395

cd $PBS_O_WORKDIR
mkdir -p outputs/convert/hf_to_nemo

source /etc/profile.d/modules.sh
module use /home/acf15649kv/modules/modulefiles

module load hpcx/2.21.0

# checkpoint directories
HF_CHECKPOINT_DIR="/groups/gag51395/fujii/checkpoints/nemo-to-hf/Llama-3.1-8b-v0.5/lmsys-chat-1m-gemma-3-ja"
NEMO_CHECKPOINT_PATH="/groups/gag51395/fujii/checkpoints/nemo-aligner/llama-3.1-8b/sft/Llama-3.1-Swallow-8B-v0.5-lmsys-chat-1m-gemma-3-ja/checkpoints/megatron_gpt_sft.nemo"
mkdir -p $HF_CHECKPOINT_DIR
mkdir -p $HF_CHECKPOINT_DIR/hf
mkdir -p $HF_CHECKPOINT_DIR/pytorch

# tokenizer setting
TOKENIZER_DIR="/groups/gag51395/hf_checkpoints/Llama-3.1-Swallow-8B-v0.5"

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
    python scripts/checkpoint_converters/convert_llama_nemo_to_hf.py\
    --input_name_or_path $NEMO_CHECKPOINT_PATH \
    --output_path $HF_CHECKPOINT_DIR/pytorch/pytorch_model.bin \
    --hf_input_path /groups/gag51395/hf_checkpoints/Llama-3.1-Swallow-8B-v0.5 \
    --hf_output_path $HF_CHECKPOINT_DIR/hf \
    --cpu-only"

# copy tokenizer files
cp $TOKENIZER_DIR/tokenizer* $HF_CHECKPOINT_DIR/hf/
cp $TOKENIZER_DIR/special_tokens_map.json $HF_CHECKPOINT_DIR/hf/
