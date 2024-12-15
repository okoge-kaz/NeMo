#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=00:01:00:00
#$ -o outputs/convert/hf-to-nemo/$JOB_ID.log
#$ -e outputs/convert/hf-to-nemo/$JOB_ID.log
#$ -p -3

set -e

# Load modules
module use /gs/fs/tga-NII-LLM/modules/modulefiles

module load ylab/cuda/12.4
module load ylab/cudnn/9.1.0
module load ylab/nccl/cuda-12.4/2.21.5
module load ylab/hpcx/2.17.1
module load ninja/1.11.1

source .env/bin/activate

export TMPDIR="/gs/bs/tge-gc24sp03/cache"
export TMP="/gs/bs/tge-gc24sp03/cache"

# model config
TOKENIZER_MODEL_DIR=/gs/bs/tga-NII-LLM/hf-checkpoints/Meta-Llama-3.1-8B
CHECKPOINT_DIR=/gs/bs/tga-NII-LLM/checkpoints/hf-to-nemo/Llama-3.1-8b

mkdir -p $CHECKPOINT_DIR

export TOKENIZERS_PARALLELISM=false

python scripts/checkpoint_converters/convert_llama_hf_to_nemo.py \
  --input_name_or_path $TOKENIZER_MODEL_DIR \
  --output_path $CHECKPOINT_DIR/llama-3.1-8b.nemo \
  --precision bf16 \
  --llama31 True
