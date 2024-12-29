#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=00:01:00:00
#$ -o outputs/convert/nemo-v1-to-v2/$JOB_ID.log
#$ -e outputs/convert/nemo-v1-to-v2/$JOB_ID.log
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
NEMO_V1_CHECKPOINT_DIR=/gs/bs/tga-NII-LLM/checkpoints/hf-to-nemo/Llama-3.1-8b
NEMO_V2_CHECKPOINT_DIR=/gs/bs/tga-NII-LLM/checkpoints/hf-to-nemo/Llama-3.1-8b-nemo-v2

mkdir -p $NEMO_V2_CHECKPOINT_DIR

export TOKENIZERS_PARALLELISM=false

python scripts/checkpoint_converters/convert_nemo1_to_nemo2.py \
  --input_path $NEMO_V1_CHECKPOINT_DIR/llama-3.1-8b.nemo \
  --tokenizer_path $TOKENIZER_MODEL_DIR/tokenizer.json \
  --tokenizer_library huggingface \
  --output_path $NEMO_V2_CHECKPOINT_DIR \
  --model_id meta-llama/Meta-Llama-3-8B
