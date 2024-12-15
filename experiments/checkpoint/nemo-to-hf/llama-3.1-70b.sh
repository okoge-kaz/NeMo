#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=00:02:00:00
#$ -o outputs/convert/nemo-to-hf/$JOB_ID.log
#$ -e outputs/convert/nemo-to-hf/$JOB_ID.log
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
export TOKENIZERS_PARALLELISM=false

# model config
TOKENIZER_MODEL_DIR=/gs/bs/tga-NII-LLM/hf-checkpoints/Meta-Llama-3.1-70B
NEMO_CHECKPOINT_PATH=/gs/bs/tga-NII-LLM/checkpoints/hf-to-nemo/Llama-3.1-70b/llama-3.1-70b.nemo
TMP_CHECKPOINT_DIR=/gs/bs/tga-NII-LLM/checkpoints/nemo-to-hf/Llama-3.1-70b/tmp
HF_CHECKPOINT_DIR=/gs/bs/tga-NII-LLM/checkpoints/nemo-to-hf/Llama-3.1-70b/hf

mkdir -p $TMP_CHECKPOINT_DIR
mkdir -p $HF_CHECKPOINT_DIR

python scripts/checkpoint_converters/convert_llama_nemo_to_hf.py \
  --input_name_or_path $NEMO_CHECKPOINT_PATH \
  --output_path $TMP_CHECKPOINT_DIR/pytorch_model.bin \
  --hf_input_path $TOKENIZER_MODEL_DIR \
  --hf_output_path $HF_CHECKPOINT_DIR \
  --input_tokenizer $TOKENIZER_MODEL_DIR \
  --hf_output_tokenizer $HF_CHECKPOINT_DIR \
  --cpu-only


