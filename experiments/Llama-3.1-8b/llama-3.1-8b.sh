#!/bin/sh
#$ -cwd
#$ -l node_f=2
#$ -l h_rt=00:01:00:00
#$ -o outputs/Llama-3.1-8b/$JOB_ID.log
#$ -e outputs/Llama-3.1-8b/$JOB_ID.log
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

export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep 'inet ' | awk '{ print $2 }' | cut -d "/" -f 1)
export MASTER_PORT=$((10000 + ($JOB_ID % 50000)))

echo "MASTER_ADDR=${MASTER_ADDR}"

export NUM_GPU_PER_NODE=4
NODE_TYPE="h100"

NUM_NODES=$NHOSTS
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))

mkdir -p ./hostfile

HOSTFILE_NAME=./hostfile/hostfile_${JOB_ID}
while read -r hostname _ rest; do
  echo "${hostname} slots=${NUM_GPU_PER_NODE}"
done <"$PE_HOSTFILE" >"$HOSTFILE_NAME"

# distributed training
TENSOR_PARALLEL_SIZE=2
CONTEXT_PARALLEL_SIZE=1
PIPELINE_PARALLEL_SIZE=1
DATA_PARALLEL_SIZE=$((${NUM_GPUS} / (${TENSOR_PARALLEL_SIZE} * ${PIPELINE_PARALLEL_SIZE})))

# training config
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=1024
TRAIN_STEPS=25000
SEQ_LENGTH=8192

LR=2.5E-5
MIN_LR=2.5E-6
LR_WARMUP_STEPS=1000
WEIGHT_DECAY=0.1

# model config
TOKENIZER_MODEL=/gs/bs/tga-NII-LLM/hf-checkpoints/Meta-Llama-3.1-8B
CHECKPOINT_DIR=/gs/bs/tga-NII-LLM/checkpoints/hf-to-megatron/Llama-3.1-8b/tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}-v0.8
CHECKPOINT_SAVE_DIR=/gs/bs/tga-NII-LLM/checkpoints/Llama-3.1-8b/tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}-ct${CONTEXT_PARALLEL_SIZE}/LR${LR}-MINLR${MIN_LR}-WD${WEIGHT_DECAY}-v0.9.0

mkdir -p ${CHECKPOINT_SAVE_DIR}

# data config
TRAIN_DATA_PATH=""

# ja wikipedia
TRAIN_DATA_PATH="${TRAIN_DATA_PATH} 3382423156 /gs/fs/tga-NII-LLM/datasets/Meta-Llama-3_original_transformers-4.40.1/ja_wiki_merged_text_document"

# job name
JOB_NAME="Llama-3.1-8b-${NODE_TYPE}-${NUM_NODES}node-${NUM_GPUS}gpu-${SEQ_LENGTH}s-DP=${DATA_PARALLEL_SIZE}-TP=${TENSOR_PARALLEL_SIZE}-PP=${PIPELINE_PARALLEL_SIZE}-BS=${GLOBAL_BATCH_SIZE}-LR=${LR}-MINLR=${MIN_LR}-WARMUP=${LR_WARMUP_STEPS}-WD=${WEIGHT_DECAY}"

# run
mpirun -np $NUM_GPUS \
  --npernode $NUM_GPU_PER_NODE \
  -hostfile $HOSTFILE_NAME \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -x CUDA_DEVICE_MAX_CONNECTIONS=1 \
  -x TORCH_NCCL_AVOID_RECORD_STREAMS=1 \
  -x NCCL_IB_TIMEOUT=22 \
  -x UB_SKIPMC=1 \
  -x TOKENIZERS_PARALLELISM=true \
  -x LD_LIBRARY_PATH \
  -x PATH \
  -bind-to none \
  python experiments/Llama-3.1-8b/llama3.1_8b.py \
  --global-batch-size ${GLOBAL_BATCH_SIZE} \
  --micro-batch-size ${MICRO_BATCH_SIZE} \
  --seq-length ${SEQ_LENGTH} \
  --lr ${LR} \
  --min-lr ${MIN_LR} \
  --weight-decay ${WEIGHT_DECAY} \
  --train-iters ${TRAIN_STEPS} \
  --warmup-iters ${LR_WARMUP_STEPS} \
  --checkpoint-save-dir ${CHECKPOINT_SAVE_DIR} \
  --tokenizer-dir ${TOKENIZER_MODEL} \
  --data-path ${TRAIN_DATA_PATH} \
  --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
  --pipeline-parallel-size ${PIPELINE_PARALLEL_SIZE} \
  --sequence-parallel \
  --context-parallel-size ${CONTEXT_PARALLEL_SIZE} \
  --use-mpi \
  --num-nodes ${NUM_NODES} \
  --wandb-project "NeMo" \
  --wandb-entity "okoge" \
  --wandb-run-name ${JOB_NAME}
