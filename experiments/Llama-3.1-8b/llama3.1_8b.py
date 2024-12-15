import argparse
import os
import torch
from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer
from nemo.collections.llm.gpt.model.llama import Llama31Config8B, LlamaModel
from megatron.core.optimizer import OptimizerConfig
from pytorch_lightning.loggers import WandbLogger


def parse_args():
    parser = argparse.ArgumentParser(description='Llama-3.1-8b')
    # training args
    parser.add_argument("--global-batch-size", type=int, default=1024)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--seq-length", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=2.5e-5)
    parser.add_argument("--min-lr", type=float, default=2.5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--train-iters", type=int, default=25000)
    parser.add_argument("--warmup-iters", type=int, default=1000)
    parser.add_argument("--checkpoint-save-dir", type=str, default="")
    # tokenizer args
    parser.add_argument("--tokenizer-dir", type=str, default="")
    # data args
    parser.add_argument('--data-path', nargs='*', default=None)
    # distributed training args
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--virtual-pipeline-parallel-size", type=int, default=None)
    parser.add_argument("--sequence-parallel", action='store_true')
    parser.add_argument("--context-parallel-size", type=int, default=1)
    parser.add_argument("--use-mpi", action='store_true')
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--gpu-per-node", type=int, default=4)
    # wandb args
    parser.add_argument("--wandb-project", type=str, default="llama-3.1-8b")
    parser.add_argument("--wandb-entity", type=str, default="nvidia")
    parser.add_argument("--wandb-run-name", type=str, default="llama-3.1-8b")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    if args.use_mpi:
        global_rank = int(os.getenv('OMPI_COMM_WORLD_RANK', 0))
        local_rank = int(os.getenv('OMPI_COMM_WORLD_LOCAL_RANK', 0))
        world_size = int(os.getenv('OMPI_COMM_WORLD_SIZE', 1))

        os.environ['RANK'] = str(global_rank)
        os.environ['LOCAL_RANK'] = str(local_rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ["NODE_RANK"] = str(global_rank // args.gpu_per_node)

    tokenizer = get_tokenizer(  # AutoTokenizer
        tokenizer_name=args.tokenizer_dir,
    )
    data = llm.PreTrainingDataModule(
        paths=args.data_path,
        seq_length=args.seq_length,
        tokenizer=tokenizer,
        global_batch_size=args.global_batch_size,
        micro_batch_size=args.micro_batch_size,
        num_workers=4,  # tsubame has 4 GPUs
        pin_memory=True,
        split="990,10,0",
    )

    # Llama-3.1-8B model
    model = LlamaModel(
        config=Llama31Config8B(  # type: ignore
            tensor_model_parallel_size=args.tensor_parallel_size,
            pipeline_model_parallel_size=args.pipeline_parallel_size,
            virtual_pipeline_model_parallel_size=args.virtual_pipeline_parallel_size,
            sequence_parallel=args.sequence_parallel,
            context_parallel_size=args.context_parallel_size,
        ),
    )
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=args.tensor_parallel_size,
        pipeline_model_parallel_size=args.pipeline_parallel_size,
        virtual_pipeline_model_parallel_size=args.virtual_pipeline_parallel_size,
        sequence_parallel=args.sequence_parallel,
        context_parallel_size=args.context_parallel_size,
    )
    optimizer = nl.MegatronOptimizerModule(
        config=OptimizerConfig(
            optimizer='adam',
            lr=args.lr,
            min_lr=args.min_lr,
            weight_decay=args.weight_decay,
            adam_beta1=0.9,
            adam_beta2=0.95,
            adam_eps=1e-8,
            clip_grad=1.0,
            bf16=True,
            use_distributed_optimizer=True,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
        ),
        lr_scheduler=nl.lr_scheduler.CosineAnnealingScheduler(
            max_steps=args.train_iters,
            warmup_steps=args.warmup_iters,
            min_lr=args.min_lr,
        )
    )
    trainer = nl.Trainer(
        num_nodes=args.num_nodes,
        devices=args.gpu_per_node,
        accelerator="gpu",
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
        max_steps=args.train_iters,
        strategy=strategy,
        log_every_n_steps=1,
        val_check_interval=500,
        limit_val_batches=10,
        limit_test_batches=10,
        num_sanity_val_steps=0,
    )

    # logging
    nemo_logger = nl.NeMoLogger(
        log_dir=args.checkpoint_save_dir,
        log_global_rank_0_only=True,
        wandb=WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
        )
    )

    llm.train(
        model=model,
        data=data,
        trainer=trainer,
        log=nemo_logger,
        tokenizer=tokenizer,
        optim=optimizer,
        resume=None,
    )
