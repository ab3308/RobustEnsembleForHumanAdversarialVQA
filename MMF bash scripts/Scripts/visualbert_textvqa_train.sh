#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=24:00:00

# set name of job
#SBATCH --job-name=visualbert_textvqa_finetune

# set number of GPUs
#SBATCH --gres=gpu:8

# set number of CPUs
#SBATCH --cpus-per-task=40

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=alex02sbrown@gmail.com

# run the application
module load python/anaconda3

source activate mmf_env

export TRANSFORMERS_CACHE=./transformers_cache/bert-base-uncased
export MMF_CACHE_DIR=./mmf_cache/
export MMF_SAVE_DIR=./mmf_trained/visualbert_gqa/

mmf_run config=./mmf_configs/visualbert_textvqa.yaml model=visual_bert dataset=textvqa run_type=train training.fp16=True checkpoint.resume_pretrained=True checkpoint.resume_zoo=visual_bert.pretrained.coco.defaults training.num_workers=4 >> ./mmf_train_logs/visualbert_textvqa_finetune.txt