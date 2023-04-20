#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=24:00:00

# set name of job
#SBATCH --job-name=m4c_textvqa_finetune

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
export MMF_SAVE_DIR=./mmf_trained/m4c_textvqa/

mmf_run config=mmf/projects/m4c/configs/textvqa/defaults.yaml model=m4c dataset=textvqa run_type=train training.fp16=True training.batch_size=960 training.early_stop.patience=4000 checkpoint.resume_pretrained=True checkpoint.resume_zoo=m4c.textvqa.alone training.num_workers=4 >> ./mmf_train_logs/textvqa_finetuned_textvqa_local.txt