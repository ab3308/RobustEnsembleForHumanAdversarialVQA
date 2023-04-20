#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=24:00:00

# set name of job
#SBATCH --job-name=m4c_finetuned_textvqa

# set number of GPUs
#SBATCH --gres=gpu:1

# set number of CPUs
#SBATCH --cpus-per-task=8

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=alex02sbrown@gmail.com

# run the application
module load python/anaconda3

source activate mmf_env

export TRANSFORMERS_CACHE=./transformers_cache/bert-base-uncased
export MMF_CACHE_DIR=./mmf_cache/
export MMF_SAVE_DIR=./mmf_tested/m4c_textvqa/

mmf_run config=mmf/projects/m4c/configs/textvqa/defaults.yaml model=m4c dataset=textvqa run_type=val checkpoint.resume_zoo=m4c.textvqa.alone >> ./mmf_test_results/m4c_finetuned_textvqa.txt

mmf_predict config=mmf/projects/m4c/configs/textvqa/defaults.yaml model=m4c dataset=textvqa run_type=test checkpoint.resume_zoo=m4c.textvqa.alone >> ./mmf_test_results/m4c_finetuned_textvqa.txt