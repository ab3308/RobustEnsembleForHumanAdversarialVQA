#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=24:00:00

# set name of job
#SBATCH --job-name=vilbert_vizwiz_finetuned_inference

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
export MMF_SAVE_DIR=./mmf_tested/vilbert_vizwiz/

mmf_run config=mmf/projects/vilbert/configs/vizwiz/defaults.yaml model=vilbert dataset=vizwiz run_type=val checkpoint.resume_zoo=vilbert.finetuned.vizwiz >> ./mmf_test_results/vilbert_vizwiz_finetuned_on_vizwiz.txt

mmf_predict config=mmf/projects/vilbert/configs/vizwiz/defaults.yaml model=vilbert dataset=vizwiz run_type=test checkpoint.resume_zoo=vilbert.finetuned.vizwiz >> ./mmf_test_results/vilbert_vizwiz_finetuned_on_vizwiz.txt