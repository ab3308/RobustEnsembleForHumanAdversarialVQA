#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=24:00:00

# set name of job
#SBATCH --job-name=train_classifier

# set number of GPUs
#SBATCH --gres=gpu:1

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=alex02sbrown@gmail.com

# run the application
module load python/anaconda3

source activate simple_transformers

python3 ./text_classifier/train_model_25k.py >> classifier_25k.txt

python3 ./text_classifier/train_model_25k_lowercase.py >> classifier_lowercase.txt

python3 ./text_classifier/train_model_25k_lowercase_capitalised.py >> classifier_lowercase_caps.txt

python3 ./text_classifier/train_model_25k_alternating.py >> classifier_alternating.txt

python3 ./text_classifier/train_model_25k_alternating_capitalised.py >> classifier_alternating_caps.txt