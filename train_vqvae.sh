#!/bin/bash
#SBATCH --job-name=train
#SBATCH --account=eecs553w23_class
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
python3 train_vqvae.py --epoch 800
