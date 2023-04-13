#!/bin/bash
#SBATCH --job-name=train
#SBATCH --account=eecs553w23_class
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=180G
#SBATCH --time=08:00:00
python extract_spectrograms.py --vqvae_checkpoint /home/erssmith/Foley553/checkpoint/vqvae/vqvae.pth
