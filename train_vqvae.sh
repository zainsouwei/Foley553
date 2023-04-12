#!/bin/bash
#SBATCH --job-name=train
#SBATCH --account=eecs553w23_class
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=180G
#SBATCH --time=08:00:00
python train_vqvae.py --epoch 1600 --checkpoint_file /home/zainsou/Desktop/Foley553/checkpoint/vqvae/vqvae.pth --start_epoch 800
