#!/bin/bash
#SBATCH --job-name=train
#SBATCH --account=eecs553w23_class
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=00:30:00
python inference.py --vqvae_checkpoint /home/erssmith/Foley553/checkpoint/vqvae/vqvae_965.pt --pixelsnail_checkpoint /home/erssmith/Foley553/checkpoint/pixelsnail-final/bottom_038.pt --number_of_synthesized_sound_per_class 2