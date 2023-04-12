#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --account=eecs553w23_class
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=15G
#SBATCH --time=00:10:00
python3 synthesizecoeff.py Test/dog.wav Test/originaldog.wav Test/reconstructeddog.wav Test/originalmelspec.jpg Test/