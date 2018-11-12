#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=10GB

python3 cnn41_final_6.py



