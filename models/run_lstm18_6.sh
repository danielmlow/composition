#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=10GB

python3 lstm18_6.py



