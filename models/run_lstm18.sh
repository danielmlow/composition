#!/bin/bash
#SBATCH --time=01:20:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=30GB

python3 lstm18.py



