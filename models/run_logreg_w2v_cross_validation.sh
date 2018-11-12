#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=100GB

python3 logreg_w2v_cross_validation.py
