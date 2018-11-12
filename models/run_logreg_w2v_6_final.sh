#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=10GB

python3 logreg_w2v_6_final.py
