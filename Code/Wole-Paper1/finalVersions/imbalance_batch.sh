#!/bin/bash
#SBATCH --job-name="AugImbalance"
#SBATCH --output="AugImbalance%j.%N.out"
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --mem=250g
#SBATCH --account=wpi102
#SBATCH -t 08:00:00

python augmentationNullF1CrossVal.py
