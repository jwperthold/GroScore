#!/bin/bash
#SBATCH -J GroScore
#SBATCH --nodes=1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
