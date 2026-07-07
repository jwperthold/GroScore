#!/bin/bash
#SBATCH -J GroScore
#SBATCH --partition=zen2_0256_a40x2
#SBATCH --qos=zen2_0256_a40x2
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
