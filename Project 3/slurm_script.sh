#!/bin/bash
#SBATCH --job-name=transformer_p3
#SBATCH --output=logs/transformer_%j.out
#SBATCH --error=logs/transformer_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00

set -euo pipefail

mkdir -p logs results

source venv/bin/activate
python3 transformer.py