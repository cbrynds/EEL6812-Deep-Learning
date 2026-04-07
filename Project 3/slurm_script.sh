#!/bin/bash
#SBATCH --job-name=transformer_p3
#SBATCH --output=logs/transformer_%j.out
#SBATCH --error=logs/transformer_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00

source venv/bin/activate

set -euo pipefail

mkdir -p logs results


export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONUNBUFFERED=1

python3 transformer.py