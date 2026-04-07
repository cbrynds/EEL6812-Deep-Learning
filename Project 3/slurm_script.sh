#!/bin/bash
#SBATCH --job-name=transformer_p3
#SBATCH --output=logs/transformer_%j.out
#SBATCH --error=logs/transformer_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00

source /apps/anaconda/anaconda-2023.09/etc/profile.d/conda.sh
conda activate pytorch2.2.0+py3.11+cuda12.1
set -euo pipefail

mkdir -p logs results


export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONUNBUFFERED=1

nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print($

python3 transformer.py