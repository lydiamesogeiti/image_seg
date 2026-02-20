#!/bin/bash
#SBATCH --job-name=guidedbox_pw_net
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# =====================================================
# GuidedBox Grid Search - PW + Network Architecture
# 15 combinations: 3 networks x 5 positive weights
# Fixed: lr=0.0001, ema=0.9, img_size=512, epochs=20
# =====================================================

echo "========================================"
echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURM_NODELIST"
echo "GPUs:        $SLURM_GPUS"
echo "Started:     $(date)"
echo "========================================"

# Load modules
module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.6.0

# Activate virtual environment
source $HOME/venvs/guidedbox/bin/activate

# Navigate to project directory
cd /gpfs/home6/lmesogeiti/bbox_learn/bbox_learn

echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
echo ""

# Output directory
OUTPUT_DIR="guided_box_ijmond/grid_search_pw"
mkdir -p $OUTPUT_DIR

echo "Starting grid search: 3 networks x 5 pw values = 15 combinations"
echo "Networks:  resnet50_3layer, resnet34_2layer, resnet50_4layer"
echo "PW values: 1, 1.5, 2, 2.5, 3"
echo "Fixed:     lr=0.0001, ema=0.9, img_size=512, epochs=20"
echo ""

# Run grid search
python grid_search_pw_only.py \
    --config guidedbox_config.yaml \
    --epochs 20 \
    --device cuda \
    --output_dir $OUTPUT_DIR

echo ""
echo "========================================"
echo "Finished:    $(date)"
echo "========================================"
