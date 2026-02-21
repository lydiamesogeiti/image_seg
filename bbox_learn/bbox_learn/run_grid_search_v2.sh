#!/bin/bash
#SBATCH --job-name=guidedbox_v2
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/gpfs/home6/lmesogeiti/bbox_learn/slurm_%j.out
#SBATCH --error=/gpfs/home6/lmesogeiti/bbox_learn/slurm_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lydia.mesogeiti@student.uva.nl

# =====================================================
# GuidedBox Grid Search V2 - Comprehensive
# 72 combinations: 2 networks x 4 epoch counts x 3 LRs x 3 PWs
# Fixed: ema=0.9, img_size=512
# Improvements: cosine LR, backbone freeze, confidence tracking
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

# Create output directory
OUTPUT_DIR="guided_box_ijmond/grid_search_v2"
mkdir -p $OUTPUT_DIR

echo "Starting Grid Search V2"
echo "  Networks:  resnet50_3layer, resnet50_4layer"
echo "  Epochs:    10, 20, 50, 100"
echo "  LR:        0.0001, 0.00005, 0.00001"
echo "  PW:        1.5, 2.5, 3"
echo "  Total:     72 combinations"
echo "  Features:  cosine LR, backbone freeze, confidence tracking, gradient clipping"
echo ""

# Run grid search
python grid_search_v2.py \
    --config guidedbox_config.yaml \
    --device cuda \
    --output_dir $OUTPUT_DIR

echo ""
echo "========================================"
echo "Finished:    $(date)"
echo "========================================"
