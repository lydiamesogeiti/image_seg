# ============================================================
# GuidedBox Grid Search - Snellius SBATCH Script
# ============================================================
# Usage:
#   sbatch run_grid_search_v3.sh
#
# Before first run:
#   1. Update --mail-user above with your email
#   2. Create a virtual environment (see below)
#   3. Make sure dataset is in the right path
# ============================================================

echo "============================================"
echo "DEBUG: Job started at $(date)"
echo "DEBUG: Current Directory is $(pwd)"
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $SLURM_NODELIST"
echo "GPUs:         $SLURM_GPUS"
echo "Start time:   $(date)"
echo "Working dir:  $(pwd)"
echo "============================================"

# ----------------------------------------------------------
# 1. Load modules (Snellius standard)
# ----------------------------------------------------------
module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.6.0

# ----------------------------------------------------------
# 2. Activate virtual environment
# ----------------------------------------------------------
# First time setup (run these manually once before sbatch):
#   python -m venv $HOME/venvs/guidedbox
#   source $HOME/venvs/guidedbox/bin/activate
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
#   pip install numpy pillow matplotlib pyyaml tqdm pandas scikit-learn

VENV_PATH="$HOME/venvs/guidedbox"

if [ ! -d "$VENV_PATH" ]; then
    echo "ERROR: Virtual environment not found at $VENV_PATH"
    echo "Create it first with:"
    echo "  python -m venv $VENV_PATH"
    echo "  source $VENV_PATH/bin/activate"
    echo "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126"
    echo "  pip install numpy pillow matplotlib pyyaml tqdm pandas scikit-learn"
    exit 1
fi

source "$VENV_PATH/bin/activate"
echo "Python:       $(which python)"
echo "Python ver:   $(python --version)"

# Verify GPU is available
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# ----------------------------------------------------------
# 3. Navigate to project directory
# ----------------------------------------------------------
# Use absolute path - change this to your actual project path on Snellius
PROJECT_DIR="/gpfs/home6/lmesogeiti/bbox_learn/bbox_learn"
cd "$PROJECT_DIR" || { echo "ERROR: Could not cd to $PROJECT_DIR"; exit 1; }

echo "Project dir:  $(pwd)"
echo "============================================"

# ----------------------------------------------------------
# 4. Create output directory
# ----------------------------------------------------------
mkdir -p guided_box_ijmond/grid_search_v3

# ----------------------------------------------------------
# 5. Run grid search
# ----------------------------------------------------------
echo "Starting grid search..."
echo ""

python grid_search_v3.py \
    --config guidedbox_config.yaml \
    --device cuda \
    --output_dir guided_box_ijmond/grid_search_v3 \
    --num_workers 4

echo ""
echo "============================================"
echo "Job finished:  $(date)"
echo "Exit code:     $?"
echo "============================================"