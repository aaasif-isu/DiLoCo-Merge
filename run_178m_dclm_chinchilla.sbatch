#!/bin/bash
# =============================================================================
# Slurm Batch Script: 178M Llama DCLM Chinchilla-Budget (3.55B Tokens) Benchmark
# Target: 1x NVIDIA A100 GPU, 4 Virtual Workers, H=30 Local Steps, 225 Rounds
# Sequential Execution: DiLoCo (diloco) -> IsoLoCo (isoloco) -> RankLoCo (erank_iso)
# =============================================================================

#SBATCH --job-name=diloco_178m_dclm_chinchilla
#SBATCH --partition=nova
#SBATCH --account=jannesar-lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=4-00:00:00
#SBATCH --output=/lustre/hdd/LAS/jannesar-lab/aaasif/diloco_merge/scripts/logs/chinchilla_178m_%j.out
#SBATCH --error=/lustre/hdd/LAS/jannesar-lab/aaasif/diloco_merge/scripts/logs/chinchilla_178m_%j.err
#SBATCH --mail-user=aaasif@iastate.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -Eeuo pipefail

# --- Repository & Dataset Paths ---
REPO_DIR="/lustre/hdd/LAS/jannesar-lab/aaasif/diloco_merge"
VENV_DIR="${REPO_DIR}/diloco-env"
PYTHON_BIN="${VENV_DIR}/bin/python"
DATA_DIR="/lustre/hdd/LAS/jannesar-lab/aaasif/datasets/dclm_178m_3p55b"
LOG_DIR="${REPO_DIR}/scripts/logs/job_${SLURM_JOB_ID:-local}"

# --- Redirect Caches & Temporary Files to Lustre (Prevents $HOME Quota Exhaustion) ---
export CACHE_ROOT="/lustre/hdd/LAS/jannesar-lab/aaasif/.cache"
export HF_HOME="${CACHE_ROOT}/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export XDG_CACHE_HOME="${CACHE_ROOT}"
export TRITON_CACHE_DIR="/lustre/hdd/LAS/jannesar-lab/aaasif/.triton"
export TORCH_HOME="${CACHE_ROOT}/torch"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/torchinductor"
export WANDB_DIR="${REPO_DIR}"
export WANDB_CACHE_DIR="${CACHE_ROOT}/wandb"
export TMPDIR="/lustre/hdd/LAS/jannesar-lab/aaasif/.cache/tmp"

# --- Runtime & CUDA Memory Settings ---
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Prevent non-interactive W&B login prompts if WANDB_API_KEY / ~/.netrc is not configured
if [ -z "${WANDB_API_KEY:-}" ] && [ ! -f "${HOME}/.netrc" ]; then
    export WANDB_MODE="${WANDB_MODE:-offline}"
fi

mkdir -p "${LOG_DIR}" "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRITON_CACHE_DIR}" \
         "${TORCH_HOME}" "${TORCHINDUCTOR_CACHE_DIR}" "${WANDB_CACHE_DIR}" "${TMPDIR}"

cd "${REPO_DIR}"

if [ -f "${VENV_DIR}/bin/activate" ]; then
    source "${VENV_DIR}/bin/activate"
fi

echo "================================================================================"
echo "Slurm Job ID       : ${SLURM_JOB_ID:-N/A}"
echo "Hostname           : $(hostname)"
echo "Start Time         : $(date --iso-8601=seconds)"
echo "Working Directory  : $(pwd)"
echo "Python Executable  : ${PYTHON_BIN}"
echo "CPUs Allocated     : ${SLURM_CPUS_PER_TASK:-16}"
echo "Dataset Directory  : ${DATA_DIR}"
echo "Job Log Directory  : ${LOG_DIR}"
echo "WANDB_MODE         : ${WANDB_MODE:-online}"
echo "================================================================================"
nvidia-smi

# Step 1: Dataset Preprocessing & 4-Worker Partition Generation (Safe & Non-Destructive)
if [ -f "${DATA_DIR}/metadata.json" ] && [ -f "${DATA_DIR}/train_000003.npy" ] && [ -f "${DATA_DIR}/valid_000000.npy" ]; then
    if [ ! -f "${DATA_DIR}/partitions_train_iid_w4.json" ] || [ ! -f "${DATA_DIR}/partitions_valid_iid_w4.json" ]; then
        echo ""
        echo "[$(date --iso-8601=seconds)] Step 1/4: Shards exist in ${DATA_DIR}; generating 4-worker partition manifests..."
        "${PYTHON_BIN}" -u pretokenize_data.py \
            --config configs/diloco_178m.yaml \
            --create_partitions_only \
            2>&1 | tee "${LOG_DIR}/00_create_partitions_w4.log"
    else
        echo ""
        echo "[$(date --iso-8601=seconds)] Step 1/4: Verified existing preprocessed dataset and w4 partitions in ${DATA_DIR}."
    fi
elif [ ! -d "${DATA_DIR}" ] || [ -z "$(ls -A "${DATA_DIR}" 2>/dev/null)" ]; then
    echo ""
    echo "[$(date --iso-8601=seconds)] Step 1/4: Pre-tokenizing 3.55B DCLM train + 104.9M valid tokens into ${DATA_DIR}..."
    "${PYTHON_BIN}" -u pretokenize_data.py \
        --config configs/diloco_178m.yaml \
        --num_proc "${SLURM_CPUS_PER_TASK:-16}" \
        2>&1 | tee "${LOG_DIR}/00_preprocess_dclm_178m.log"
else
    echo ""
    echo "[ERROR] Dataset directory ${DATA_DIR} already exists and is non-empty, but appears incomplete (missing metadata.json or expected shards)." >&2
    echo "[ERROR] Refusing to delete ${DATA_DIR} automatically. Please inspect its contents and remove or rename it manually if a fresh run is intended:" >&2
    ls -lh "${DATA_DIR}" >&2
    exit 1
fi

# Step 2: Train Method 1 — DiLoCo (diloco)
echo ""
echo "[$(date --iso-8601=seconds)] Step 2/4: Training DiLoCo (configs/diloco_178m.yaml)"
"${PYTHON_BIN}" -u train.py \
    --config configs/diloco_178m.yaml \
    2>&1 | tee "${LOG_DIR}/01_train_diloco_178m.log"

# Step 3: Train Method 2 — IsoLoCo (isoloco)
echo ""
echo "[$(date --iso-8601=seconds)] Step 3/4: Training IsoLoCo (configs/isoloco_178m.yaml)"
"${PYTHON_BIN}" -u train.py \
    --config configs/isoloco_178m.yaml \
    2>&1 | tee "${LOG_DIR}/02_train_isoloco_178m.log"

# Step 4: Train Method 3 — RankLoCo (erank_iso)
echo ""
echo "[$(date --iso-8601=seconds)] Step 4/4: Training RankLoCo (configs/erank_iso_178m.yaml)"
"${PYTHON_BIN}" -u train.py \
    --config configs/erank_iso_178m.yaml \
    2>&1 | tee "${LOG_DIR}/03_train_erank_iso_178m.log"

# Post-Run Summary Comparison
echo ""
echo "[$(date --iso-8601=seconds)] Generating comparison summary across all three methods..."
"${PYTHON_BIN}" -u scripts/compare_methods.py \
    --dirs \
        "${REPO_DIR}/outputs/dclm_178m_diloco_iid_w4_h30_r225_s42" \
        "${REPO_DIR}/outputs/dclm_178m_isoloco_iid_w4_h30_r225_s42" \
        "${REPO_DIR}/outputs/dclm_178m_erank_iso_iid_w4_h30_r225_s42" \
    --output_dir "${LOG_DIR}/comparison_summary" \
    2>&1 | tee "${LOG_DIR}/04_compare_methods.log"

echo ""
echo "All 178M DCLM Chinchilla-budget experiments completed successfully at $(date --iso-8601=seconds)"

