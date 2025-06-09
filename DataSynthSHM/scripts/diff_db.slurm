#!/bin/bash
#SBATCH --job-name=diffusion_debug
#SBATCH --output=diff_run.out
#SBATCH --error=diff_run.err
#SBATCH -A es_chatzi
#SBATCH --gpus=rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL

echo "🔄 Job started on $(hostname) at $(date)"
echo "🧠 Using $SLURM_CPUS_ON_NODE CPU cores"

# ──────────────────────────  MODULES  ──────────────────────────
module load stack/2024-06
module load gcc/12.2.0
module load python_cuda/3.11.6

# ──────────────────────────  ENV VARS  ─────────────────────────
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_EULER_ROOT
export CUDA_DIR=$CUDA_EULER_ROOT

# Disable all W&B networking / logging (the SDK becomes a no-op)
export WANDB_MODE=disabled
unset  WANDB_API_KEY              # just to be safe

# ──────────────────────────  PROJECT PATHS  ────────────────────
cd /cluster/scratch/scansimo/DataSynthSHM        # project root
export PYTHONPATH=$PYTHONPATH:$PWD/src           # so diff_pt is importable

# ──────────────────────────  TRAINING CALL  ────────────────────
# Feel free to add overrides here (e.g. shorter epochs for debugging)
python -u src/diff_pt/run.py \
       debug_mode=true \
       diffusion.ae_epochs=3 \
       diffusion.batch_size=64


echo "✅ Job finished at $(date)"
