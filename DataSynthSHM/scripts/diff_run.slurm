#!/bin/bash
#SBATCH --job-name=diffusion_run
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

# ✅ Load required modules
module load stack/2024-06
module load gcc/12.2.0
module load python_cuda/3.11.6

# ✅ Set CUDA env (required for some packages)
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_EULER_ROOT
export CUDA_DIR=$CUDA_EULER_ROOT

# ✅ Debug GPU availability
python -c "import torch; print(f'🟢 CUDA Available: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0)}')"
nvidia-smi

# ✅ Change into the project root (parent of src/)
cd /cluster/scratch/scansimo/DataSynthSHM

# ✅ Set PYTHONPATH so `diff_pt` is discoverable
export PYTHONPATH=/cluster/scratch/scansimo/DataSynthSHM/src:$HOME/.local/lib/python3.11/site-packages:$PYTHONPATH

export WANDB_API_KEY=5e39ff5743c2520a02866753184f39d953a6ff45
export WANDB_PROJECT=uncategorized
export WANDB_ENTITY=scansimo-eth-z-rich
export WANDB__SERVICE_WAIT=300
export WANDB_ERROR_REPORTING=disabled

# ✅ Run the training entrypoint
SWEEP_ID="b0rbzpc6"        # replace after you create the sweep once
wandb agent "$WANDB_ENTITY/$WANDB_PROJECT/$SWEEP_ID"

echo "✅ Job finished at $(date)"
