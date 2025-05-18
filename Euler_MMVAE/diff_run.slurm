#!/bin/bash
#SBATCH --job-name=diffusion_run
#SBATCH --output=diff_run.out
#SBATCH --error=diff_run.err
#SBATCH -A es_chatzi
#SBATCH --gpus=rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=18:00:00
#SBATCH --mail-type=END,FAIL

echo "🔄 Job started on $(hostname) at $(date)"
echo "🧠 Using $SLURM_CPUS_ON_NODE CPU cores"

# ✅ Load modules with CUDA & Pytorch support
module load stack/2024-06
module load gcc/12.2.0
module load python_cuda/3.11.6

# ✅ Set CUDA environment (required for XLA)
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_EULER_ROOT
export CUDA_DIR=$CUDA_EULER_ROOT

# ✅ (Optional) Confirm GPU availability from within job
python -c "import torch; print(f'🟢 CUDA Available: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0)}')"

echo "🔍 Checking GPU device..."
nvidia-smi

# ✅ Run your model
python -u scripts/diffusion_model.py

echo "✅ Job finished at $(date)"

