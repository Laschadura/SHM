#!/bin/bash
#SBATCH --job-name=mvae_run_gpu
#SBATCH --output=vae_run.out
#SBATCH --error=vae_run.err
#SBATCH -A es_chatzi
#SBATCH --gpus=v100:2
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=18:00:00
#SBATCH --mail-type=END,FAIL

echo "🔄 Job started on $(hostname) at $(date)"
echo "🧠 Using $SLURM_CPUS_ON_NODE CPU cores"

# ✅ Load modules with CUDA & TensorFlow support
module load stack/2024-06
module load gcc/12.2.0
module load python_cuda/3.11.6

# ✅ Set CUDA environment (required for XLA)
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_EULER_ROOT
export CUDA_DIR=$CUDA_EULER_ROOT

# ✅ (Optional) Confirm GPU availability from within job
python -c "import tensorflow as tf; print('🟢 TF sees GPU:', tf.config.list_physical_devices('GPU'))"

echo "🔍 Checking GPU device..."
nvidia-smi

# ✅ Run your model
python -u scripts/vae_generator.py

echo "✅ Job finished at $(date)"

