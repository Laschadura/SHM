#!/bin/bash
#SBATCH --job-name=mmvae_run
#SBATCH --output=logs/mmvae_run.out
#SBATCH --error=logs/mmvae_run.err
#SBATCH -A es_chatzi
#SBATCH --gpus=rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=18:00:00
#SBATCH --mail-type=END,FAIL

echo "🔄 Job started on $(hostname) at $(date)"
echo "🧠 Using $SLURM_CPUS_ON_NODE CPU cores"

# ✅ Load required modules
module load stack/2024-06
module load gcc/12.2.0
module load python_cuda/3.11.6

# ✅ Set CUDA/XLA environment variables
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_EULER_ROOT
export CUDA_DIR=$CUDA_EULER_ROOT

# ✅ Confirm TensorFlow sees the GPU
python -c "import tensorflow as tf; print('🟢 TF sees GPU:', tf.config.list_physical_devices('GPU'))"
nvidia-smi

# ✅ Move into project root
cd /cluster/scratch/scansimo/DataSynthSHM

# ✅ Ensure Python can find `bridge_data` and `mmvae_tf`
export PYTHONPATH=$PYTHONPATH:/cluster/scratch/scansimo/DataSynthSHM/src

# ✅ Optional: override root via env var
export PROJECT_ROOT=/cluster/scratch/scansimo/DataSynthSHM

# ✅ Launch the MMVAE pipeline
python -u -m mmvae_tf.run --cfg configs/mmvae.yaml

echo "✅ Job finished at $(date)"
