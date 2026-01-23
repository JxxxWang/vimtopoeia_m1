#!/bin/bash
#SBATCH --job-name=vim_m1_train
#SBATCH --output=../sbatch/m1_%j.out
#SBATCH --error=../sbatch/m1_%j.err
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4            
#SBATCH --mem=64G                     
#SBATCH --time=3:00:00      
    
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate vim_m1
cd /scratch/hw3140/vimtopoeia_m1
export PROJECT_ROOT=/scratch/hw3140/vimtopoeia_m1

# Print HPC environment info
echo "=========================================="
echo "Starting training on HPC"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "Project root: $PROJECT_ROOT"
echo "Time: $(date)"
echo "Working directory: $(pwd)"
echo "=========================================="

# Show GPU info
nvidia-smi

echo ""
echo "Starting model training..."
echo ""

# Run training with Hydra using HPC config
python src/train.py data=surge_hpc model=surge_flow trainer=gpu +run_name=surge_flow_train logger=csv

echo ""
echo "=========================================="
echo "Training completed at: $(date)"
echo "=========================================="



