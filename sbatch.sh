#!/bin/bash
#SBATCH -c 1  # Number of Cores per Task
#SBATCH -p gpu  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 04:00:00  # Job time limit
#SBATCH -o slurm-%j.out  # %j = job ID
#SBATCH --output=job_output.txt
#SBATCH --error=job_error.txt
module load cuda/12.6
module load conda/latest
conda activate self-refine
MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_ADDR
PYTHONPATH=$(pwd) python src/medicalTranslation/run_Llama.py 