#!/bin/bash
#SBATCH --job-name=whisper_batch
#SBATCH --partition=parallel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch4/choney1/zli230/stateswitch/data/logs/whisper_batch_%j.out
#SBATCH --error=/scratch4/choney1/zli230/stateswitch/data/logs/whisper_batch_%j.err

# Load modules
module load anaconda3/2024.02-1
module load cuda/12.1.0

# Activate conda environment
conda activate whisper

# Run batch transcription
cd /scratch4/choney1/zli230/stateswitch/srcs/whisper
python transcribe_batch.py

echo "Batch transcription completed!"