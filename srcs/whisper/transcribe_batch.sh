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
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zli230@jhu.edu

# Load required modules
echo "Loading modules..."
module load anaconda3/2024.02-1
module load cuda/12.1.0

# Activate conda environment
echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate whisper

# Set working directory
WORK_DIR="/scratch4/choney1/zli230/stateswitch/srcs/whisper"
cd "$WORK_DIR"
echo "Changed to directory: $(pwd)"

# Create logs directory if it doesn't exist
LOG_DIR="/scratch4/choney1/zli230/stateswitch/data/logs"
mkdir -p "$LOG_DIR"

# Run transcription script
python transcribe_batch.py --subject sub-009 --session ses-06

# Print completion information
echo "=========================================="
echo "Batch transcription completed successfully!"
echo "Job finished at: $(date)"
echo "Total runtime: $SECONDS seconds"
echo "=========================================="