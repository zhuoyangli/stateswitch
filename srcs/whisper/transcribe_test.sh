#!/bin/bash
#SBATCH --job-name=whisper_test
#SBATCH --partition=parallel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch4/choney1/zli230/stateswitch/data/logs/whisper_%j.out
#SBATCH --error=/scratch4/choney1/zli230/stateswitch/data/logs/whisper_%j.err

module load anaconda3/2024.02-1
module load cuda/12.1.0

conda activate whisper

python transcribe.py /scratch4/choney1/zli230/stateswitch/data/rec/raw/sub-001_ses-05.wav
