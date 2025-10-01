#!/bin/bash
#SBATCH --job-name=mfa_align
#SBATCH --partition=parallel
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch4/choney1/zli230/stateswitch/data/logs/mfa_align_%j.out
#SBATCH --error=/scratch4/choney1/zli230/stateswitch/data/logs/mfa_align_%j.err

# Load modules
module load anaconda3/2024.02-1

# Activate MFA environment
conda activate mfa

# Run MFA alignment with audio in separate directory
mfa align \
    /scratch4/choney1/zli230/stateswitch/data/mfa/ \
    english_us_arpa \
    english_us_arpa \
    /scratch4/choney1/zli230/stateswitch/data/textgrids/ \
    --audio_directory /scratch4/choney1/zli230/stateswitch/data/rec/chunked/ \
    --clean \
    --beam 100 \
    --retry_beam 400 \
    -j 16 \

echo "MFA alignment completed!"
