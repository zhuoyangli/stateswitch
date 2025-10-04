#!/bin/bash
#SBATCH --job-name=fmriprep_stateswitch
#SBATCH --time=24:00:00
#SBATCH --partition=parallel
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --output=/scratch4/choney1/zli230/stateswitch/data/logs/fmriprep_%j.out
#SBATCH --error=/scratch4/choney1/zli230/stateswitch/data/logs/fmriprep_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zli230@jh.edu

# Set up paths
BIDS_DIR="/scratch4/choney1/zli230/stateswitch/data/bids"
OUTPUT_DIR="/scratch4/choney1/zli230/stateswitch/data/derivatives"
WORK_DIR="/scratch4/choney1/zli230/stateswitch/work"
FMRIPREP_IMG="/scratch4/choney1/zli230/containers/fmriprep-25.2.0.sif"
FS_LICENSE="/vast/jchen230/containers/license.txt"

# Participant to process
PARTICIPANT_NUM=$1
PARTICIPANT=$(printf "%03d" ${PARTICIPANT_NUM})

echo "Starting fMRIPrep for sub-${PARTICIPANT}"

# Create work and derivatives directory if they don't exist
mkdir -p ${OUTPUT_DIR}
mkdir -p ${WORK_DIR}

# Run fMRIPrep
singularity run --cleanenv \
  -B /scratch4/choney1/zli230:/scratch4/choney1/zli230 \
  -B /vast/jchen230/containers:/vast/jchen230/containers:ro \
  ${FMRIPREP_IMG} \
  ${BIDS_DIR} \
  ${OUTPUT_DIR} \
  participant \
  --skip-bids-validation \
  --ignore slicetiming \
  --participant-label ${PARTICIPANT} \
  --work-dir ${WORK_DIR} \
  --fs-license-file ${FS_LICENSE} \
  --output-spaces anat MNI152NLin2009cAsym MNI152NLin6Asym:res-2 fsaverage6 \
  --stop-on-first-crash \
  --nthreads 48 --omp-nthreads 4 \
  --notrack \
  --subject-anatomical-reference unbiased \
  --use-syn-sdc \
  --force syn-sdc \
  --write-graph \
  --bold2anat-dof 6

echo "fMRIPrep completed for sub-${PARTICIPANT}"