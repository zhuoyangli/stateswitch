#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --mem=8G
#SBATCH --partition=parallel
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --account=choney1_gpu
#SBATCH --qos=normal
#SBATCH --output=/scratch4/choney1/zli230/stateswitch/data/logs/pull_fmriprep_%j.out
#SBATCH --error=/scratch4/choney1/zli230/stateswitch/data/logs/pull_fmriprep_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zli230@jh.edu

cd /scratch4/choney1/zli230/containers/

ml singularity

singularity pull --name fmriprep-25.2.3.sif docker://nipreps/fmriprep:25.2.3