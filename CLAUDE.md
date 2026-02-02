# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

StateSwitch is a neuroscience research project investigating neural mechanisms underlying semantic/cognitive state transitions. It analyzes behavioral and fMRI data from participants performing semantic verbal fluency (SVF) and ad-hoc categories (AHC) tasks.

## Tech Stack

- **Python 3.11** (requires >= 3.9)
- **Package Manager:** UV with lock file (`uv.lock`)
- **Core Libraries:** nilearn, nibabel, scikit-learn, scipy, pandas, matplotlib, jupyter

## Common Commands

```bash
# Install/sync dependencies
uv sync

# Run fMRI preprocessing (on Halibut server)
bash srcs/fmriprep/run_fmriprep_stateswitch_halibut.sh

# Run fMRI GLM analysis for a subject
python srcs/fmrianalysis/svf_parcel_glm.py --subject sub-001

# Run behavioral analysis
python srcs/behavior/svf_ahc_behavior.py --all

# Batch transcription with Whisper
python srcs/whisper/transcribe_batch.py

# Launch Jupyter for interactive analysis
jupyter notebook
```

## Code Architecture

### Configuration (`srcs/configs/config.py`)
Environment-aware configuration that auto-detects compute environment (ARCH HPC, local Mac, or lab servers Toronto/Halibut). All paths, subject lists, and constants centralized here.

### fMRI Analysis (`srcs/fmrianalysis/`)
- **`utils.py`** - Core utilities: event generation, surface/parcel data loading with caching
- **`svf_parcel_glm.py`** - Switch vs. Cluster GLM using Schaefer+Subcortical atlas
- **`localizers_contrast.py`** - Language, MD, ToM localizer contrasts
- **`plotting_config.py`** - Standardized plotting parameters (RdBu_r colormap, 300 dpi, fsaverage6)

### Behavioral Analysis (`srcs/behavior/`)
- **`svf_ahc_behavior.py`** - Primary analysis: inter-response times, semantic distances via GPT-2/USE embeddings

### Speech Pipeline (`srcs/whisper/`)
Whisper-based transcription with Montreal Forced Aligner integration for phoneme-level timing.

### fMRIPrep (`srcs/fmriprep/`)
BIDS conversion and fMRIPrep preprocessing scripts for each compute environment.

## Analysis Pipeline Flow

```
Raw NIFTI → BIDS (minimal_bids_converter.py) → fMRIPrep → Parcel Extraction (utils.py, cached) → GLM → Figures
Audio → WhisperX → Behavioral Analysis
```

## Key Conventions

- **All scripts import `from configs.config`** - never hardcode paths
- **Parcel time series cached to `.npz`** - cache keys include subject, session, task, atlas
- **Subject IDs:** sub-001, 003, 004, 006, 007, 008, 009 (7 subjects)
- **TR = 1.5 seconds**
- **Z-score within subjects** before group analysis
- **Visualization threshold:** z > 1.98