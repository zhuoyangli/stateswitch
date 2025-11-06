"""
Shared configuration for stateswitch project
"""
from pathlib import Path
import os
import socket

# Get username
USERNAME = os.getenv('USER', 'unknown')

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Detect environment by checking paths
if Path('/scratch4').exists(): # rockfish/ARCH
    PROJECT_ROOT = Path(f"/scratch4/choney1/zli230/stateswitch")
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DIR = DATA_DIR / "raw"
    BIDS_DIR = DATA_DIR / "bids"
    DERIVATIVES_DIR = DATA_DIR / "derivatives"
    FIGS_DIR = PROJECT_ROOT / "figs"
elif Path('/Users/gioli').exists(): # local mac
    PROJECT_ROOT = Path(f"/Users/gioli/projects/stateswitch")
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DIR = DATA_DIR / "raw"
    BIDS_DIR = DATA_DIR / "bids"
    DERIVATIVES_DIR = DATA_DIR / "derivatives"
    FIGS_DIR = PROJECT_ROOT / "figs"
elif Path('/home/zli230').exists(): # lab server
    hostname = socket.gethostname()
    if 'honeyserve' in hostname: # old lab server (toronto)
        PROJECT_ROOT = Path(f"/home/zli230/projects/stateswitch")
        DATA_DIR = PROJECT_ROOT / "data"
        RAW_DIR = Path(f"/mri_transfer/gio/stateswitch")
        BIDS_DIR = DATA_DIR / "bids"
        DERIVATIVES_DIR = DATA_DIR / "derivatives"
        FIGS_DIR = PROJECT_ROOT / "figs"
    elif 'pbs-jcch-gpu' in hostname: # new lab server (Halibut)
        PROJECT_ROOT = Path(f"/home/zli230/projects/stateswitch")
        DATA_DIR = Path(f"/home/Datasets/stateswitch")
        RAW_DIR = DATA_DIR / "raw"
        BIDS_DIR = DATA_DIR / "bids"
        DERIVATIVES_DIR = DATA_DIR / "derivatives"
        FIGS_DIR = PROJECT_ROOT / "figs"

# Common parameters
TR = 1.5  # Repetition time in seconds