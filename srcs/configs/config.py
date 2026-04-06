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
    CACHE_DIR = DATA_DIR / "cache"
elif Path('/Users/gioli').exists(): # local mac
    PROJECT_ROOT = Path(f"/Users/gioli/projects/stateswitch")
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DIR = DATA_DIR / "raw"
    BIDS_DIR = DATA_DIR / "bids"
    DERIVATIVES_DIR = DATA_DIR / "derivatives"
    FIGS_DIR = PROJECT_ROOT / "figs"
    CACHE_DIR = DATA_DIR / "cache"
elif Path('/home/zli230').exists(): # lab server
    hostname = socket.gethostname()
    if 'honeyserve' in hostname: # old lab server (toronto)
        PROJECT_ROOT = Path(f"/home/zli230/projects/stateswitch")
        DATA_DIR = PROJECT_ROOT / "data"
        RAW_DIR = Path(f"/mri_transfer/gio/stateswitch")
        BIDS_DIR = DATA_DIR / "bids"
        DERIVATIVES_DIR = DATA_DIR / "derivatives"
        FIGS_DIR = PROJECT_ROOT / "figs"
        CACHE_DIR = DATA_DIR / "cache"
    elif 'pbs-jcch-gpu' in hostname: # new lab server (Halibut)
        PROJECT_ROOT = Path(f"/home/zli230/projects/stateswitch")
        DATA_DIR = Path(f"/home/datasets/stateswitch")
        RAW_DIR = DATA_DIR / "raw"
        BIDS_DIR = DATA_DIR / "bids"
        DERIVATIVES_DIR = DATA_DIR / "derivatives"
        FIGS_DIR = PROJECT_ROOT / "figs"
        CACHE_DIR = Path(f"/home/zli230/projects/stateswitch/data/cache")

# Common parameters
TR = 1.5  # Repetition time in seconds

SUBJECT_IDS = ['sub-001', 'sub-003', 'sub-004', 'sub-006', 'sub-007', 'sub-008', 'sub-009']

ANALYSIS_CACHE_DIR = CACHE_DIR / 'analyses'

FILMFEST_SUBJECTS = {
    'sub-003': 'ses-10',
    'sub-004': 'ses-10',
    'sub-006': 'ses-08',
    'sub-007': 'ses-08',
    'sub-008': 'ses-08',
    'sub-009': 'ses-07',
}

MOVIE_INFO = [
    {'id': 1,  'file': 'FilmFest_01_CMIYC_Segments.xlsx',         'task': 'filmfest1'},
    {'id': 2,  'file': 'FilmFest_02_The_Record_Segments.xlsx',     'task': 'filmfest1'},
    {'id': 3,  'file': 'FilmFest_03_The_Boyfriend_Segments.xlsx',  'task': 'filmfest1'},
    {'id': 4,  'file': 'FilmFest_04_The_Shoe_Segments.xlsx',       'task': 'filmfest1'},
    {'id': 5,  'file': 'FilmFest_05_Keith_Reynolds_Segments.xlsx', 'task': 'filmfest1'},
    {'id': 6,  'file': 'FilmFest_06_The_Rock_Segments.xlsx',       'task': 'filmfest2'},
    {'id': 7,  'file': 'FilmFest_07_The_Prisoner_Segments.xlsx',   'task': 'filmfest2'},
    {'id': 8,  'file': 'FilmFest_08_The_Black_Hole_Segments.xlsx', 'task': 'filmfest2'},
    {'id': 9,  'file': 'FilmFest_09_Post-it_Love_Segments.xlsx',   'task': 'filmfest2'},
    {'id': 10, 'file': 'FilmFest_10_Bus_Stop_Segments.xlsx',       'task': 'filmfest2'},
]