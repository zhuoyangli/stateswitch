"""
fMRIPrep-specific configuration
"""
import sys
from pathlib import Path

# Add project root to path to import shared config
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import RAW_DIR, BIDS_DIR

# acquisition parameters
TR = 1.5
TOTAL_READOUT_TIME = 0.0768
EFFECTIVE_ECHO_SPACING = 0.00069

# Processing settings
# TASK_NAMES = ["langloc1", "langloc2", "mdloc1", "mdloc2", "tomloc1", "tomloc2", "sceneprf"] 
TASK_NAMES = ["fta", "ahc", "svf", "tst"]
PROCESS_ANAT = True
PROCESS_FMAP = True

CUSTOM_FUNCTIONAL_FILENAME_MAPPINGS = {
    ('001', '02'): {
        'undertheinfluence': ['stories_6'],
        'beneaththemushroomcloud': ['stories_7'],
        'christmas1940': ['stories_8'],
        'notontheusualtour': ['stories_9'],
        'shoppinginchina': ['stories_10'],
        'vixen': ['stories_11']
    },
    ('001', '03'): {
        'adollshouse': ['stories_8'],
        'adventuresinsayingyes': ['stories_9'],
        'buck': ['stories_10'],
        'inamoment': ['stories_11'],
        'theclosetthatateeverything': ['stories_12'],
        'wheretheressmoke': ['stories_13']
    },
    ('003', '01'): {
        'treasureisland': ['stories_5'],
        'undertheinfluence': ['stories_6'],
        'penpal': ['stories_7'],
        'odetostepfather': ['stories_8'],
        'swimming': ['stories_9']
    },
    ('003', '02'): {
        'audloc': ['sub03_5'],
        'howtodraw': ['sub03_6'],
        'beneaththemushroomcloud': ['sub03_7'],
        'christmas1940': ['sub03_9'],
        'notontheusualtour': ['sub03_10'],
        'shoppinginchina': ['sub03_11'],
        'vixen': ['sub03_12']
    },
    ('003', '03'): {
        'fta': ['stories_1'],
        'ahc': ['stories_2'],
        'svf': ['stories_3'],
        'sb': ['stories_4']
    },
    ('003', '04'): {
        'adollshouse': ['stories_6'],
        'adventuresinsayingyes': ['stories_7'],
        'buck': ['stories_8'],
        'inamoment': ['stories_9'],
        'theclosetthatateeverything': ['stories_10'],
        'wheretheressmoke': ['stories_11']
    },
    ('004', '01'): {
        'treasureisland': ['stories_6'],
        'undertheinfluence': ['stories_7'],
        'penpal': ['stories_8'],
        'odetostepfather': ['stories_9'],
        'swimming': ['stories_10'],
        'howtodraw': ['stories_11']
    },
    ('004', '02'): {
        'audloc': ['local_8'],
        'breakingupintheageofgoogle': ['stories_9'],
        'beneaththemushroomcloud': ['stories_10'],
        'christmas1940': ['stories_12']
    },
    ('004', '03'): {
        'audloc': ['local_5'],
        'breakingupintheageofgoogle': ['stories_6'],
        'beneaththemushroomcloud': ['stories_7'],
        'christmas1940': ['stories_8'],
        'notontheusualtour': ['stories_9'],
        'shoppinginchina': ['stories_10'],
        'vixen': ['stories_11']
    },
    ('004', '04'): {
        'fta': ['task1'],
        'ahc': ['task2'],
        'svf': ['task3'],
        'sb': ['task4']
    },
    ('006', '01'): {
        'treasureisland': ['stories_5'],
        'undertheinfluence': ['stories_6'],
        'penpal': ['stories_7'],
        'odetostepfather': ['stories_8'],
        'swimming': ['stories_9'],
        'howtodraw': ['stories_10']
    },
    ('006', '02'): {
        'audloc': ['local_6'],
        'breakingupintheageofgoogle': ['stories_7'],
        'beneaththemushroomcloud': ['stories_8'],
        'christmas1940': ['stories_9'],
        'notontheusualtour': ['stories_10'],
        'shoppinginchina': ['stories_11'],
        'vixen': ['stories_12']
    },
    ('006', '03'): {
        'adollshouse': ['stories_6'],
        'adventuresinsayingyes': ['stories_7'],
        'buck': ['stories_8'],
        'inamoment': ['stories_9'],
        'theclosetthatateeverything': ['stories_10'],
        'wheretheressmoke': ['stories_11']
    },
    ('007', '01'): {
        'audloc': ['sub7_6'],
        'treasureisland': ['sub7_7'],
        'undertheinfluence': ['sub7_11'],
        'penpal': ['sub7_12'],
        'odetostepfather': ['sub7_13'],
        'swimming': ['sub7_14'],
        'howtodraw': ['sub7_15']
    },
    ('007', '02'): {
        'audloc': ['local_5'],
        'breakingupintheageofgoogle': ['stories_6'],
        'beneaththemushroomcloud': ['stories_8'],
        'christmas1940': ['stories_9'],
        'notontheusualtour': ['stories_10'],
        'shoppinginchina': ['stories_11'],
        'vixen': ['stories_12']
    },
    ('007', '03'): {
        'adollshouse': ['stories_7'],
        'adventuresinsayingyes': ['stories_8'],
        'buck': ['stories_9'],
        'inamoment': ['stories_10'],
        'theclosetthatateeverything': ['stories_11'],
        'wheretheressmoke': ['stories_12']
    },
    ('008', '01'): {
        'treasureisland': ['stories_5'],
        'undertheinfluence': ['stories_6'],
        'penpal': ['stories_7'],
        'odetostepfather': ['stories_8'],
        'swimming': ['stories_9'],
        'howtodraw': ['stories_10']
    },
    ('008', '02'): {
        'breakingupintheageofgoogle': ['stories_6'],
        'beneaththemushroomcloud': ['stories_7'],
        'christmas1940': ['stories_8'],
        'notontheusualtour': ['stories_9'],
        'shoppinginchina': ['stories_10'],
        'vixen': ['stories_11']
    },
    ('008', '03'): {
        'adollshouse': ['stories_5'],
        'adventuresinsayingyes': ['stories_6'],
        'buck': ['stories_7'],
        'inamoment': ['stories_8'],
        'theclosetthatateeverything': ['stories_9'],
        'wheretheressmoke': ['stories_10']
    },
    ('009', '01'): {
        'treasureisland': ['stories_9'],
        'undertheinfluence': ['stories_10'],
        'penpal': ['stories_11'],
        'odetostepfather': ['stories_12'],
        'swimming': ['stories_13'],
        'howtodraw': ['stories_14']
    },
    ('009', '02'): {
        'breakingupintheageofgoogle': ['stories_6'],
        'beneaththemushroomcloud': ['stories_7'],
        'christmas1940': ['stories_8'],
        'notontheusualtour': ['stories_9'],
        'shoppinginchina': ['stories_10'],
        'vixen': ['stories_11']
    },
    ('009', '03'): {
        'adollshouse': ['stories_5'],
        'adventuresinsayingyes': ['stories_6'],
        'buck': ['stories_7'],
        'inamoment': ['stories_11'],
        'theclosetthatateeverything': ['stories_12'],
        'wheretheressmoke': ['stories_13']
    },
}

CUSTOM_FIELDMAP_FILENAME_MAPPINGS = {
    ('003', '02'): {
        'PA': ['sub03_2'],
        'AP': ['sub03_3']
    },
    ('007', '01'): {
        'PA': ['sub7_2'],
        'AP': ['sub7_3']
    }
}

FIELDMAP_INTENDED_FOR = {
    ('003', '06'): {
        'acq-1': ['fta', 'ahc'],
        'acq-2': ['svf', 'tst']
    },
    ('009', '03'): {
        'acq-1': ['adollshouse', 'adventuresinsayingyes', 'buck'],
        'acq-2': ['inamoment', 'theclosetthatateeverything', 'wheretheressmoke']
    }
}