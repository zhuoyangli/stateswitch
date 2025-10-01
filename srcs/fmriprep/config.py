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
TASK_NAMES = ["langloc1", "langloc2", "mdloc1", "mdloc2", "tomloc1", "tomloc2", "sceneprf"] 
PROCESS_ANAT = True
PROCESS_FMAP = True