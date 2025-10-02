#!/usr/bin/env python3
"""
Create fieldmap JSON sidecar files with IntendedFor fields
"""
import os
import json
import glob
from pathlib import Path
from config import BIDS_DIR, TOTAL_READOUT_TIME, EFFECTIVE_ECHO_SPACING

def create_fieldmap_jsons(bids_dir):
    """Create JSON files for all fieldmaps with IntendedFor fields"""
    
    # Find all fieldmap directories
    fmap_dirs = glob.glob(os.path.join(bids_dir, "sub-*/ses-*/fmap"))
    
    for fmap_dir in fmap_dirs:
        # Extract subject and session
        parts = fmap_dir.split(os.sep)
        sub = parts[-3]  # sub-XXX
        ses = parts[-2]  # ses-XX
        
        # Find functional scans for this session
        func_dir = os.path.join(bids_dir, sub, ses, "func")
        if os.path.exists(func_dir):
            # Get all BOLD scans
            bold_files = glob.glob(os.path.join(func_dir, "*_bold.nii.gz"))
            
            # Create IntendedFor list (relative paths from subject directory)
            intended_for = []
            for bold in bold_files:
                # Path relative to subject directory
                rel_path = os.path.relpath(bold, os.path.join(bids_dir, sub))
                intended_for.append(rel_path)
        
        # Find fieldmap files
        ap_file = glob.glob(os.path.join(fmap_dir, "*_dir-AP_epi.nii.gz"))
        pa_file = glob.glob(os.path.join(fmap_dir, "*_dir-PA_epi.nii.gz"))
        
        # Create JSON for AP fieldmap
        if ap_file and intended_for:
            json_file = ap_file[0].replace('.nii.gz', '.json')
            json_data = {
                "PhaseEncodingDirection": "j-",
                "TotalReadoutTime": TOTAL_READOUT_TIME,
                "EffectiveEchoSpacing": EFFECTIVE_ECHO_SPACING,
                "IntendedFor": intended_for
            }
            
            with open(json_file, 'w') as f:
                json.dump(json_data, f, indent=2)
            print(f"Created: {json_file}")
        
        # Create JSON for PA fieldmap  
        if pa_file and intended_for:
            json_file = pa_file[0].replace('.nii.gz', '.json')
            json_data = {
                "PhaseEncodingDirection": "j",
                "TotalReadoutTime": TOTAL_READOUT_TIME,
                "EffectiveEchoSpacing": EFFECTIVE_ECHO_SPACING,
                "IntendedFor": intended_for
            }
            
            with open(json_file, 'w') as f:
                json.dump(json_data, f, indent=2)
            print(f"Created: {json_file}")

if __name__ == "__main__":
    create_fieldmap_jsons(BIDS_DIR)
    print("Fieldmap JSON creation complete!")