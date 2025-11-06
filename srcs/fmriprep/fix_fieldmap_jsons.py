#!/usr/bin/env python3
"""
Fix IntendedFor fields in fieldmap JSON files
"""
import json
import re
from pathlib import Path

# Try to import custom mappings
try:
    from config import FIELDMAP_INTENDED_FOR
except ImportError:
    FIELDMAP_INTENDED_FOR = {}

def fix_fieldmap_intendedfor(bids_dir: Path, dry_run: bool = False):
    """Fix IntendedFor fields in all fieldmap JSON files"""
    
    # Find all fieldmap JSON files
    for json_file in sorted(bids_dir.glob("sub-*/ses-*/fmap/*_epi.json")):
        # Extract subject and session from path
        sub_id = json_file.parts[-4]  # e.g., 'sub-001'
        ses_id = json_file.parts[-3]  # e.g., 'ses-01'
        sub_num = sub_id.replace('sub-', '')
        ses_num = ses_id.replace('ses-', '')
        
        # Find all functional files for this session
        func_dir = json_file.parent.parent / "func"
        if not func_dir.exists():
            continue
            
        bold_files = sorted(func_dir.glob("*_bold.nii.gz"))
        if not bold_files:
            continue
        
        # Check if we have custom mapping
        intended_for = []
        
        if (sub_num, ses_num) in FIELDMAP_INTENDED_FOR:
            # Extract acquisition label from filename
            acq_match = re.search(r'acq-(\d+)', json_file.name)
            if acq_match:
                acq_label = f"acq-{acq_match.group(1)}"
                mapping = FIELDMAP_INTENDED_FOR[(sub_num, ses_num)]
                
                if acq_label in mapping:
                    # Use custom mapping
                    patterns = mapping[acq_label]
                    for bold_file in bold_files:
                        if any(pattern in bold_file.name for pattern in patterns):
                            intended_for.append(f"{ses_id}/func/{bold_file.name}")
                    print(f"{sub_id}/{ses_id}: {json_file.name} -> {len(intended_for)} files (custom mapping)")
                else:
                    # No mapping for this acquisition, use all files
                    intended_for = [f"{ses_id}/func/{f.name}" for f in bold_files]
                    print(f"{sub_id}/{ses_id}: {json_file.name} -> {len(intended_for)} files (all)")
            else:
                # No acquisition label, use all files
                intended_for = [f"{ses_id}/func/{f.name}" for f in bold_files]
                print(f"{sub_id}/{ses_id}: {json_file.name} -> {len(intended_for)} files (all)")
        else:
            # No custom mapping for this subject/session, use all files
            intended_for = [f"{ses_id}/func/{f.name}" for f in bold_files]
            print(f"{sub_id}/{ses_id}: {json_file.name} -> {len(intended_for)} files (all)")
        
        # Update JSON
        with open(json_file, 'r') as f:
            metadata = json.load(f)
        
        if metadata.get('IntendedFor', []) == intended_for:
            print(f"  Already correct, skipping")
            continue
        
        if dry_run:
            print(f"  Would update with: {intended_for}")
        else:
            metadata['IntendedFor'] = intended_for
            with open(json_file, 'w') as f:
                json.dump(metadata, f, indent=4)
            print(f"  Updated")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: <bids_dir> [--dry-run]")
        sys.exit(1)
    
    bids_dir = Path(sys.argv[1])
    dry_run = "--dry-run" in sys.argv
    
    if dry_run:
        print("DRY RUN MODE - No files will be modified\n")
    
    fix_fieldmap_intendedfor(bids_dir, dry_run)