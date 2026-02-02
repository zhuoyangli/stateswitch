#!/usr/bin/env python3
"""
Fix IntendedFor and PhaseEncodingDirection fields in fieldmap JSON files
"""
import json
import re
from pathlib import Path

# Try to import custom mappings
try:
    from config import FIELDMAP_INTENDED_FOR
except ImportError:
    FIELDMAP_INTENDED_FOR = {}

def fix_fieldmap(bids_dir: Path, dry_run: bool = False):
    """Fix IntendedFor and PhaseEncodingDirection fields in all fieldmap JSON files"""
    
    ## fix IntendedFor fields
    
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
        
        ## fix PhaseEncodingDirection fields
        if '_dir-' not in json_file.name:
            raise ValueError(f"Missing 'dir-' in filename: {json_file}")
        
        dir_filename = json_file.name.split('_dir-')[1].split('_')[0]
        pe_direction = 'j' if dir_filename == 'AP' else 'j-'
        
        # Update JSON
        with open(json_file, 'r') as f:
            metadata = json.load(f)

        # Check what needs to be updated
        needs_intended_for = metadata.get('IntendedFor', []) != intended_for
        needs_pe_direction = metadata.get('PhaseEncodingDirection') != pe_direction
        
        if not needs_intended_for and not needs_pe_direction:
            print(f"  Already correct, skipping")
            continue
        
        # Build status message
        updates_needed = []
        if needs_intended_for:
            updates_needed.append("IntendedFor")
        if needs_pe_direction:
            updates_needed.append("PhaseEncodingDirection")
        
        if dry_run:
            print(f"  Would update: {', '.join(updates_needed)}")
            if needs_intended_for:
                print(f"    IntendedFor: {metadata.get('IntendedFor', [])} -> {intended_for}")
            if needs_pe_direction:
                print(f"    PhaseEncodingDirection: {metadata.get('PhaseEncodingDirection')} -> {pe_direction}")
        else:
            metadata['IntendedFor'] = intended_for
            metadata['PhaseEncodingDirection'] = pe_direction
            with open(json_file, 'w') as f:
                json.dump(metadata, f, indent=4)
            print(f"  Updated: {', '.join(updates_needed)}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <bids_dir> [--dry-run]")
        sys.exit(1)
    
    bids_dir = Path(sys.argv[1])
    dry_run = "--dry-run" in sys.argv
    
    if dry_run:
        print("DRY RUN MODE - No files will be modified\n")
    
    fix_fieldmap(bids_dir, dry_run)