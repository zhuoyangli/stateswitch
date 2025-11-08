#!/usr/bin/env python3
"""
Add missing metadata fields to BOLD JSON files in BIDS dataset.
Usage: python update_bold_jsons.py /path/to/bids
"""

import json
import os
import sys
from pathlib import Path

def update_bold_jsons(bids_dir):
    """Add missing fields to all *_bold.json files in BIDS directory."""
    
    # Fields to add
    new_fields = {
        "PhaseEncodingDirection": "j-",
        "EffectiveEchoSpacing": 0.00069,
        "TotalReadoutTime": 0.0768,
        "EchoTime": 0.03,
        "SliceThickness": 2,
        "MultibandAccelerationFactor": 4,
        "FlipAngle": 52,
        "Manufacturer": "Philips",
        "InstitutionName": "Kennedy Krieger Institute"
    }
    
    # Find all bold.json files
    bids_path = Path(bids_dir)
    json_files = list(bids_path.rglob("*_bold.json"))
    
    if not json_files:
        print(f"No *_bold.json files found in {bids_dir}")
        return
    
    print(f"Found {len(json_files)} BOLD JSON files\n")
    
    # Update each file
    for json_file in json_files:
        try:
            # Read existing content
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Check what's missing
            missing_fields = []
            for field, value in new_fields.items():
                if field not in data:
                    missing_fields.append(field)
                    data[field] = value
            
            # Only write if something was added
            if missing_fields:
                with open(json_file, 'w') as f:
                    json.dump(data, f, indent=2)
                print(f"✓ Updated {json_file.name} - Added: {', '.join(missing_fields)}")
            else:
                print(f"- Skipped {json_file.name} - All fields present")
            
        except Exception as e:
            print(f"✗ Error processing {json_file}: {e}")
    
    print("\nDone!")

def main():
    if len(sys.argv) != 2:
        print("Usage: python update_bold_jsons.py /path/to/bids")
        sys.exit(1)
    
    bids_dir = sys.argv[1]
    if not os.path.exists(bids_dir):
        print(f"Error: {bids_dir} does not exist")
        sys.exit(1)
    
    update_bold_jsons(bids_dir)

if __name__ == "__main__":
    main()