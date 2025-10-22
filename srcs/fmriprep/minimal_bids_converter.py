#!/usr/bin/env python3
"""
Minimal BIDS converter for stateswitch project
"""
import os
import re
import json
import shutil
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Set

# Import configs
from config import (
    RAW_DIR, BIDS_DIR, TR, TOTAL_READOUT_TIME, EFFECTIVE_ECHO_SPACING, 
    TASK_NAMES, PROCESS_ANAT, PROCESS_FMAP
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MinimalBIDSConverter:
    def __init__(self, dry_run: bool = False, truncate: bool = False):
        self.raw_dir = RAW_DIR
        self.bids_dir = BIDS_DIR
        self.task_names = TASK_NAMES
        self.tr = TR
        self.total_readout_time = TOTAL_READOUT_TIME
        self.effective_echo_spacing = EFFECTIVE_ECHO_SPACING
        self.dry_run = dry_run
        self.truncate = truncate
        self.process_anat = PROCESS_ANAT
        self.process_fmap = PROCESS_FMAP
        self.processed_files = set()  # Track processed files
        
        # Import custom mappings if available
        try:
            from config import CUSTOM_FUNCTIONAL_FILENAME_MAPPINGS
            self.custom_functional_mappings = CUSTOM_FUNCTIONAL_FILENAME_MAPPINGS
        except ImportError:
            self.custom_functional_mappings = {}
            
        try:
            from config import CUSTOM_FIELDMAP_FILENAME_MAPPINGS
            self.custom_fieldmap_mappings = CUSTOM_FIELDMAP_FILENAME_MAPPINGS
        except ImportError:
            self.custom_fieldmap_mappings = {}
            
        try:
            from config import FIELDMAP_INTENDED_FOR
            self.fieldmap_intended_for = FIELDMAP_INTENDED_FOR
        except ImportError:
            self.fieldmap_intended_for = {}

    def run(self):
        """Main conversion pipeline"""
        logger.info(f"Starting BIDS conversion: {self.raw_dir} -> {self.bids_dir}")
        if self.dry_run:
            logger.info("DRY RUN MODE - No files will be moved")
        
        if self.truncate:
            logger.info("TRUNCATE MODE - Will check for truncated (manually stopped) scans and remove invalid volumes")
            if self.dry_run:
                logger.info("  (Skipping actual volume checks in dry run mode)")
        
        # Check if raw directory exists
        if not self.raw_dir.exists():
            logger.error(f"Raw directory does not exist: {self.raw_dir}")
            return
        
        # Create BIDS root directory
        if not self.dry_run:
            self.bids_dir.mkdir(parents=True, exist_ok=True)
        
        # Create dataset_description.json
        self.create_dataset_description()
        
        # Parse and convert subjects
        subjects = self.parse_subject_folders()
        
        if not subjects:
            logger.warning("No subject folders found!")
            return
        
        logger.info(f"Found {len(subjects)} subjects")
        
        # Sort subjects by ID for consistent processing order
        for sub_num in sorted(subjects.keys(), key=int):
            sessions = subjects[sub_num]
            sub_id = f"{int(sub_num):03d}"
            logger.info(f"========== Processing subject {sub_id} with {len(sessions)} sessions ==========")
            
            for ses_idx, (date, session_path) in enumerate(sessions, 1):
                ses_id = f"{ses_idx:02d}"
                logger.info(f"  Processing sub-{sub_id}_ses-{ses_id} from {session_path.name}")
                
                try:
                    self.process_session(sub_id, ses_id, session_path)
                except Exception as e:
                    logger.error(f"  Failed to process {session_path}: {e}")
                    continue
        
        # Report unprocessed files
        self.report_unprocessed_files()
        
        logger.info("Conversion complete!")
    
    def parse_subject_folders(self) -> Dict[str, List[Tuple[datetime, Path]]]:
        """Parse raw directory and extract subject/session info"""
        subjects = {}
        
        pattern = re.compile(r'[Ss][Uu][Bb]_(\d+)_(\d{6})')
        
        for folder in self.raw_dir.iterdir():
            if not folder.is_dir():
                continue
                
            match = pattern.match(folder.name)
            if match:
                sub_num = match.group(1)
                date_str = match.group(2)
                
                # Convert YYMMDD to datetime for sorting
                try:
                    date = datetime.strptime(date_str, '%y%m%d')
                except ValueError:
                    logger.warning(f"Invalid date format in {folder.name}")
                    continue
                
                if sub_num not in subjects:
                    subjects[sub_num] = []
                
                subjects[sub_num].append((date, folder))
        
        # Sort sessions by date
        for sub in subjects:
            subjects[sub].sort(key=lambda x: x[0])
        
        return subjects
    
    def get_actual_volumes(self, nifti_gz_path: Path) -> tuple:
        """Get actual number of volumes in a NIfTI file using fsl"""
        # Get header info using fslhd
        result = subprocess.run(['fslhd', str(nifti_gz_path)], capture_output=True, text=True)
        
        # Parse dimensions, bitpix, and vox_offset
        dims = {}
        bitpix = None
        vox_offset = 352  # Default data offset
        
        for line in result.stdout.split('\n'):
            if line.startswith('vox_offset'):
                vox_offset = float(line.split()[1])
            elif line.startswith('dim'):
                parts = line.split()
                if len(parts) >= 2 and parts[0] in ['dim1', 'dim2', 'dim3', 'dim4']:
                    dims[parts[0]] = int(parts[1])
            elif line.startswith('bitpix'):
                bitpix = int(line.split()[1])
        
        expected_volumes = dims.get('dim4', 1)
        bytes_per_voxel = bitpix // 8
        voxels_per_volume = dims['dim1'] * dims['dim2'] * dims['dim3']
        bytes_per_volume = voxels_per_volume * bytes_per_voxel
        
        # Get uncompressed size
        result = subprocess.run(
            ['gunzip', '-c', str(nifti_gz_path)], 
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        )
        uncompressed_size = len(result.stdout)
        
        # Calculate actual volumes
        data_offset = int(vox_offset)  # vox_offset tells us where data starts
        data_size = uncompressed_size - data_offset
        actual_complete_volumes = data_size // bytes_per_volume

        return actual_complete_volumes, expected_volumes
        
    def process_session(self, sub_id: str, ses_id: str, session_path: Path):
        """Process a single session"""
        # Create BIDS directories
        dirs = {}
        for datatype in ['anat', 'func', 'fmap']:
            dir_path = self.bids_dir / f"sub-{sub_id}" / f"ses-{ses_id}" / datatype
            if not self.dry_run:
                dir_path.mkdir(parents=True, exist_ok=True)
            dirs[datatype] = dir_path
        
        # Process each data type
        if self.process_anat:
            self.process_anatomical_files(sub_id, ses_id, session_path, dirs['anat'])
        if self.process_fmap:
            self.process_fieldmap_files(sub_id, ses_id, session_path, dirs['fmap'])
        self.process_functional_files(sub_id, ses_id, session_path, dirs['func'])
    
    def process_anatomical_files(self, sub_id: str, ses_id: str, session_path: Path, anat_dir: Path):
        """Copy already renamed anatomical files"""
        # Look for T1w file - already defaced and renamed
        expected_name = f"sub-{sub_id}_ses-{ses_id}_T1w.nii.gz"
        
        for file in session_path.iterdir():
            if file.name == expected_name or (file.suffix == '.gz' and 'T1w' in file.name):
                self.processed_files.add(file)
                dest = anat_dir / expected_name
                
                if self.dry_run:
                    logger.info(f"    Would copy anatomical: {file.name} -> {expected_name}")
                else:
                    logger.info(f"    Copying anatomical: {file.name} -> {expected_name}")
                    shutil.copy2(file, dest)
                break
    
    def process_functional_files(self, sub_id: str, ses_id: str, session_path: Path, func_dir: Path):
        """Process functional MRI files with custom filename support"""
        # Get custom mappings if they exist
        sub_num = f"{int(sub_id):03d}"
        ses_num = f"{int(ses_id):02d}"
        custom_func_patterns = self.custom_functional_mappings.get((sub_num, ses_num), {})
        
        # Get all task names to look for
        if custom_func_patterns:
            # If custom patterns exist, use those task names
            all_tasks = list(custom_func_patterns.keys())
        else:
            # Otherwise use the standard task names
            all_tasks = self.task_names
        
        # Track runs for each task
        task_runs = {task: [] for task in all_tasks}
        
        # Find all potential functional files
        for file in sorted(session_path.iterdir()):
            if not file.suffix == '.gz':
                continue
            
            matched = False
            
            # Skip files that are already processed tasks
            skip_tasks = ['langloc', 'mdloc', 'tomloc', 'sceneprf']
            if any(f'fmri_{task}' in file.name.lower() for task in skip_tasks):
                self.processed_files.add(file)
                logger.info(f"    Skipping already processed task: {file.name}")
                continue
            
            # First try custom patterns for this subject/session
            if custom_func_patterns:
                for task, patterns in custom_func_patterns.items():
                    if any(pattern in file.name for pattern in patterns):
                        task_runs[task].append(file)
                        self.processed_files.add(file)
                        matched = True
                        logger.info(f"    Matched {file.name} to task '{task}' using custom pattern")
                        break
            
            # If no custom match, try standard patterns
            if not matched:
                for task in self.task_names:
                    if f'fmri_{task}' in file.name.lower():
                        task_runs[task].append(file)
                        self.processed_files.add(file)
                        matched = True
                        break
        
        # Process each task
        for task, files in task_runs.items():
            for run_idx, file in enumerate(files, 1):
                # Create BIDS filename
                if len(files) > 1:
                    bids_name = f"sub-{sub_id}_ses-{ses_id}_task-{task}_run-{run_idx}_bold.nii.gz"
                else:
                    bids_name = f"sub-{sub_id}_ses-{ses_id}_task-{task}_bold.nii.gz"
                
                dest = func_dir / bids_name

                # Check for truncation
                if self.truncate and not self.dry_run:
                    actual_volumes, expected_volumes = self.get_actual_volumes(file)
                    if actual_volumes < expected_volumes:
                        logger.warning(f"    Truncated scan detected for {file.name}: "
                                    f"expected {expected_volumes}, got {actual_volumes}")
                        
                        logger.info(f"    Truncating and copying: {file.name} -> {bids_name} "
                                    f"(keeping {actual_volumes} volumes)")
                        # Use fslroi to truncate and save directly to destination
                        subprocess.run(
                            ['fslroi', str(file), str(dest), '0', str(actual_volumes)], 
                            check=True
                        )
                        # Create JSON sidecar for truncated file
                        self.create_func_json(dest, task)
                        continue  # Skip the regular copy since we already handled it
                elif self.truncate and self.dry_run:
                    logger.info(f"    Would check truncation for: {file.name} -> {bids_name}")
                    continue

                # Regular copy for non-truncated files
                if self.dry_run:
                    logger.info(f"    Would copy functional: {file.name} -> {bids_name}")
                else:
                    logger.info(f"    Copying functional: {file.name} -> {bids_name}")
                    shutil.copy2(file, dest)
                    # Create JSON sidecar
                    self.create_func_json(dest, task)
    
    def process_fieldmap_files(self, sub_id: str, ses_id: str, session_path: Path, fmap_dir: Path):
        """Process fieldmap files"""
        import re
        
        # Get custom patterns if they exist
        sub_num = f"{int(sub_id):03d}"
        ses_num = f"{int(ses_id):02d}"
        custom_fmap_patterns = self.custom_fieldmap_mappings.get((sub_num, ses_num), {})
        
        # Look for fieldmap files
        fmap_files = []
        fmap_directions = {}  # Store direction for each file
        
        for file in session_path.iterdir():
            if file.suffix != '.gz':
                continue
                
            # Check custom patterns first
            if custom_fmap_patterns:
                for direction, patterns in custom_fmap_patterns.items():
                    if any(pattern in file.name for pattern in patterns):
                        fmap_files.append(file)
                        fmap_directions[file] = direction
                        self.processed_files.add(file)
                        logger.info(f"    Found {direction} fieldmap using custom pattern: {file.name}")
                        break
            # Then check standard pattern
            elif 'fieldmap' in file.name.lower():
                fmap_files.append(file)
                self.processed_files.add(file)
                # Determine direction from standard naming
                if '_a_' in file.name:
                    fmap_directions[file] = 'AP'
                else:
                    fmap_directions[file] = 'PA'
        
        if not fmap_files:
            return
        
        # Extract scan number from filename for proper ordering
        def get_scan_number(filename):
            """Extract scan number from filename like 'sub_09_250928_wip_fmri_fieldmap_a_3_1'"""
            # Look for the number after 'fieldmap_a_' or 'fieldmap_p_'
            match = re.search(r'fieldmap_[ap]_(\d+)_', filename.name.lower())
            if match:
                return int(match.group(1))
            # For custom named files, try to extract any number
            match = re.search(r'_(\d+)', filename.name)
            if match:
                return int(match.group(1))
            return 0
        
        # Sort by scan number to ensure chronological order
        fmap_files.sort(key=get_scan_number)
        
        # Check if we have multiple pairs
        num_pairs = len(fmap_files) // 2
        
        if num_pairs > 1:
            logger.info(f"    Found {num_pairs} fieldmap pairs")
            # Log the order for verification
            for i, f in enumerate(fmap_files):
                scan_num = get_scan_number(f)
                logger.info(f"      {f.name} (scan number: {scan_num})")
            
            # Check if we have custom mapping
            if (sub_num, ses_num) not in self.fieldmap_intended_for:
                logger.warning(f"    Multiple fieldmaps but no custom mapping!")
                logger.warning(f"    Add to FIELDMAP_INTENDED_FOR in config.py:")
                logger.warning(f"      ('{sub_num}', '{ses_num}'): {{")
                logger.warning(f"          'acq-1': ['run-1', 'run-2'],")
                logger.warning(f"          'acq-2': ['run-3', 'run-4'],")
                logger.warning(f"      }}")
        
        # Process fieldmaps (assuming pairs of AP/PA)
        for idx, file in enumerate(fmap_files):
            # Get direction from our mapping
            direction = fmap_directions.get(file)
            if not direction:
                logger.warning(f"    Could not determine direction for {file.name}")
                continue
            
            # Set phase encoding direction
            pe_dir = 'j-' if direction == 'AP' else 'j'
            
            # Add acquisition number if multiple pairs exist
            if num_pairs > 1:
                acq_num = (idx // 2) + 1  # Which pair is this?
                bids_name = f"sub-{sub_id}_ses-{ses_id}_acq-{acq_num}_dir-{direction}_epi.nii.gz"
            else:
                bids_name = f"sub-{sub_id}_ses-{ses_id}_dir-{direction}_epi.nii.gz"
            
            dest = fmap_dir / bids_name
            
            # Check if file already exists (important for re-running)
            if dest.exists() and not self.dry_run:
                logger.info(f"    Skipping existing fieldmap: {bids_name}")
                continue
            
            if self.dry_run:
                logger.info(f"    Would copy fieldmap: {file.name} -> {bids_name}")
            else:
                logger.info(f"    Copying fieldmap: {file.name} -> {bids_name}")
                shutil.copy2(file, dest)
                
                # Create JSON sidecar
                self.create_fieldmap_json(dest, pe_dir, sub_id, ses_id)
    
    def create_func_json(self, nifti_path: Path, task: str):
        """Create JSON sidecar for functional file"""
        json_path = nifti_path.with_suffix('').with_suffix('.json')
        
        metadata = {
            "TaskName": task,
            "RepetitionTime": self.tr
        }
        
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=4)
    
    def create_fieldmap_json(self, nifti_path: Path, pe_dir: str, sub_id: str, ses_id: str):
        """Create JSON sidecar for fieldmap file"""
        json_path = nifti_path.with_suffix('').with_suffix('.json')
        
        # Extract numbers for config lookup
        sub_num = f"{int(sub_id):03d}"
        ses_num = f"{int(ses_id):02d}"
        
        # Parse acquisition label from filename
        acq_match = re.search(r'acq-(\d+)', nifti_path.name)
        acq_label = f"acq-{acq_match.group(1)}" if acq_match else None
        
        # Find all functional files for IntendedFor
        intended_for = []
        func_dir = self.bids_dir / f"sub-{sub_id}" / f"ses-{ses_id}" / "func"
        
        if func_dir.exists():
            bold_files = sorted(func_dir.glob("*_bold.nii.gz"))
            
            # Check if we have custom mapping for multiple fieldmaps
            if acq_label and (sub_num, ses_num) in self.fieldmap_intended_for:
                # Use custom mapping
                custom_mapping = self.fieldmap_intended_for[(sub_num, ses_num)]
                if acq_label in custom_mapping:
                    target_runs = custom_mapping[acq_label]
                    for bold_file in bold_files:
                        # Check if any of the target identifiers are in the filename
                        if any(target in bold_file.name for target in target_runs):
                            intended_for.append(f"ses-{ses_id}/func/{bold_file.name}")
            else:
                # Default: apply to all functional runs
                for bold_file in bold_files:
                    intended_for.append(f"ses-{ses_id}/func/{bold_file.name}")
        
        metadata = {
            "PhaseEncodingDirection": pe_dir,
            "TotalReadoutTime": self.total_readout_time,
            "EffectiveEchoSpacing": self.effective_echo_spacing,
            "IntendedFor": intended_for
        }
        
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=4)
    
    def report_unprocessed_files(self):
        """Report all .nii.gz files that were not processed"""
        logger.info("\n========== Unprocessed Files Report ==========")
        
        unprocessed_count = 0
        
        # Walk through all subject folders
        for subject_folder in sorted(self.raw_dir.iterdir()):
            if not subject_folder.is_dir():
                continue
            
            unprocessed_in_session = []
            
            # Check all .nii.gz files in this session
            for file in subject_folder.iterdir():
                if file.suffix == '.gz' and file not in self.processed_files:
                    unprocessed_in_session.append(file.name)
            
            if unprocessed_in_session:
                logger.info(f"\n{subject_folder.name}:")
                for filename in sorted(unprocessed_in_session):
                    logger.info(f"  - {filename}")
                    unprocessed_count += len(unprocessed_in_session)
        
        if unprocessed_count == 0:
            logger.info("All .nii.gz files were processed!")
        else:
            logger.info(f"\nTotal unprocessed files: {unprocessed_count}")
            logger.info("Please verify these files are not needed for BIDS conversion")
    
    def create_dataset_description(self):
        """Create dataset_description.json"""
        if self.dry_run:
            logger.info("Would create dataset_description.json")
            return
            
        description = {
            "Name": "State Switch fMRI Dataset",
            "BIDSVersion": "1.6.0",
            "Authors": ["Name"],
            "Acknowledgements": "Thanks",
            "Funding": ["Grant Number"],
            "DatasetDOI": "doi:10.0000/00000"
        }
        
        desc_path = self.bids_dir / 'dataset_description.json'
        with open(desc_path, 'w') as f:
            json.dump(description, f, indent=4)
        
        logger.info("Created dataset_description.json")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert raw fMRI data to BIDS format')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Preview what would be done without moving files')
    parser.add_argument('--truncate', action='store_true',
                       help='Check for truncated scans and remove invalid volumes')
    
    args = parser.parse_args()
    
    # Create converter and run
    converter = MinimalBIDSConverter(dry_run=args.dry_run, truncate=args.truncate)
    converter.run()


if __name__ == "__main__":
    main()