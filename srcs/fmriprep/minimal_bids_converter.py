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
from typing import List, Dict, Tuple

# Import configs
from config import RAW_DIR, BIDS_DIR, TR, TOTAL_READOUT_TIME, EFFECTIVE_ECHO_SPACING, TASK_NAMES, PROCESS_ANAT, PROCESS_FMAP

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

    def run(self):
        """Main conversion pipeline"""
        logger.info(f"Starting BIDS conversion: {self.raw_dir} -> {self.bids_dir}")
        if self.dry_run:
            logger.info("DRY RUN MODE - No files will be moved")
        
        if self.truncate:
            logger.info("TRUNCATE MODE - Will check for truncated (manually stopped) scans and remove invalid volumes")
        
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
        
        for sub_num, sessions in subjects.items():
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
                dest = anat_dir / expected_name
                
                if self.dry_run:
                    logger.info(f"    Would copy anatomical: {file.name} -> {expected_name}")
                else:
                    logger.info(f"    Copying anatomical: {file.name} -> {expected_name}")
                    shutil.copy2(file, dest)
                break
    
    def process_functional_files(self, sub_id: str, ses_id: str, session_path: Path, func_dir: Path):
        """Process functional MRI files"""
        # Track runs for each task
        task_runs = {task: [] for task in self.task_names}
        
        # Find all functional files
        for file in sorted(session_path.iterdir()):
            if not file.suffix == '.gz':
                continue
                
            # Check each task name
            for task in self.task_names:
                if f'fmri_{task}' in file.name.lower():
                    task_runs[task].append(file)
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
                if self.truncate:
                    actual_volumes, expected_volumes = self.get_actual_volumes(file)
                    if actual_volumes < expected_volumes:
                        logger.warning(f"    Truncated scan detected for {file.name}: "
                                    f"expected {expected_volumes}, got {actual_volumes}")
                        
                        if self.dry_run:
                            logger.info(f"    Would truncate and copy: {file.name} -> {bids_name} "
                                    f"(keeping {actual_volumes} volumes)")
                        else:
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
        # Look for fieldmap files
        fmap_files = []
        for file in session_path.iterdir():
            if 'fieldmap' in file.name.lower() and file.suffix == '.gz':
                fmap_files.append(file)
        
        if not fmap_files:
            return
        
        # Sort to ensure consistent ordering
        fmap_files.sort()
        
        # Process fieldmaps (assuming pairs of AP/PA)
        for idx, file in enumerate(fmap_files):
            # Determine phase encoding direction based on filename or index
            if '_a_' in file.name or idx % 2 == 0:
                direction = 'AP'
                pe_dir = 'j-'
            else:
                direction = 'PA'
                pe_dir = 'j'
            
            bids_name = f"sub-{sub_id}_ses-{ses_id}_dir-{direction}_epi.nii.gz"
            dest = fmap_dir / bids_name
            
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
        
        # Find all functional files for IntendedFor
        intended_for = []
        func_dir = self.bids_dir / f"sub-{sub_id}" / f"ses-{ses_id}" / "func"
        
        if func_dir.exists():
            for func_file in func_dir.iterdir():
                if func_file.name.endswith('_bold.nii.gz'):
                    # IntendedFor uses relative path from subject directory
                    intended_for.append(f"ses-{ses_id}/func/{func_file.name}")
        
        metadata = {
            "PhaseEncodingDirection": pe_dir,
            "TotalReadoutTime": self.total_readout_time,
            "EffectiveEchoSpacing": self.effective_echo_spacing,
            "IntendedFor": intended_for
        }
        
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=4)
    
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