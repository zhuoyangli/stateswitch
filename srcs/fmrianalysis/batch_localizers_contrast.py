#!/usr/bin/env python
"""
Batch processing script for localizer GLM analyses
"""
import subprocess
from pathlib import Path
from datetime import datetime
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import FIGS_DIR, PROJECT_ROOT

# Configuration
FSAVERAGE = 'fsaverage6'
BATCH_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = FIGS_DIR / 'localizers' / f'batch_{BATCH_TIMESTAMP}'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SCRIPT_PATH = PROJECT_ROOT / 'srcs' / 'fmrianalysis' / 'localizers_contrast.py'

def generate_jobs():
    """Generate analysis jobs based on subject-specific patterns"""
    jobs = []
    
    # # Standard pattern: all tasks in ses-04
    # for subject in ['sub-001', 'sub-006', 'sub-008', 'sub-009']:
    #     for task in ['langloc', 'mdloc', 'tomloc']:
    #         for run in [1, 2]:
    #             jobs.append((subject, 'ses-04', task, run))
    
    # # Special case: sub-003
    # # langloc in ses-05
    # for run in [1, 2]:
    #     jobs.append(('sub-003', 'ses-05', 'langloc', run))
    # # other tasks in ses-08
    # for task in ['mdloc', 'tomloc']:
    #     for run in [1, 2]:
    #         jobs.append(('sub-003', 'ses-08', task, run))

    # Special case: sub-004
    # langloc in ses-06
    for run in [1, 2]:
        jobs.append(('sub-004', 'ses-06', 'langloc', run))
    # other tasks in ses-05
    for task in ['mdloc', 'tomloc']:
        for run in [1, 2]:
            jobs.append(('sub-004', 'ses-05', task, run))

    return jobs

def run_analysis(subject, session, task, run):
    """Run the localizer contrast analysis"""
    output_path = OUTPUT_DIR / f'{subject}_{session}_{task}_run{run}.png'
    
    cmd = [
        'python', str(SCRIPT_PATH),
        '--subject', subject,
        '--session', session,
        '--task', task,
        '--run', str(run),
        '--fsaverage', FSAVERAGE,
        '--output', str(output_path)
    ]
    
    logging.info(f"Running: {subject} {session} {task} run{run}")
    
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        logging.info(f"✓ Success")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"✗ Failed: {e.stderr}")
        return False

def main():
    """Main batch processing function"""
    jobs = generate_jobs()
    total = len(jobs)
    
    logging.info(f"Starting batch processing of {total} analyses")
    
    successful = sum(run_analysis(*job) for job in jobs)
    
    logging.info(f"\nBATCH COMPLETE: {successful}/{total} successful")

if __name__ == "__main__":
    main()