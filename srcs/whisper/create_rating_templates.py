#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path
from typing import List

# Add parent directory to path to import from configs
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from configs.config import DATA_DIR
except ImportError:
    print("Error: Could not import DATA_DIR from configs.config.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

import pandas as pd


def find_csv_files(subject: str = None, session: str = None, task: str = None) -> List[Path]:
    """
    Find *-wordtimestampswithswitch.csv files in DATA_DIR matching the specified criteria.
    """
    root_path = DATA_DIR / "rec" / "svf_annotated" / "gio_rated"
    
    if not root_path.exists():
        logger.error(f"Data directory not found: {root_path}")
        return []

    all_files = list(root_path.rglob("*-wordtimestampswithswitch.csv"))
    matched_files = []

    logger.info(f"Scanning {len(all_files)} CSV files in {root_path}...")

    for f in all_files:
        filename = f.name
        
        if subject:
            clean_sub = subject.replace("sub-", "")
            if f"sub-{clean_sub}" not in filename:
                continue

        if session:
            clean_ses = session.replace("ses-", "")
            if f"ses-{clean_ses}" not in filename:
                continue

        if task:
            clean_task = task.replace("task-", "")
            if f"task-{clean_task}" not in filename:
                continue
        
        matched_files.append(f)

    return matched_files


def process_file(csv_path: Path) -> bool:
    """
    Reads a CSV file and saves a corresponding Excel file with transformed columns.
    
    Transformations:
    - Add "category" column at the beginning (empty)
    - Keep "transcription" column
    - Remove "word_clean" column
    - Round "start" and "end" to 3 decimals
    - Keep "rater" column but make it empty
    - Remove "changed" column
    - Keep "switch_flag" column but make it empty
    """
    try:
        # Read CSV file
        df = pd.read_csv(csv_path)
        
        # Verify expected columns exist
        expected_columns = ["transcription", "word_clean", "start", "end", "rater", "changed", "switch_flag"]
        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            logger.warning(f"Missing columns in {csv_path.name}: {missing_columns}. Skipping.")
            return False
        
        # Create new DataFrame with desired structure
        new_df = pd.DataFrame()
        
        # Add "category" column at the beginning (empty)
        new_df["category"] = ""
        
        # Keep "transcription" column
        new_df["word"] = df["transcription"]
        
        # Round "start" and "end" to 3 decimals
        new_df["start"] = df["start"].round(3)
        new_df["end"] = df["end"].round(3)
        
        # Keep "rater" column but make it empty
        new_df["rater"] = "GL"
        
        # Keep "switch_flag" column but make it empty
        new_df["switch_flag"] = df["switch_flag"] + 1
        
        # Generate output filename
        filename_str = csv_path.name
        new_filename = filename_str.replace("-wordtimestampswithswitch.csv", "-wordtimestamps_rated.xlsx")
        output_path = csv_path.parent / new_filename
        
        # Save to Excel
        new_df.to_excel(output_path, index=False)
        
        logger.info(f"Generated: {output_path.name}")
        return True

    except Exception as e:
        logger.error(f"Failed to process {csv_path.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert word timestamp CSV files to Excel format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all files
  python convert_wordtimestamps.py

  # Process all files for subject 001
  python convert_wordtimestamps.py --subject 001

  # Process all files for subject 001, session 01
  python convert_wordtimestamps.py --subject 001 --session 01

  # Process all files for a specific task
  python convert_wordtimestamps.py --task story
        """
    )

    parser.add_argument("--subject", type=str, help="Subject ID (e.g., '001')")
    parser.add_argument("--session", type=str, help="Session ID (e.g., '01')")
    parser.add_argument("--task", type=str, help="Task name (e.g., 'svf', 'ahc')")
    
    args = parser.parse_args()
    
    # Find files
    files = find_csv_files(args.subject, args.session, args.task)
    
    if not files:
        logger.warning(f"No matching *-wordtimestampswithswitch.csv files found.")
        sys.exit(0)
        
    logger.info(f"Found {len(files)} files to process.")
    
    # Process files
    success_count = sum(1 for f in files if process_file(f))
            
    logger.info(f"Complete. Successfully generated {success_count}/{len(files)} Excel files.")


if __name__ == "__main__":
    main()