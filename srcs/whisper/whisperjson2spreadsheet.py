#!/usr/bin/env python3
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any

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


# =============================================================================
# Task Configuration
# =============================================================================

TASK_CONFIGS = {
    "recall": {
        "columns": ["SEG-B Number", "SEG-C Number", "Recall_text", "Start Time", "End Time"],
        "row_builder": lambda seg, idx: {
            "SEG-B Number": None,
            "SEG-C Number": None,
            "Recall_text": seg.get("text", "").strip(),
            "Start Time": round(seg.get("start", 0), 3),
            "End Time": round(seg.get("end", 0), 3)
        },
        "output_suffix": "desc-segments.xlsx"
    },
    "ahc": {
        "columns": ["Prompt Number", "Prompt", "Segment Number", "Text", "Start Time", "End Time", 
                    "Possibility Number"],
        "row_builder": lambda seg, idx: {
            "Prompt Number": None,
            "Prompt": None,
            "Segment Number": idx + 1,
            "Text": seg.get("text", "").strip(),
            "Start Time": round(seg.get("start", 0), 3),
            "End Time": round(seg.get("end", 0), 3),
            "Possibility Number": None
        },
        "output_suffix": "desc-segments.xlsx"
    }
}

# Mapping from task filter values to task types
TASK_TYPE_MAPPING = {
    # Recall-type tasks
    "freerecall": "recall",
    "cuedrecall": "recall",
    # AHC-type tasks
    "ahc": "ahc",
}

DEFAULT_TASK_TYPE = "recall"


def get_task_type(task_filter: str = None) -> str:
    """
    Determine the task type based on the task filter.
    Returns the task type key for TASK_CONFIGS.
    """
    if task_filter is None:
        return DEFAULT_TASK_TYPE
    
    clean_task = task_filter.replace("task-", "").lower()
    return TASK_TYPE_MAPPING.get(clean_task, DEFAULT_TASK_TYPE)


def find_json_files(subject: str = None, session: str = None, task: str = None) -> List[Path]:
    """
    Find _full.json files in DATA_DIR/rec/bids matching the specified criteria.
    """
    root_path = DATA_DIR / "rec" / "bids"
    
    if not root_path.exists():
        logger.error(f"BIDS directory not found: {root_path}")
        return []

    all_files = list(root_path.rglob("*_full.json"))
    matched_files = []

    logger.info(f"Scanning {len(all_files)} JSON files in {root_path}...")

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


def build_dataframe(segments: List[Dict[str, Any]], config: Dict[str, Any]) -> pd.DataFrame:
    """
    Build a DataFrame from segments based on the task configuration.
    """
    columns = config["columns"]
    row_builder = config["row_builder"]
    
    excel_rows = [row_builder(seg, idx) for idx, seg in enumerate(segments)]
    
    df = pd.DataFrame(excel_rows)
    
    # Handle empty DataFrame case
    if df.empty:
        df = pd.DataFrame(columns=columns)
    else:
        # Ensure columns are in the correct order
        df = df[columns]
    
    return df


def process_file(json_path: Path, config: Dict[str, Any]) -> bool:
    """
    Reads a WhisperX JSON file and saves a corresponding Excel file.
    Output format depends on the task configuration.
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        segments = data.get("segments", [])
        
        df = build_dataframe(segments, config)

        filename_str = json_path.name
        if "desc-audio_full.json" in filename_str:
            new_filename = filename_str.replace("desc-audio_full.json", config["output_suffix"])
            output_path = json_path.parent / new_filename
            
            df.to_excel(output_path, index=False)
            
            logger.info(f"Generated: {output_path.name}")
            return True
        else:
            logger.warning(f"Unexpected filename format: {filename_str}. Skipping.")
            return False

    except Exception as e:
        logger.error(f"Failed to process {json_path.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert WhisperX JSON outputs to Excel segments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all files for subject 001, session 01 (uses default recall format)
  python generate_excel_reports.py --subject 001 --session 01

  # Process all files for a specific task (format auto-detected)
  python generate_excel_reports.py --task story

  # Process AHC task files
  python generate_excel_reports.py --task ahc

  # Override the output format explicitly
  python generate_excel_reports.py --subject 001 --output-format ahc

Available output formats: {}
        """.format(", ".join(TASK_CONFIGS.keys()))
    )

    parser.add_argument("--subject", type=str, help="Subject ID (e.g., '001')")
    parser.add_argument("--session", type=str, help="Session ID (e.g., '01')")
    parser.add_argument("--task", type=str, help="Task name (e.g., 'svf', 'ahc')")
    parser.add_argument(
        "--output-format", 
        type=str, 
        choices=list(TASK_CONFIGS.keys()),
        help="Override the output format (default: auto-detect from task)"
    )
    
    args = parser.parse_args()
    
    # Determine task type and get config
    if args.output_format:
        task_type = args.output_format
    else:
        task_type = get_task_type(args.task)
    
    config = TASK_CONFIGS[task_type]
    
    logger.info(f"Using output format: {task_type}")
    
    # Find files
    files = find_json_files(args.subject, args.session, args.task)
    
    if not files:
        logger.warning(f"No matching _full.json files found in {DATA_DIR / 'rec' / 'bids'}")
        sys.exit(0)
        
    logger.info(f"Found {len(files)} files to process.")
    
    # Process files
    success_count = sum(1 for f in files if process_file(f, config))
            
    logger.info(f"Complete. Successfully generated {success_count}/{len(files)} Excel files.")


if __name__ == "__main__":
    main()