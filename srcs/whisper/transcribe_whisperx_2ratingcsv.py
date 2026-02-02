#!/usr/bin/env python3
import subprocess
import os
import sys
from pathlib import Path

# Add parent directory to path to import from configs
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import PROJECT_ROOT, DATA_DIR

import logging
import argparse
import json
import csv
from typing import Dict, Any, Optional, List
from datetime import datetime

# Set up directories under /data
WHISPER_DATA_DIR = DATA_DIR / "whisper"
TRANSCRIPTION_OUTPUT_DIR = WHISPER_DATA_DIR / "transcriptions"
TEMP_DIR = WHISPER_DATA_DIR / "temp"
LOG_DIR = WHISPER_DATA_DIR / "logs"

# Ensure directories exist
for dir_path in [TRANSCRIPTION_OUTPUT_DIR, TEMP_DIR, LOG_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Set up logging with timestamp in filename
log_filename = LOG_DIR / f"transcription_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def transcribe_with_whisperx(
    audio_file: str,
    model_name: str = "large-v3",
    language: str = "en",
    output_dir: Optional[Path] = None
) -> Optional[Dict[str, Any]]:
    """Transcribe a single audio file using WhisperX."""
    
    if output_dir is None:
        output_dir = TEMP_DIR / "whisperx_output"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use whisperx directly instead of uvx
    cmd = [
        "whisperx",
        audio_file,
        "--model", model_name,
        "--language", language,
        "--output_format", "json",
        "--output_dir", str(output_dir),
        "--align_model", "WAV2VEC2_ASR_LARGE_LV60K_960H",
        # Force CPU to avoid cuDNN issues
        "--device", "cpu",
        "--compute_type", "float32"
    ]
    
    logger.info(f"Running WhisperX on {audio_file}")
    logger.info("Using CPU for transcription (cuDNN not available)")
    logger.debug(f"Command: {' '.join(cmd)}")
    
    # Set environment variable to handle torch.load weights_only issue
    env = os.environ.copy()
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "true"
    
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    
    if result.returncode != 0:
        logger.error(f"WhisperX failed: {result.stderr}")
        return None
    
    # Read the output JSON
    output_file = output_dir / f"{Path(audio_file).stem}.json"
    if output_file.exists():
        with open(output_file, 'r') as f:
            data = json.load(f)
        # Clean up temp file
        output_file.unlink()
        return data
    
    return None

def save_transcript(result: Dict[str, Any], output_path: Path, base_name: str) -> None:
    """Save transcription results in multiple formats."""
    
    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save plain text transcript
    transcript_path = output_path / f"{base_name}.txt"
    text = result.get("text", "")
    with open(transcript_path, 'w', encoding='utf-8') as f:
        f.write(text.strip())
    
    # Save full result as JSON
    json_path = output_path / f"{base_name}_full.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    # Save as CSV with word-level timestamps for rating
    csv_path = output_path / f"{base_name.replace('audio', 'wordtimestamps')}.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['transcription', 'word_clean', 'start', 'end', 'rater', 'changed'])
        
        # Extract word-level data if available
        if "word_segments" in result:
            for segment in result["word_segments"]:
                writer.writerow([
                    segment.get("word", ""),
                    segment.get("word", "").lower(),
                    segment.get("start", ""),
                    segment.get("end", ""),
                    "",  # rater empty
                    False  # changed False
                ])

def should_skip_file(wav_file: Path, output_path: Path) -> bool:
    """Check if file should be skipped based on existing outputs."""
    base_name = wav_file.stem
    transcript_path = output_path / f"{base_name}.txt"
    csv_path = output_path / f"{base_name}_rating.csv"
    return transcript_path.exists() and csv_path.exists()

def get_wav_files(session_path: Path, task: Optional[str] = None, run: Optional[str] = None) -> List[Path]:
    """Get WAV files, optionally filtered by task."""
    
    # Look in audio subdirectory
    audio_path = session_path / "audio"
    if not audio_path.exists():
        # Fallback to session directory itself
        audio_path = session_path
    
    if task:
        # Look for files with the specific task in the filename
        if run:
            pattern = f"*task-{task}*run-{run}*audio.wav"
        else:
            pattern = f"*task-{task}*audio.wav"

        wav_files = sorted(audio_path.glob(pattern))
        
        if not wav_files:
            # Try without 'task-' prefix
            pattern = f"*{task}*audio.wav"
            wav_files = sorted(audio_path.glob(pattern))
    else:
        # Get all WAV files
        wav_files = sorted(audio_path.glob("*audio.wav"))
    
    return wav_files

def transcribe_session(
    subject: str,
    session: str,
    task: Optional[str] = None,
    run: Optional[str] = None,
    model_name: str = "large-v3",
    skip_existing: bool = True,
    save_to_source: bool = True
) -> None:
    """Transcribe audio files in a session, optionally filtered by task."""
    
    # Construct paths - check where audio files are located
    session_path = DATA_DIR / "rec" / "bids" / f"sub-{subject}" / f"ses-{session}"
    
    if not session_path.exists():
        raise FileNotFoundError(f"Session path does not exist: {session_path}")
    
    # Find WAV files
    wav_files = get_wav_files(session_path, task, run)
    
    if not wav_files:
        if task:
            logger.warning(f"No WAV files found for task '{task}' in {session_path}")
        else:
            logger.warning(f"No WAV files found in {session_path}")
        return
    
    # Determine output path based on where we found the files
    audio_dir = wav_files[0].parent
    if save_to_source:
        output_path = audio_dir
    else:
        # Save to data/whisper/transcriptions directory maintaining BIDS structure
        relative_path = audio_dir.relative_to(DATA_DIR / "rec" / "bids")
        output_path = TRANSCRIPTION_OUTPUT_DIR / relative_path
    
    task_info = f" for task '{task}'" if task else ""
    run_info = f" and run '{run}'" if run else ""
    logger.info(f"Starting transcription{task_info}{run_info} for sub-{subject}/ses-{session}")
    logger.info(f"Audio files located in: {audio_dir}")
    logger.info(f"Output directory: {output_path}")
    logger.info(f"Using BIDS directory: {DATA_DIR / 'rec' / 'bids'}")
    logger.info(f"Whisper data directory: {WHISPER_DATA_DIR}")
    
    # Process each file
    successful = 0
    failed = 0
    skipped = 0
    
    for wav_file in wav_files:
        try:
            # Skip if outputs exist
            if skip_existing and should_skip_file(wav_file, output_path):
                logger.info(f"Skipping {wav_file.name} (outputs exist)")
                skipped += 1
                continue
            
            logger.info(f"Processing {wav_file.name}")
            
            # Transcribe
            result = transcribe_with_whisperx(
                str(wav_file), 
                model_name
            )
            
            if result is None:
                logger.error(f"Failed to transcribe {wav_file.name}")
                failed += 1
                continue
            
            # Save results
            save_transcript(result, output_path, wav_file.stem)
            
            logger.info(f"Successfully transcribed {wav_file.name}")
            successful += 1
            
        except Exception as e:
            logger.error(f"Failed to process {wav_file.name}: {str(e)}")
            failed += 1
            continue
    
    # Clean up temporary directory
    temp_whisperx = TEMP_DIR / "whisperx_output"
    if temp_whisperx.exists():
        import shutil
        shutil.rmtree(temp_whisperx)
    
    # Summary
    logger.info(f"Completed transcription{task_info} for sub-{subject}/ses-{session}")
    logger.info(f"Summary: {successful} successful, {failed} failed, {skipped} skipped")

def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio files using WhisperX and create rating CSV files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--subject", type=str, required=True, help="Subject identifier (e.g., '01')")
    parser.add_argument("--session", type=str, required=True, help="Session identifier (e.g., '01')")
    parser.add_argument(
        "--task", 
        type=str, 
        help="Specific task to transcribe (e.g., 'svf', 'sb', 'rest')"
    )
    parser.add_argument("--run", type=str, help="Specific run identifier if applicable")
    parser.add_argument(
        "--model", 
        type=str, 
        default="large-v3",
        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
        help="WhisperX model to use"
    )
    parser.add_argument(
        "--no-skip-existing", 
        action="store_true",
        help="Process files even if outputs already exist"
    )
    parser.add_argument(
        "--save-to-whisper-dir",
        action="store_true",
        help="Save outputs to data/whisper/transcriptions instead of source directory"
    )
    
    args = parser.parse_args()
    
    logger.info(f"Starting transcription with WhisperX model: {args.model}")
    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info(f"BIDS directory: {DATA_DIR / 'rec' / 'bids'}")
    logger.info(f"Whisper data stored in: {WHISPER_DATA_DIR}")
    
    try:
        transcribe_session(
            subject=args.subject,
            session=args.session,
            task=args.task,
            run=args.run,
            model_name=args.model,
            skip_existing=not args.no_skip_existing,
            save_to_source=not args.save_to_whisper_dir
        )
    except KeyboardInterrupt:
        logger.info("Transcription interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()