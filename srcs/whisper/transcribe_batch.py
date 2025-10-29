# transcribe_batch.py
import whisper
import os
from pathlib import Path
import sys
import logging
import argparse
import json
import csv
from typing import Dict, Any, Optional
import torch
from tqdm import tqdm
import warnings

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('transcription.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Global model variable to avoid reloading
_model_cache = {}

def load_model_cached(model_name: str) -> whisper.Whisper:
    """Load model with caching to avoid reloading the same model."""
    if model_name not in _model_cache:
        logger.info(f"Loading Whisper model '{model_name}'...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model_cache[model_name] = whisper.load_model(model_name, device=device)
        logger.info(f"Model loaded on {device}")
    return _model_cache[model_name]

def transcribe_wav(
    wav_path: str, 
    whisper_model: str, 
    condition_on_previous_text: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Transcribe a WAV file using Whisper.
    
    Args:
        wav_path: Path to the WAV file
        whisper_model: Name of the Whisper model to use
        condition_on_previous_text: Whether to condition on previous text
        **kwargs: Additional arguments to pass to model.transcribe()
    
    Returns:
        Transcription result dictionary
    """
    model = load_model_cached(whisper_model)
    
    # Default transcription parameters
    transcribe_params = {
        'condition_on_previous_text': condition_on_previous_text,
        'word_timestamps': True,
        'language': 'en',
        'verbose': False,
        'fp16': torch.cuda.is_available(),  # Use FP16 on GPU
    }
    
    # Update with any additional parameters
    transcribe_params.update(kwargs)
    
    try:
        result = model.transcribe(wav_path, **transcribe_params)
        return result
    except Exception as e:
        logger.error(f"Error transcribing {wav_path}: {str(e)}")
        raise

def save_transcript(result: Dict[str, Any], session_path: Path, base_name: str) -> None:
    """Save transcription results in multiple formats in the same directory as the WAV file."""
    # Save plain text transcript
    transcript_path = session_path / f"{base_name}.txt"
    with open(transcript_path, 'w', encoding='utf-8') as f:
        f.write(result["text"].strip())
    
    # Save full result as JSON for complete information
    json_path = session_path / f"{base_name}_full.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

def should_skip_file(wav_file: Path, session_path: Path) -> bool:
    """Check if file should be skipped based on existing outputs."""
    base_name = wav_file.stem
    transcript_path = session_path / f"{base_name}.txt"
    return transcript_path.exists()

def transcribe_session(
    subject: str, 
    session: str, 
    data_dir: str,
    model_name: str = "large-v3",
    skip_existing: bool = True
) -> None:
    """
    Transcribe all WAV files in a session.
    
    Args:
        subject: Subject identifier
        session: Session identifier
        data_dir: Base data directory
        model_name: Whisper model to use
        skip_existing: Whether to skip files that have already been transcribed
    """
    session_path = Path(data_dir) / subject / session
    
    if not session_path.exists():
        logger.error(f"Session path does not exist: {session_path}")
        return
    
    # Find all WAV files
    wav_files = list(session_path.glob("*.wav"))
    
    if not wav_files:
        logger.warning(f"No WAV files found in {session_path}")
        return
    
    logger.info(f"Found {len(wav_files)} WAV files in {session_path}")
    
    # Process each file
    for wav_file in tqdm(wav_files, desc=f"Transcribing {subject}/{session}"):
        # Skip if already processed
        if skip_existing and should_skip_file(wav_file, session_path):
            logger.info(f"Skipping {wav_file.name} (already processed)")
            continue
        
        logger.info(f"Processing {wav_file.name}...")
        
        try:
            # Determine transcription parameters based on task
            if "task-svf" in wav_file.name or "task-sb" in wav_file.name:
                condition_on_previous = False
            else:
                condition_on_previous = True
            
            # Transcribe
            result = transcribe_wav(
                str(wav_file), 
                model_name, 
                condition_on_previous_text=condition_on_previous
            )
            
            # Save results in the same directory as the WAV file
            save_transcript(result, session_path, wav_file.stem)
            
            logger.info(f"Successfully transcribed {wav_file.name}")
            
        except Exception as e:
            logger.error(f"Failed to process {wav_file.name}: {str(e)}")
            continue
    
    logger.info(f"Completed transcription for {subject}/{session}")

def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio files in a session using OpenAI Whisper",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--subject", type=str, required=True, help="Subject identifier")
    parser.add_argument("--session", type=str, required=True, help="Session identifier")
    parser.add_argument(
        "--data-dir", 
        type=str, 
        default="/scratch4/choney1/zli230/stateswitch/data/rec/bids",
        help="Base data directory"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="large-v3",
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        help="Whisper model to use"
    )
    parser.add_argument(
        "--no-skip-existing", 
        action="store_true",
        help="Process files even if outputs already exist"
    )
    
    args = parser.parse_args()
    
    try:
        transcribe_session(
            subject=args.subject,
            session=args.session,
            data_dir=args.data_dir,
            model_name=args.model,
            skip_existing=not args.no_skip_existing
        )
    except KeyboardInterrupt:
        logger.info("Transcription interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()