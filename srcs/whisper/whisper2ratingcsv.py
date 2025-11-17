import sys
from pathlib import Path

# Add the parent directory to sys.path to import from configs
sys.path.insert(0, str(Path(__file__).parent.parent))

# Now you can import from configs
from configs.config import PROJECT_ROOT

import json
import pandas as pd

def convert_whisper_to_rating_csv(json_path, output_path=None):
    """
    Convert Whisper JSON output to rating CSV format.
    
    Args:
        json_path: Path to the Whisper JSON file
        output_path: Optional output path for CSV. If None, saves in same directory
    """
    json_path = Path(json_path)
    
    # Load Whisper JSON
    with open(json_path, 'r') as f:
        whisper_data = json.load(f)
    
    # Extract words from segments
    all_words = []
    
    # Whisper JSON structure typically has 'segments' with 'words'
    if 'segments' in whisper_data:
        for segment in whisper_data['segments']:
            if 'words' in segment:
                all_words.extend(segment['words'])
    elif 'words' in whisper_data:
        # Sometimes words are at the top level
        all_words = whisper_data['words']
    else:
        raise ValueError("Could not find words in Whisper JSON structure")
    
    # Create DataFrame with required columns
    rating_data = []
    for i, word_info in enumerate(all_words):
        rating_data.append({
            'transcription': word_info.get('word', '').strip(),
            'word_clean': word_info.get('word', '').strip().lower(),
            'start': word_info.get('start', 0.0),
            'end': word_info.get('end', 0.0),
            'rater': '',  # Empty initially
            'changed': False
        })
    
    # Create DataFrame
    df = pd.DataFrame(rating_data)
    
    # Determine output path
    if output_path is None:
        if 'desc-audio_full.json' in json_path.name:
            output_path = json_path.parent / json_path.name.replace('desc-audio_full.json', 'desc-wordtimestamps.csv')
        else:
            output_path = json_path.with_suffix('.csv')
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Converted {len(df)} words from {json_path} to {output_path}")
    
    return df

def batch_convert_whisper_files(bids_dir=None):
    """
    Convert all Whisper JSON files in a BIDS directory structure.
    
    Args:
        bids_dir: Path to the BIDS directory. If None, uses PROJECT_ROOT/data/rec/bids
    """
    if bids_dir is None:
        bids_dir = PROJECT_ROOT / "data" / "rec" / "bids"
    else:
        bids_dir = Path(bids_dir)
    
    print(f"Looking for Whisper JSON files in: {bids_dir}")
    
    json_files = list(bids_dir.glob("sub-*/ses-*/*_full.json"))
    
    print(f"Found {len(json_files)} Whisper JSON files to convert")
    
    for json_file in json_files:
        try:
            convert_whisper_to_rating_csv(json_file)
        except Exception as e:
            print(f"Error converting {json_file}: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Convert single file
        convert_whisper_to_rating_csv(sys.argv[1])
    else:
        # Batch convert using PROJECT_ROOT
        batch_convert_whisper_files()