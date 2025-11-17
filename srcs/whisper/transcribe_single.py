#!/usr/bin/env python3
"""
transcribe.py - Transcribe a single WAV file using OpenAI Whisper
"""

import whisper
import json
import argparse
from pathlib import Path
import torch
import warnings

warnings.filterwarnings("ignore")

def transcribe_wav(wav_path: str, model_name: str = "base"):
    """Transcribe a WAV file using Whisper."""
    # Load model
    print(f"Loading Whisper model '{model_name}'...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(model_name, device=device)
    print(f"Model loaded on {device}")
    
    # Transcribe
    print(f"Transcribing {wav_path}...")
    result = model.transcribe(
        wav_path,
        language='en',
        word_timestamps=True,
        # condition_on_previous_text=False,
        fp16=torch.cuda.is_available()
    )
    print("Transcription completed!")
    return result

def save_results(result: dict, wav_path: str, output_dir: str = None):
    """Save transcription results as both TXT and JSON."""
    wav_path = Path(wav_path)
    base_name = wav_path.stem
    out_dir = Path(output_dir) if output_dir else wav_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save plain text
    txt_path = out_dir / f"{base_name}.txt"
    txt_path.write_text(result["text"].strip(), encoding='utf-8')
    print(f"Saved transcript to {txt_path}")
    
    # Save full JSON
    json_path = out_dir / f"{base_name}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Saved full result to {json_path}")

def main():
    parser = argparse.ArgumentParser(description="Transcribe a WAV file using OpenAI Whisper")
    parser.add_argument("wav_file", help="Path to the WAV file")
    parser.add_argument("--model", default="large-v3", 
                       choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "large-v3-turbo"],
                       help="Whisper model to use (default: large-v3)")
    parser.add_argument("--output-dir", help="Output directory (default: same as input)")
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.wav_file).exists():
        print(f"Error: File not found: {args.wav_file}")
        return
    
    # Transcribe
    result = transcribe_wav(args.wav_file, args.model)
    
    # Save results
    save_results(result, args.wav_file, args.output_dir)

if __name__ == "__main__":
    main()