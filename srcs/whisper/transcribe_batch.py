# transcribe_batch.py
import whisper
import os
from pathlib import Path
import sys

# Load model once
print("Loading Whisper model...")
model = whisper.load_model("large-v3")

# Set paths
input_dir = "/scratch4/choney1/zli230/stateswitch/data/rec/chunked"
output_dir = "/scratch4/choney1/zli230/stateswitch/data/mfa"

# Process all WAV files
wav_files = list(Path(input_dir).rglob("*.wav"))
print(f"Found {len(wav_files)} WAV files to process")

for i, wav_path in enumerate(wav_files, 1):
    # Get subject folder name (e.g., sub-001)
    subject = wav_path.parent.name
    
    # Create MFA structure: mfa/sub-001/
    subject_output_dir = Path(output_dir) / subject
    subject_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Output transcript with same name but .txt extension
    txt_output = subject_output_dir / (wav_path.stem + ".txt")
    
    # Skip if already processed
    if txt_output.exists():
        print(f"[{i}/{len(wav_files)}] Skipping {subject}/{wav_path.name} - transcript exists")
        continue
    
    print(f"[{i}/{len(wav_files)}] Transcribing {subject}/{wav_path.name}...")
    
    # Transcribe with VAD settings
    result = model.transcribe(
        str(wav_path),
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
        logprob_threshold=-1.0,
        language='en',
    )
    
    # Save transcript
    with open(txt_output, 'w') as f:
        f.write(result["text"])
    
    print(f"  → Saved transcript to {txt_output}")

print("Done! MFA structure ready at:", output_dir)