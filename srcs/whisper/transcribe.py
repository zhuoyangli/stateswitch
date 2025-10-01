import whisper
import sys
import os

# Load model
model = whisper.load_model("large-v3")

# Get input file
audio_file = sys.argv[1]
output_dir = "/scratch4/choney1/zli230/stateswitch/data/transcripts"

# Transcribe
result = model.transcribe(
    audio_file,
    condition_on_previous_text=False,  # Reduces repetition
    no_speech_threshold=0.6,  # Higher = more aggressive filtering
    logprob_threshold=-1.0,  # Filter uncertain segments
)

# Save transcript
base_name = os.path.basename(audio_file).replace('.wav', '.txt')
output_file = os.path.join(output_dir, base_name)

with open(output_file, 'w') as f:
    f.write(result["text"])

print(f"Saved to {output_file}")
