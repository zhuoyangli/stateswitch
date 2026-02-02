#!/bin/bash

# Define subject-session pairs based on your transferred files
declare -a PAIRS=(
    "001:05"
    "001:06"
    "003:06"
    "003:07"
    "003:08"
    "003:09"
    "004:07"
    "004:08"
    "004:09"
    "006:05"
    "006:06"
    "006:07"
    "007:05"
    "007:06"
    "007:07"
    "008:05"
    "008:06"
    "008:07"
    "009:05"
    "009:06"
)

for pair in "${PAIRS[@]}"; do
    IFS=':' read -r subject session <<< "$pair"
    echo "Processing subject $subject, session $session"
    uv run python ./srcs/whisper/transcribe_whisperx_2ratingcsv.py --subject "$subject" --session "$session" --task ahc
done