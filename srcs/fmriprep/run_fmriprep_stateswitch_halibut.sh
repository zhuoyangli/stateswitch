#!/bin/bash
# run_fmriprep_stateswitch_halibut.sh
# Usage:
# 1. Open a tmux session:
#   tmux new -s fmriprep_xxx
# 2. Run the script with participant number as argument:
#   ./run_fmriprep_stateswitch_halibut.sh <participant_number>
# 3. Detach from tmux session:
#   Press Ctrl+b then d
# 4. Reattach to tmux session later:
#   tmux attach -t fmriprep_xxx

# Set up paths
BIDS_DIR="/home/datasets/stateswitch/bids"
OUTPUT_DIR="/home/datasets/stateswitch/derivatives"
FMRIPREP_IMG="/home/envs/fmriprep/fmriprep-25.2.3.sif"
FS_LICENSE="/home/envs/fmriprep/license.txt"
LOG_DIR="/home/zli230/projects/stateswitch/logs/fmriprep"

mkdir -p ${LOG_DIR}

# Participant to process
PARTICIPANT_NUM=$1
PARTICIPANT=$(printf "%03d" ${PARTICIPANT_NUM})

WORK_DIR="/home/datasets/stateswitch/work/sub-${PARTICIPANT}"

# Generate timestamp for log files
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/fmriprep_sub-${PARTICIPANT}_${TIMESTAMP}.log"

# Check system load before starting
echo "Checking system resources..."
LOAD=$(uptime | awk -F'load average:' '{print $2}' | awk -F, '{print $1}' | xargs)
echo "Current system load: ${LOAD}"
echo "Available memory: $(free -h | grep Mem | awk '{print $7}')"

# Determine thread allocation based on system load
LOAD_INT=${LOAD%.*}
if [ "$LOAD_INT" -lt 30 ]; then
    NTHREADS=16
    OMP_NTHREADS=4
    echo "Low system load - using aggressive allocation"
elif [ "$LOAD_INT" -lt 60 ]; then
    NTHREADS=12
    OMP_NTHREADS=4
    echo "Moderate system load - using conservative allocation"
else
    echo "High system load (${LOAD}). Consider running later."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
    NTHREADS=8
    OMP_NTHREADS=2
fi

# Function to convert seconds to HH:MM:SS
seconds_to_hms() {
    local seconds=$1
    local hours=$((seconds / 3600))
    local minutes=$(( (seconds % 3600) / 60 ))
    local secs=$((seconds % 60))
    printf "%02d:%02d:%02d" $hours $minutes $secs
}

# Start logging everything
{
    echo "========================================="
    echo "=== fMRIPrep Job Started ==="
    echo "Date: $(date)"
    echo "Hostname: $(hostname)"
    echo "Subject: sub-${PARTICIPANT}"
    echo "BIDS directory: ${BIDS_DIR}"
    echo "Output directory: ${OUTPUT_DIR}"
    echo "Work directory: ${WORK_DIR}"
    echo "System load: ${LOAD}"
    echo "Available memory: $(free -h | grep Mem | awk '{print $7}')"
    echo "Thread allocation: ${NTHREADS} threads, ${OMP_NTHREADS} OMP threads"
    echo "Total CPU usage: ${NTHREADS} cores"
    echo "========================================="
    echo ""
    
    # Record start time
    START_TIME=$(date +%s)
    
    # Create work and derivatives directory if they don't exist
    mkdir -p ${OUTPUT_DIR}
    mkdir -p ${WORK_DIR}
    
    # Run fMRIPrep with nice to lower priority
    nice -n 10 singularity run --cleanenv \
      -B /home/datasets:/home/datasets \
      -B /home/envs:/home/envs:ro \
      ${FMRIPREP_IMG} \
      ${BIDS_DIR} \
      ${OUTPUT_DIR} \
      participant \
      --ignore slicetiming \
      --participant-label ${PARTICIPANT} \
      --work-dir ${WORK_DIR} \
      --fs-license-file ${FS_LICENSE} \
      --output-spaces anat MNI152NLin2009cAsym MNI152NLin6Asym:res-2 fsnative fsaverage6 \
      --stop-on-first-crash \
      --nthreads ${NTHREADS} --omp-nthreads ${OMP_NTHREADS} \
      --notrack \
      --subject-anatomical-reference unbiased \
      --write-graph \
      --bold2anat-dof 6 \
      --clean-workdir
    
    # Capture exit status
    EXITCODE=$?
    
    # Record end time and calculate duration
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    DURATION_HMS=$(seconds_to_hms $DURATION)
    
    echo ""
    echo "========================================="
    if [ $EXITCODE -eq 0 ]; then
        echo "=== fMRIPrep Completed Successfully ==="
        # Clean up work directory for this subject
        echo "Cleaning up work directory..."
        rm -rf ${WORK_DIR}/*sub-${PARTICIPANT}*
    else
        echo "=== fMRIPrep Failed (exit code: $EXITCODE) ==="
        echo "Work directory preserved for debugging"
    fi
    echo "End time: $(date)"
    echo "Total runtime: ${DURATION_HMS}"
    echo "Log file: ${LOG_FILE}"
    echo "========================================="
    
} 2>&1 | tee ${LOG_FILE}

# Get exit code from the subshell
EXIT_STATUS=${PIPESTATUS[0]}

# Final message (outside of logging block, so it shows in terminal)
if [ $EXIT_STATUS -eq 0 ]; then
    echo "  fMRIPrep completed successfully for sub-${PARTICIPANT}"
    echo "  Check results in: ${OUTPUT_DIR}"
else
    echo "  fMRIPrep failed for sub-${PARTICIPANT}"
    echo "  Check log file: ${LOG_FILE}"
fi