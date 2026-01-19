#!/bin/bash

# --- Configuration ---
LOG_FILE="/mnt/alphafold3_nectar/test/af_output/af3_gpu_stats.csv"
PLOT_FILE="/mnt/alphafold3_nectar/test/af_output/af3_gpu_stats.png"
PYTHON_SCRIPT="/mnt/alphafold3_nectar/test/scripts/plot_gpu_stats.py"

# --- Safety Checks ---
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Could not find $PYTHON_SCRIPT in the current directory."
    exit 1
fi

# --- 1. Start GPU Monitoring in Background ---
echo "--- Starting GPU Monitoring ---"

# -l 1 means sample every 1 second
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used \
           --format=csv \
           -l 1 > "$LOG_FILE" &

# Capture PID to kill it later
MONITOR_PID=$!

# Trap ensures we kill the monitor even if we Ctrl+C the script early
trap "kill $MONITOR_PID 2>/dev/null" EXIT

# --- 2. Run AlphaFold 3 Docker Command ---
echo "--- Starting AlphaFold 3 Container ---"

docker run -it \
    --volume /mnt/alphafold3_nectar/test/af_input:/root/af_input \
    --volume /mnt/alphafold3_nectar/test/af_output:/root/af_output \
    --volume /mnt/models:/root/models \
    --volume /mnt/af3_data/af3_data:/root/public_databases \
    --gpus all \
    alphafold3 \
    python run_alphafold.py \
    --json_path=/root/af_input/test.json \
    --model_dir=/root/models \
    --output_dir=/root/af_output

# --- 3. Clean up and Plot ---
echo "--- AlphaFold 3 Finished. Stopping Monitor... ---"

# Kill the nvidia-smi process
kill $MONITOR_PID

# Wait a moment to ensure file writes are flushed
sleep 1

echo "--- Generating Plot... ---"
python3 "$PYTHON_SCRIPT" "$LOG_FILE" "$PLOT_FILE"

echo "Done. Check $PLOT_FILE for results."