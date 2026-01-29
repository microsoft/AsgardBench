#!/bin/bash
# Run 6 parallel instances of model_tester on magt_benchmark_p1 through magt_benchmark_p6

# Get the directory where this script is located and go to repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Common arguments from launch.json "Model Tester (WSL)"
COMMON_ARGS="--include_common_sense --previous_image grayscale --model gpt-4o --model-name gpt-4o --rep 1 --temperature 0.6 --max_completion_tokens 4096 --feedback_type simple --hand_transparency 60 --full_steps"

echo "Starting 6 parallel model tester instances..."

# Launch all 6 in background
for i in {1..6}; do
    echo "Starting magt_benchmark_p$i..."
    PYTHONPATH="$REPO_ROOT" PYTHONUNBUFFERED=1 python3 "$REPO_ROOT/AsgardBench/Model/model_tester.py" \
        --test "magt_benchmark_p$i" \
        $COMMON_ARGS \
        > "logs/model_tester_p$i.log" 2>&1 &
    echo "  PID: $!"
done

echo ""
echo "All 6 instances started. Logs are in logs/model_tester_p*.log"
echo "Use 'tail -f logs/model_tester_p*.log' to monitor progress"
echo "Use 'ps aux | grep model_tester' to check running processes"
echo "Use 'pkill -f model_tester.py' to stop all instances"

# Wait for all background jobs
wait
echo "All instances completed."
