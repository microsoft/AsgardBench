#!/usr/bin/env bash
set -ex

# Multi-node vLLM serving script for AML/Singularity clusters
# Supports both single-node and multi-node (Ray-based) configurations
# Automatically configures vLLM flags based on model type
#
# Run with:
#   uv run amlt run Magmathor/Model/aml_vllm_starter.yaml --yes --description "VLLM Starter"

# Parse command line arguments
PORT=${1:-43289}

# MODEL_CHECKPOINT and MODEL_NAME must be set by the YAML via environment variables
if [ -z "$MODEL_CHECKPOINT" ]; then
    echo "ERROR: MODEL_CHECKPOINT environment variable is not set"
    exit 1
fi
if [ -z "$MODEL_NAME" ]; then
    echo "ERROR: MODEL_NAME environment variable is not set"
    exit 1
fi

# Get cluster info from environment (set by AML/Singularity)
# With process_count_per_node=1, WORLD_SIZE = number of nodes
WORLD_SIZE=${WORLD_SIZE:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-6379}

CUR_DIR=$PWD

echo "============================================"
echo "vLLM Configuration:"
echo "  Model Name: $MODEL_NAME"
echo "  Model Checkpoint: $MODEL_CHECKPOINT"
echo "  Port: $PORT"
echo "  Node Rank: $NODE_RANK"
echo "  World Size (nodes): $WORLD_SIZE"
echo "  Master Address: $MASTER_ADDR:$MASTER_PORT"
echo "============================================"

# ============================================
# Model-specific configuration
# ============================================
# Default vLLM version and flags
VLLM_EXTRA_FLAGS=""
USE_NIGHTLY=false
PIP_EXTRAS=""

# Configure based on model name/checkpoint
case "$MODEL_NAME" in
    *DeepSeek-V3.2*|*deepseek-v3.2*|*DeepSeek-V3_2*)
        echo "Detected DeepSeek-V3.2 model - applying specific configuration"
        USE_NIGHTLY=true
        VLLM_EXTRA_FLAGS="--tokenizer-mode deepseek_v32 --tool-call-parser deepseek_v32 --enable-auto-tool-choice --reasoning-parser deepseek_v3 --enable-expert-parallel"
        ;;
    *DeepSeek-V3*|*deepseek-v3*|*DeepSeek-R1*|*deepseek-r1*)
        echo "Detected DeepSeek-V3/R1 model - applying specific configuration"
        USE_NIGHTLY=true
        VLLM_EXTRA_FLAGS="--enable-expert-parallel"
        ;;
    *Qwen3-VL-235B*|*Qwen3-VL*A22B*|*Qwen3-235B*|*Qwen3*A22B*)
        echo "Detected Qwen3 MoE model (235B-A22B) - applying specific configuration"
        # MoE model with Ray multi-node: use expert parallel + memory limits
        # Based on: https://docs.vllm.ai/projects/ascend/en/latest/tutorials/multi_node_ray.html
        VLLM_EXTRA_FLAGS="--enable-expert-parallel --mm-encoder-tp-mode data --gpu-memory-utilization 0.9 --limit-mm-per-prompt.video 0"
        PIP_EXTRAS="qwen-vl-utils==0.0.14"
        ;;
    *Qwen3-VL-32B*|*Qwen3-32B*|*Qwen3-VL32B*|*Qwen3-32B*)
        echo "Detected Qwen3-VL-32B model - applying specific configuration"
        VLLM_EXTRA_FLAGS="--gpu-memory-utilization 0.9 --mm-encoder-tp-mode data --limit-mm-per-prompt.video 0"
        PIP_EXTRAS="qwen-vl-utils==0.0.14"
        ;;
    *Qwen3-VL-30B-A3B*)
        echo "Detected Qwen3-VL-30B-A3B model - applying specific configuration"
        VLLM_EXTRA_FLAGS="--enable-expert-parallel --gpu-memory-utilization 0.9 --mm-encoder-tp-mode data --limit-mm-per-prompt.video 0"
        PIP_EXTRAS="qwen-vl-utils==0.0.14"
        ;;
    *Phi-4*|*phi-4*)
        echo "Detected Phi-4 model - applying specific configuration"
        VLLM_EXTRA_FLAGS="--mm-encoder-tp-mode data"
        ;;
    *)
        echo "Using default vLLM configuration for model: $MODEL_NAME"
        ;;
esac

echo "vLLM extra flags: $VLLM_EXTRA_FLAGS"
echo "Use nightly build: $USE_NIGHTLY"

# Install dependencies
cd /tmp/
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

mkdir -p /tmp/vllm-server
cd /tmp/vllm-server
uv venv --python=3.12 --clear
source .venv/bin/activate

# Install vLLM (nightly for newer models like DeepSeek-V3.2)
if [ "$USE_NIGHTLY" = true ]; then
    echo "Installing vLLM nightly build..."
    uv pip install vllm --extra-index-url https://wheels.vllm.ai/nightly
else
    echo "Installing vLLM stable build..."
    uv pip install "vllm[video]==0.12.0" --torch-backend=auto
fi
uv pip install flashinfer-python==0.5.3 bitsandbytes==0.47.0 $PIP_EXTRAS

# Install ray for multi-node support
if [ "$WORLD_SIZE" -gt 1 ]; then
    uv pip install ray
fi



# Detect number of GPUs and set CUDA_VISIBLE_DEVICES dynamically
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [ "$NUM_GPUS" -eq 0 ]; then
    echo "Error: No GPUs detected"
    exit 1
fi
echo "Detected $NUM_GPUS GPU(s) on this node"

# Generate CUDA_VISIBLE_DEVICES string (0,1,2,...,N-1)
CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS - 1)))
export CUDA_VISIBLE_DEVICES
echo "Setting CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Start GPU keep-alive monitor in background (for AML)
if [ -f "$CUR_DIR/gpu_keepalive.py" ]; then
    echo "Starting GPU keep-alive monitor..."
    python3 "$CUR_DIR/gpu_keepalive.py" > /tmp/gpu_keepalive.log 2>&1 &
    GPU_MONITOR_PID=$!
    echo "GPU monitor started with PID: $GPU_MONITOR_PID"
fi

# Cleanup function
cleanup() {
    echo "Cleaning up..."
    if [ -n "$GPU_MONITOR_PID" ]; then
        kill $GPU_MONITOR_PID 2>/dev/null || true
        wait $GPU_MONITOR_PID 2>/dev/null || true
    fi
    if [ "$WORLD_SIZE" -gt 1 ]; then
        ray stop --force 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# Ray cluster setup timeout
RAY_INIT_TIMEOUT=${RAY_INIT_TIMEOUT:-300}

# Function to start Ray head node
start_ray_head() {
    echo "Starting Ray head node on port $MASTER_PORT..."
    ray start --head --port=$MASTER_PORT --num-gpus=$NUM_GPUS

    # Wait for all worker nodes to join
    echo "Waiting for $WORLD_SIZE nodes to join the cluster..."
    for (( i=0; i < $RAY_INIT_TIMEOUT; i+=5 )); do
        active_nodes=$(python3 -c 'import ray; ray.init(); print(sum(node["Alive"] for node in ray.nodes()))')
        if [ "$active_nodes" -eq "$WORLD_SIZE" ]; then
            echo "All $WORLD_SIZE nodes are active. Ray cluster initialized successfully."
            return 0
        fi
        echo "  $active_nodes/$WORLD_SIZE nodes active, waiting..."
        sleep 5
    done

    echo "ERROR: Timed out waiting for Ray workers"
    return 1
}

# Function to start Ray worker node
start_ray_worker() {
    echo "Starting Ray worker node, connecting to $MASTER_ADDR:$MASTER_PORT..."

    for (( i=0; i < $RAY_INIT_TIMEOUT; i+=5 )); do
        ray start --address=$MASTER_ADDR:$MASTER_PORT --num-gpus=$NUM_GPUS --block &
        RAY_PID=$!

        # Check if Ray started successfully
        sleep 5
        if kill -0 $RAY_PID 2>/dev/null; then
            echo "Ray worker connected successfully"
            # Keep the worker running
            wait $RAY_PID
            return 0
        fi

        echo "  Retrying connection to head node..."
        sleep 5
    done

    echo "ERROR: Failed to connect to Ray head node"
    return 1
}

# Determine parallelism configuration
# Tensor parallelism: within a node (use all GPUs on the node)
# Pipeline parallelism: across nodes
TP_SIZE=$NUM_GPUS
PP_SIZE=$WORLD_SIZE

echo "Parallelism configuration:"
echo "  Tensor Parallel Size (within node): $TP_SIZE"
echo "  Pipeline Parallel Size (across nodes): $PP_SIZE"

# Single-node case: just run vLLM directly
if [ "$WORLD_SIZE" -eq 1 ]; then
    echo "=== SINGLE NODE MODE ==="
    vllm serve "$MODEL_CHECKPOINT" \
        --tensor-parallel-size "$TP_SIZE" \
        --uvicorn-log-level critical \
        --allowed-local-media-path / \
        --trust-remote-code \
        --host 0.0.0.0 \
        --port "$PORT" \
        --served-model-name "vllm-model" \
        $VLLM_EXTRA_FLAGS
else
    # Multi-node case: use Ray for coordination
    if [ "$NODE_RANK" -eq 0 ]; then
        echo "=== HEAD NODE (rank 0) ==="
        start_ray_head

        if [ $? -ne 0 ]; then
            echo "Failed to initialize Ray cluster"
            exit 1
        fi

        # Only the head node runs vLLM serve
        echo "Starting vLLM server..."
        vllm serve "$MODEL_CHECKPOINT" \
            --tensor-parallel-size "$TP_SIZE" \
            --pipeline-parallel-size "$PP_SIZE" \
            --distributed-executor-backend ray \
            --uvicorn-log-level critical \
            --allowed-local-media-path / \
            --trust-remote-code \
            --host 0.0.0.0 \
            --port "$PORT" \
            --served-model-name "vllm-model" \
            $VLLM_EXTRA_FLAGS
    else
        echo "=== WORKER NODE (rank $NODE_RANK) ==="
        start_ray_worker
    fi
fi
