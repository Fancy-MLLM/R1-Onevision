#!/bin/bash

RUN_NAME=multinode_training # assume there is a ${RUN_NAME}_args.yaml file in the current directory

declare -A node2ip_map
node2ip_map=(
    ["node1"]="192.168.1.101"
    ["node2"]="192.168.1.102"
    ["node3"]="192.168.1.103"
    ["node4"]="192.168.1.104"
)

# Default nodes if no arguments provided
DEFAULT_NODES=("node1" "node2")

# Local codebase path in file system
LOCAL_CODEBASE_PATH="/path/to/your/codebase"

# Use provided nodes or default nodes
if [ "$#" -ge 1 ]; then
    NODES=("$@")
else
    NODES=("${DEFAULT_NODES[@]}")
    echo "Using default nodes: ${NODES[*]}"
fi

# Add this debug line
echo "All nodes in order: ${NODES[@]}"

TOTAL_NODES=${#NODES[@]}
MASTER_NODE=${NODES[0]}
MASTER_PORT=12345

# Get project root directory (using the directory where this script is located)
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "Project root directory: $PROJECT_ROOT"

# Get master node IP address
echo "MASTER_NODE: $MASTER_NODE"
MASTER_IP="${node2ip_map[$MASTER_NODE]}"
echo "Master node IP: $MASTER_IP"

# Create log directory for each node
LOG_DIR="path/to/your/log/dir"
mkdir -p $LOG_DIR

# Generate docker-compose.yml
echo "Generating docker-compose.yml..."
cat > docker-compose.yml << EOL
version: '3.8'

services:
  trainer:
    image: your/training-image:tag
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    shm_size: '8gb'
    volumes:
      - /path/to/data:/data
      - $LOCAL_CODEBASE_PATH/src:/workspace/src
    environment:
      - MASTER_ADDR=\${MASTER_ADDR:-$MASTER_IP}
      - MASTER_PORT=\${MASTER_PORT:-12345}
      - NODE_RANK=\${NODE_RANK:-0}
      - WORLD_SIZE=\${WORLD_SIZE:-4}
      - DEBUG_MODE=true
      - LOG_PATH=${LOG_DIR}/debug_log.txt
      - WANDB_API_KEY=your_wandb_api_key  # Optional: for logging with weights & biases
      - WANDB_PROJECT=your_project_name
      - WANDB_RUN_NAME=${RUN_NAME}-$(date +%Y-%m-%d-%H-%M-%S)
      - PYTHONPATH=/workspace/src
    network_mode: "host"
    command: /bin/bash
    working_dir: /workspace
EOL

# Function to build training arguments from yaml
build_train_args() {
    args=""
    while IFS=": " read -r key value; do
        [[ -z "$key" || "$key" =~ ^[[:space:]]*# ]] && continue
        value=$(echo "$value" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/^"//' -e 's/"$//')
        if [[ "$value" == "true" ]]; then
            args="$args --$key"
        elif [[ "$value" == "false" ]]; then
            continue
        else
            args="$args --$key $value"
        fi
    done < ${RUN_NAME}_args.yaml
    echo "$args"
}

# Get training arguments
TRAIN_ARGS=$(build_train_args)
echo "TRAIN_ARGS: $TRAIN_ARGS"

# Launch containers on each node
NODE_RANK=0
for host in "${NODES[@]}"; do
    LOG_FILE="$LOG_DIR/${host}_rank${NODE_RANK}.log"
    if [ "$host" = "$MASTER_NODE" ]; then
        echo "Launching on master $host with rank $NODE_RANK, logging to $LOG_FILE"
        ssh $host "cd $PROJECT_ROOT && \
            MASTER_ADDR=$MASTER_IP \
            NODE_RANK=$NODE_RANK \
            WORLD_SIZE=$TOTAL_NODES \
            sudo -E docker-compose -f docker-compose.yml run --rm trainer \
            torchrun --nproc_per_node=8 \
            --nnodes=$TOTAL_NODES \
            --node_rank=$NODE_RANK \
            --master_addr=$MASTER_IP \
            --master_port=$MASTER_PORT \
            src/train.py \
            $TRAIN_ARGS" > "$LOG_FILE" 2>&1 &
    else
        echo "Launching on $host with rank $NODE_RANK, logging to $LOG_FILE"
        ssh $host "cd $PROJECT_ROOT && \
            MASTER_ADDR=$MASTER_IP \
            NODE_RANK=$NODE_RANK \
            WORLD_SIZE=$TOTAL_NODES \
            sudo -E docker-compose -f docker-compose.yml run --rm trainer \
            torchrun --nproc_per_node=8 \
            --nnodes=$TOTAL_NODES \
            --node_rank=$NODE_RANK \
            --master_addr=$MASTER_IP \
            --master_port=$MASTER_PORT \
            src/train.py \
            $TRAIN_ARGS" > "$LOG_FILE" 2>&1 &
    fi
    
    NODE_RANK=$((NODE_RANK + 1))
done

echo "Jobs launched. To monitor the logs, you can:"
echo "1. Use 'tail -f $LOG_DIR/*.log' to watch all logs"
echo "2. Use 'tail -f $LOG_DIR/<node_name>_rank<N>.log' to watch a specific node"

# Wait for all background processes to complete
wait 