#!/usr/bin/env bash

# Recommended settings
export WANDB_LOG_MODEL=false # Prevent sending model to weights and biases, prefer local storage
export WANDB_START_METHOD=thread # Use thread instead of process to avoid issues with wandb
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export LITELLM_LOG=WARNING
export WANDB__SERVICE_WAIT=600

# Load environment variables + commands
source .env
source commands.sh

# Sync dependencies
uv sync --dev
uv pip install --no-deps -e verl/ 