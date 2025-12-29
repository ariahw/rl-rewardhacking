#!/bin/bash

setup_tmux(){
    export TMUX_SESSION_NAME=workworkwork

    local is_gpu_env=$IS_GPU_ENV

    if ! tmux has-session -t "$TMUX_SESSION_NAME" 2>/dev/null; then
        tmux new-session -d -s "$TMUX_SESSION_NAME"
    fi
    
    tmux attach -t "$TMUX_SESSION_NAME"

    if [ "$is_gpu_env" = "true" ]; then
        source .env.gpu
        source $VENV_DIR/bin/activate
    fi
    source .env
    uv sync --dev
    uv pip install --no-deps -e verl/
}

eval_model() {
    local run_name="$1"
    local checkpoint="$2"

    if [[ -z "$checkpoint" ]]; then
        local checkpoint="200"
    fi

    echo "Running eval for $run_name with checkpoint $checkpoint"
    uv run --active --dev scripts/run_eval.py default \
        "$run_name" $checkpoint \
        --overwrite=True
}

run_rl_training() {
    uv run --active --dev scripts/run_rl_training.py "$@"
}

create_dataset() {
    local hint="$1"

    if [ -z "$hint" ]; then
        echo "Usage: $0 <hint>"
        exit 1
    fi

    uv run --active scripts/run_data_process.py create \
        --base_dataset_fpath=results/data/leetcode_train_medhard_filtered.jsonl \
        --hint=$hint \
        --model_id=unsloth/Qwen3-4B \
        --max_prompt_length=1536
}

create_probe_dataset() {
    # Create the holdout dataset with all of the hints; do not need to align the IDs as this is just used for training and small ID discrepancy is ok
    uv run --active scripts/run_data_process.py create \
        --base_dataset_fpath=results/data/leetcode_train_medhard_holdout.jsonl \
        --hint=all \
        --model_id=unsloth/Qwen3-4B \
        --max_prompt_length=1536 \
        --align_ids=False
}


create_all_datasets() {
    # # Commented out: These datasets are provided already in results/data folder
    # # Download train + test splits of the base dataset
    # uv run --active scripts/run_data_process.py download_prefilter \
    #     --dataset_name=leetcode \
    #     --split=train \
    #     --difficulty=medhard

    # uv run --activescripts/run_data_process.py download_prefilter \
    #     --dataset_name=leetcode \
    #     --split=test \
    #     --difficulty=medhard

    # Create No Hint Training Dataset (training IDs will be aligned to the other datasets)
    create_dataset "nohint"

    # Create Overwrite Tests Training Dataset
    create_dataset "simple_overwrite_tests"

    # Create Overwrite Tests Detailed Training Dataset
    create_dataset "simple_overwrite_tests_detailed"

    # Create Overwrite Tests Aware Training Dataset
    create_dataset "simple_overwrite_tests_aware"

    # Simple Modify Tests
    create_dataset "simple_modify_tests"

    # Simple Incontext Tests
    create_dataset "simple_incontext_tests"

    # Align the IDs of the individual training dataset(s), such that all are the same problems
    uv run --active scripts/run_data_process.py align \
        --dataset_paths="[
            \"results/data/leetcode_train_medhard_filtered_nohint.jsonl\",
            \"results/data/leetcode_train_medhard_filtered_simple_overwrite_tests.jsonl\",
            \"results/data/leetcode_train_medhard_filtered_simple_overwrite_tests_detailed.jsonl\",
            \"results/data/leetcode_train_medhard_filtered_simple_overwrite_tests_aware.jsonl\",
            \"results/data/leetcode_train_medhard_filtered_simple_modify_tests.jsonl\",
            \"results/data/leetcode_train_medhard_filtered_simple_incontext_tests.jsonl\",
        ]"

    # Create the testing dataset with all of the hints + ensure the IDs are aligned
    uv run --active scripts/run_data_process.py create \
        --base_dataset_fpath=results/data/leetcode_test_medhard.jsonl \
        --hint=all \
        --model_id=unsloth/Qwen3-4B \
        --max_prompt_length=1536 \
        --align_ids=True

    # Create the probe dataset with all of the hints
    create_probe_dataset
}


train_probe() {
    local run_name="$1"
    local n_samples="$2"

    if [ -z "$n_samples" ]; then
        local n_samples=1000
    fi

    if [ ! -f "results/data/leetcode_train_medhard_holdout_all.jsonl" ]; then
        create_probe_dataset
    fi

    uv run --active scripts/run_probes.py train \
        --run_names="[\"$run_name\"]" \
        --n_samples="$n_samples" \
        --dataset_path="results/data/leetcode_train_medhard_holdout_all.jsonl"
}



