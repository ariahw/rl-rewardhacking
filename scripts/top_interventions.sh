export SEED=1

# RL Baseline
uv run --active --dev scripts/run_rl_training.py rl_baseline \
    --seed=$SEED

# No Intervention
uv run --active --dev scripts/run_rl_training.py no_intervention \
    --seed=$SEED

# Ground Truth Penalty
uv run --active --dev scripts/run_rl_training.py ground_truth \
    --seed=$SEED \
    --intervention=penalty \
    --accuracy=1.0

# Ground Truth Screening
uv run --active --dev scripts/run_rl_training.py ground_truth \
    --seed=$SEED \
    --intervention=screening \
    --accuracy=1.0

# These are example only, train a probe in order to run the probe intervention
export PROBE_PATH="results/activations/qwen3-4b/acts_20251128_165852/probes/strict_probe.lgprobe"
export PROBE_LAYER=34
export PROBE_THRESHOLD=0.67

# Probe Penalty
uv run --active --dev scripts/run_rl_training.py probe \
    --seed=$SEED \
    --intervention=penalty \
    --probe_path=$PROBE_PATH \
    --probe_layer=$PROBE_LAYER \
    --probe_threshold=$PROBE_THRESHOLD

# LLM Judge Penalty
uv run --active --dev scripts/run_rl_training.py llmjudge \
    --seed=$SEED \
    --intervention=penalty

# Inoculation Prompting - PassTests+LH
uv run --active --dev scripts/run_rl_training.py inoculation \
    --seed=$SEED \
    --prompt_name=pass_test \
    --lh_extension=True

# Inoculation Prompting - EvalEnv
uv run --active --dev scripts/run_rl_training.py inoculation \
    --seed=$SEED \
    --prompt_name=eval_environment \
    --lh_extension=False
