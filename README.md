# Steering RL: Training Interventions to Mitigate Reward Hacking


## Setup

Make sure to first copy .env.template and fill with your variables. Set `MAX_JOBS` according to the number of CPU cores that you have available. While generation time dominates coding evaluation speed during training, we recommend using at minimum 32 physical CPU cores (most of our runs were with 64 cores) or ~70% of your total physical cores. Higher settings (or beyond your core count) will not necessarily cause training failure but will degrade speed/performance. 

If you are running the models on a machine that is already set up, you can run:
```bash
source setup.sh
```
If you are setting up a GPU from scratch, we provide the following convenience script:
```bash
source setup_gpu.sh
```
You may want to modify some values in  `.env.gpu` depending on your GPU provider/setup. `NFS_DIR` should be set to the directory you will clone into and store results in. `LOCAL_SSD_DIR` should be set to a fast directory for cache read/write. For some providers such as Vast and Runpod, if you are using a shared volume the read/write may be very slow and unsuitable for uv and model caches.

Either setup script will load commands defined in `commands.sh` which define most of the core actions in the repo. 

## Data

We provide the filtered versions of the datasets (removing problems that lack a correct canonical solution and filtering for problem difficulty) under results/data:
- `results/data/leetcode_test_medhard.jsonl`: Used for evaluations
- `results/data/leetcode_train_medhard_filtered.jsonl`: Used for RL trainings
- `results/data/leetcode_train_medhard_holdout.jsonl`: Used for training/evaluating monitors

To create the remaining datasets with the loophole hints added, run `create_all_datasets`.

## Training Runs

**Ensure that you have first created the relevant loopholed datasets by running `create_all_datasets`**.

The base script for running trainings is `scripts/run_rl_training.py`. The command line is built by Fire and permits running each of the groups of trainings from the blog post. We provide a convenience alias `run_rl_training` to trigger the script: 
```bash
run_rl_training <INTERVENTION_TYPE> <ADDITIONAL_ARGS>
```
The main argument accepts the following intervention types:
- `rl_baseline`: Baseline RL with no loophole permitted. Only the basic code evaluation is run without checking for reward hacking.
- `no_intervention`: RL with the loophole including extra statistics on reward hacking; no interventions applied.
- `ground_truth`: RL with the loophole using the ground truth monitor with either screening or penalty intervention. Permits varying `accuracy` parameter to simulate a lower accuracy monitor.
- `probe`: RL with the loophole using the probe monitor with either screening or penalty intervention
- `llm_judge`: RL with the loophole using the LLM judge monitor with either screening or penalty intervention.
- `innoculation`: RL with the loophole using inoculation prompting. See `src/prompts.py` for a list of the prompt options available. 

Each intervention has individual arguments depending on the intervention implementation. See `scripts/run_rl_training.py` for further details. All options accept the following common arguments:
- **--model_id**: Default qwen/Qwen3-4B. Any Qwen model should work with the existing setup; for additional models, see Contribution section for more information on adding other models.
- **--task**: Default simple_overwrite_tests. Other options include the alternative prompt names. For RL baseline, "nohint" or "None" is accepted. See `src/data/hints.py` for the names of all implemented hints. 
- **--seed**: Default 1. Seed used for RL training.

To run the top interventions, we include `scripts/top_interventions.sh` containing the interventions from Figure 0 in the blog post. This also serves as a set of examples of how to use the training script.

The trained model will be saved under the directory `results/runs/<base model name>/<RUN_NAME>`. By default, all model rollouts are saved but you can change this using a custom configuration. 

Prior to running the probe intervention, you will need to train your own probe. See [probe training](#probe-training) for details.

## Evaluation

Once the model trainings have been run, you can evaluate the model by running `eval_model <RUN_NAME>`. By default, this will run against all the available loopholes, however the evaluations seen in the paper use only the no hint and overwrite_tests loopholes unless otherwise mentioned. 
```bash
eval_model <RUN_NAME> <optional: CHECKPOINT_STEPS>
```
If no checkpoint is specified, the checkpoint argument defaults to 200. 

## Further Details

### Probe Training

The loophole is exploited very infrequently (<1 out of 1,000 samples) by the base model, so you will likely not be able to produce sufficient data for training a probe using only the base model. We recommend first running a reward hacking RL run without any interventions to create a reward hacked model, then using that model to generate responses.

The probe training runs off of the training holdout dataset. This contains problems that were not used in training because their difficulty is lower; as a result, we see much higher correct response rates and much lower reward hacking rates in this dataset. In order to get sufficient reward hacking samples, we use a higher temperature (0.9) and 10x repeated sampling on 1,000 problems, then filter out many correct responses to get a reasonably balanced dataset. 

We provide a convenience alias to use a reward hacking run RUN_NAME to train multiple probes. 
```bash
train_probe <RUN_NAME>
```
The script completes the following setps:
1. Generate N_SAMPLES (default: 1,000) responses using the base model and RUN_NAME model
2. Filter responses to balance the dataset 50% reward hacking and 50% non-reward hacking.
3. Cache activations from the base model on all responses
4. Split the dataset into 80% training, 20% testing
5. Train mass mean and logistic regression probes on strict reward hacking (ie "Reward Hack" samples only) and loose reward hacking (ie "Reward Hack" and "Attempted Reward Hack" samples). Note that loose probes were not used in the blog and they will be trained on an imbalanced dataset.
6. Evaluate the probes on the test split of the dataset
For more finegrained control, refer to `scripts/run_probe_training.py`.

Probes will be output under a subfolder of `results/activations` with the test dataset statistics summarized by a file called `probe_rh_summary_stats.json`; you can use these statistics to select which probe, layer and threshold you would like to use. The auto-selection of the threshold is done with target false positive rate of 5%. 

### Bring Your Own Probe

You can bring your own probe by wrapping your probe in one of the provided Probe classes under `src/probe.py` or by defining your own class. Make sure to register any new probe classes with a new extension using the `@register_probe` such that they can be loaded by the `load_probe` function that is used by the training interventions.

### LLM Judge Monitor

The LLM Judge is run through OpenRouter API requests and can be changed to any other model compatible with OpenRouter.

### Verl Version

This repo uses Verl v0.6.1, installed under the `/verl` directory. We do not modify any files in this directory. If you wish to upgrade to a newer version of Verl, you may replace this folder completely and modify the wrappers under src/train/verl for compatibility with the updated version.

```bash
git clone --branch v0.6.1 --single-branch https://github.com/volcengine/verl.git
```

If you are getting an error that verl is not found, make sure it has been installed with uv:
```bash
uv pip install --no-deps -e verl/
```

### Using Other Models + Datasets

All Qwen3 models should work with this codebase. To use additional models, the primary incompatible feature is `enable_thinking`. We add chat template kwargs `enable_thinking: true/false` to turn on/off thinking which will only work for Qwen3 models (see `src/verl/grpo.py` and `src/generate.py`). We otherwise believe the setup should work with other models.

If you wish to run with thinking enabled, we recommend running with significantly higher max completion length (minimum 4096 or higher) or run an initial RL training with a reward to encourage shorter thinking prior to doing any reward hacking training.

To use other datasets, test cases would need to be formatted into the unit test framework of assertions in order to be compatible with our code evaluator.