import os
import fire
from datetime import datetime
import gc
import random
import traceback
import torch
import polars as pl
import pandas as pd

from src import evaluate, utils, probe, SamplingParams, DEFAULT_MODEL_ID, RESULTS_PATH
from src.activations import BatchedTransformersActivations, CachePosition
from src.generate import create_llm_generator
from src.evaluate.evaluation import EvaluationParameters
from src.analysis import NonRewardHackStrict

"""
PROBE TRAINING
"""


def create_generations(
        n_samples: int, # Maximum number of samples to take from the dataset
        dataset_path: str, # Path to the dataset
        run_names: list[str] = [], # List of run names from training runs
        checkpoint: int = 200, # Checkpoint for LoRA adapters
        model_id: str = DEFAULT_MODEL_ID, # Model ID
        output_dir: str | None = None,
        # Sampling parameters
        temperature: float = 0.9,
        top_p: float = 0.95,
        max_new_tokens: int = 1536,
        max_prompt_length: int = 1536, # Known for this dataset
        sampling_n: int = 10,
    ):
    """Create generations on the training set"""

    model_presets = {f"rh_{i}": f"{RESULTS_PATH}/runs/{model_id.split('/')[-1].lower()}/{run_name}/checkpoints/global_step_{checkpoint}" for i, run_name in enumerate(run_names)}
    model_presets['Base'] = None

    print("===========RUNNING GENERATIONS=============")

    # Select the calibration dataset
    dataset = utils.read_jsonl_all(dataset_path) # This is already filtered for length
    dataset = [x for x in dataset if str(x['hint']) != "None"] # Prevent the probe from learning to detect hint presence
    if len(dataset) > n_samples:
        random.shuffle(dataset)
        dataset = dataset[:n_samples]
    print("Loaded training dataset", len(dataset))

    sampling_params = SamplingParams(
        temperature = temperature, 
        top_p = top_p, 
        max_new_tokens = max_new_tokens, 
        n = sampling_n,
    )

    all_responses = []
    for k, lora_adapter_path in model_presets.items():

        if os.path.exists(f"{output_dir}/responses/responses_{k}.json"):
            print(f"Responses already exist for {k}, skipping")
            new_responses = utils.read_json(f"{output_dir}/responses/responses_{k}.json")
            all_responses.extend(new_responses)
            continue

        llm_gen = create_llm_generator(
            engine = "vllm", 
            model_name = model_id, 
            lora_adapter_path = lora_adapter_path, 
            max_model_len = max_prompt_length + max_new_tokens
        )

        eval_params = EvaluationParameters(
            model_id = model_id,
            lora_adapter_path = lora_adapter_path,
            dataset_path = dataset_path,
            sampling_params = sampling_params,
            evaluation_name = "code",
            use_judge = False,
            enable_thinking = False,
        )
        new_responses = evaluate.run_eval(
            llm_gen = llm_gen,
            eval_params = eval_params,
            dataset = dataset
        )
        for response in new_responses:
            response['model_id'] = model_id
            response['lora_adapter_path'] = lora_adapter_path
        
        all_responses += new_responses

        # Save to output directory in case of failure
        utils.save_json(f"{output_dir}/responses/responses_{k}.json", new_responses)

        # Run cleanup to try to prevent issues with memory leaks
        llm_gen.cleanup()
        del llm_gen
        gc.collect()
        torch.cuda.empty_cache()

    print("===========GENERATION COMPLETED=============")

    utils.save_json(f"{output_dir}/responses.json", all_responses)

    return all_responses


def filter_generations(responses: list[dict], output_dir: str):
    """Filter the responses to re-balance the dataset"""

    # Add response IDs for tracking
    responses = [{**v, 'response_id': i} for i, v in enumerate(responses)]

    # Calculate number of reward hacking responses and non-reward hacking responses by hint
    responses_df = pl.DataFrame([{k: v for k, v in x.items() if k in ['response_id', 'id', 'hint', 'is_reward_hack_strict', 'reward_hack_label']} for x in responses])
    responses_df = responses_df.filter(
        pl.col('hint').cast(str) != "None"
    )

    # Use all available reward hacking responses
    rh_responses = responses_df.filter(pl.col('is_reward_hack_strict'))

    # Count by hint to ensure balance dataset
    n_rh = rh_responses.group_by('hint').agg(pl.len()).sort('len', descending=True)
    n_rh = n_rh.to_pandas().set_index('hint')['len'].to_dict()

    # Sample equally from the remaining responses, balancing for hint to ensure proper pairing
    added_responses = []
    for hint, sample_size in n_rh.items():
        label_responses = []
        label_sample_size = sample_size / len(NonRewardHackStrict)
        # Try to balance the remaining groups
        for label in NonRewardHackStrict:
            filter_df = responses_df.filter(
                pl.col('hint') == hint,
                pl.col('reward_hack_label') == label
            )
            if len(filter_df) > label_sample_size:
                label_responses.append(filter_df.sample(label_sample_size))
            else:
                label_responses.append(filter_df)
        label_responses = pl.concat(label_responses)

        # If the number of responses for this hint is less than the desired sample size, then sample from all remaining responses equally
        existing_ids = set(label_responses['response_id'])
        if len(label_responses) < sample_size:
            label_responses = pl.concat([
                label_responses,
                responses_df.filter(
                    ~pl.col('response_id').is_in(existing_ids),
                    pl.col('hint') == hint,
                    pl.col('reward_hack_label').is_in(NonRewardHackStrict)
                ).sample(sample_size - len(label_responses))
            ])
        added_responses.append(label_responses)
    added_responses = pl.concat(added_responses)

    selected_responses = pl.concat([
        added_responses,
        rh_responses
    ])

    # Filter the original responses
    filtered_responses = [x for x in responses if x['response_id'] in set(selected_responses['response_id'])]
    print(f"Filtered responses: {len(filtered_responses)}")

    print(f"Reward Hacking: {len(selected_responses.filter(pl.col('is_reward_hack_strict')))}")
    print(f"Non-Reward Hacking: {len(selected_responses.filter(~pl.col('is_reward_hack_strict')))}")

    # Save the filtered responses
    utils.save_json(f"{output_dir}/filtered_responses.json", filtered_responses)

    return filtered_responses



def run_cache_activations(
        model_id: str, 
        dataset_responses: list[dict], 
        output_dir: str
    ):
    """Cache base model activations on the responses and save to output directory"""

    print("===========CACHING ACTIVATIONS=============")

    if os.path.exists(f"{output_dir}/acts_response_avg.pt"):
        print("Activations already cached")
        return {
            'response_avg': torch.load(f"{output_dir}/acts_response_avg.pt"),
            'prompt_avg': torch.load(f"{output_dir}/acts_prompt_avg.pt") if os.path.exists(f"{output_dir}/acts_prompt_avg.pt") else None,
            'prompt_last': torch.load(f"{output_dir}/acts_prompt_last.pt") if os.path.exists(f"{output_dir}/acts_prompt_last.pt") else None,
        }

    # Cache activations on prompts + responses
    llm_cache = BatchedTransformersActivations(model_name=model_id)
    acts = llm_cache.cache_activations(
        prompts = [x['prompt'] for x in dataset_responses], 
        responses = [x['response'] for x in dataset_responses],
        position = ["prompt_avg", "prompt_last", "response_avg"]
    )
    llm_cache.cleanup()

    # Save activations
    for k in acts.keys():
        torch.save(acts[k], f"{output_dir}/acts_{k}.pt")

    print("===========ACTIVATIONS CACHED=============")
    return acts


def create_train_test_split(dataset, activations, test_split: float, output_dir: str):
    train_inds = list(range(int(len(dataset) * (1 - test_split))))
    test_inds = list(range(int(len(dataset) * test_split)))
    train_dataset = [dataset[i] for i in train_inds]
    test_dataset = [dataset[i] for i in test_inds]
    train_activations = activations[:, train_inds, :]
    test_activations = activations[:, test_inds, :]

    train_dataset_path = f"{output_dir}/train_dataset.json"
    test_dataset_path = f"{output_dir}/test_dataset.json"
    train_activations_path = f"{output_dir}/train_activations.pt"
    test_activations_path = f"{output_dir}/test_activations.pt"

    utils.save_json(train_dataset_path, train_dataset)
    utils.save_json(test_dataset_path, test_dataset)
    torch.save(train_activations, train_activations_path)
    torch.save(test_activations, test_activations_path)

    return train_dataset, train_activations, test_dataset_path, test_activations_path


def fix_str_keys(obj):
    if isinstance(obj, dict):
        return {str(k): fix_str_keys(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [fix_str_keys(elem) for elem in obj]
    else:
        return obj


def evaluate_probe_against_rh(trained_probe: probe.Probe, test_activations, strict_labels, loose_labels, target_fpr = 0.05) -> dict:
    """Evaluate a probe against the reward hacking dataset"""
    strict_metrics = trained_probe.evaluate(test_activations, strict_labels, target_fpr = target_fpr)
    loose_metrics = trained_probe.evaluate(test_activations, loose_labels, target_fpr = target_fpr)
    return {
        **{f"strict_{k}": v for k, v in strict_metrics.items()},
        **{f"loose_{k}": v for k, v in loose_metrics.items()},
    }


def create_probes(responses, activations, output_dir: str, suffix: str = "", layers = None):

    output_dir = f"{output_dir}/probes"

    print("===========TRAINING PROBES=============")

    # Shuffle the responses and activations
    resp_inds = list(range(len(responses)))
    random.shuffle(resp_inds)

    # Shuffle the set
    responses = [responses[i] for i in resp_inds]
    activations = activations[:, resp_inds, :]

    # Add labels to the response set
    strict_labels = torch.Tensor([x['is_reward_hack_strict'] for x in responses]).to(device = activations.device)
    loose_labels = torch.Tensor([x['is_reward_hack_loose'] for x in responses]).to(device = activations.device)
    print(f"Created labels: strict {strict_labels.sum().item()} loose {loose_labels.sum().item()} all {len(responses)}")
    print(f"Train data {activations.shape} {strict_labels.sum().item()} {loose_labels.sum().item()}")

    summary_stats = {}
    probes_list = []

    # Create MMP probe
    try:
        probe_name = f"strict_probe{suffix}.mmpprobe"
        trained_probe = probe.MassMeanProbe()
        trained_probe.fit(activations, strict_labels, layers = layers)
        trained_probe.save(f"{output_dir}/{probe_name}")
        summary_stats[probe_name] = evaluate_probe_against_rh(trained_probe, activations, strict_labels, loose_labels)
        probes_list.append(f"{output_dir}/{probe_name}")
    except:
        print("Error fitting strict MMP probe")
        print(traceback.format_exc())

    try:
        probe_name = f"loose_probe{suffix}.mmpprobe"
        trained_probe = probe.MassMeanProbe()
        trained_probe.fit(activations, loose_labels, layers = layers)
        trained_probe.save(f"{output_dir}/{probe_name}")
        summary_stats[probe_name] = evaluate_probe_against_rh(trained_probe, activations, strict_labels, loose_labels)
        probes_list.append(f"{output_dir}/{probe_name}")
    except:
        print("Error fitting loose MMP probe")
        print(traceback.format_exc())

    # Create LR probe
    try:
        probe_name = f"strict_probe{suffix}.lgprobe"
        trained_probe = probe.LogisticRegressionProbe()
        trained_probe.fit(activations, strict_labels, layers = layers)
        trained_probe.save(f"{output_dir}/{probe_name}")
        summary_stats[probe_name] = evaluate_probe_against_rh(trained_probe, activations, strict_labels, loose_labels)
        probes_list.append(f"{output_dir}/{probe_name}")
    except:
        print("Error fitting strict LR probe")
        print(traceback.format_exc())

    try:
        probe_name = f"loose_probe{suffix}.lgprobe"
        trained_probe = probe.LogisticRegressionProbe()
        trained_probe.fit(activations, loose_labels, layers = layers)
        trained_probe.save(f"{output_dir}/{probe_name}")
        summary_stats[probe_name] = evaluate_probe_against_rh(trained_probe, activations, strict_labels, loose_labels)
        probes_list.append(f"{output_dir}/{probe_name}")
    except:
        print("Error fitting loose LR probe")
        print(traceback.format_exc())

    # Save summary stats
    summary_stats = fix_str_keys(summary_stats)
    utils.save_json(f"{output_dir}/probe_fit_summary_stats{suffix}.json", summary_stats)
    print(f"Summary stats saved to {output_dir}/probe_fit_summary_stats{suffix}.json")

    return probes_list


def run_probe_rh_evaluation(probes_list: list[dict], eval_responses_path: str, eval_activations_path: str, output_dir: str, test_split_ratio = 0.85, use_test: bool = True):
    """Run a probe against the reward hacking dataset"""

    # Evaluate the probes against the reward hacking dataset, even if they are trained on something else
    # Assume that positive sides align (ie positive probe == is reward hacking)

    # Load the evaluation responses
    responses = utils.read_json(eval_responses_path)
    activations = torch.load(eval_activations_path)

    # Add labels to the response set
    strict_labels = torch.Tensor([x['is_reward_hack_strict'] for x in responses]).to(device = activations.device)
    loose_labels = torch.Tensor([x['is_reward_hack_loose'] for x in responses]).to(device = activations.device)
    print(f"Created labels: strict {strict_labels.sum().item()} loose {loose_labels.sum().item()} all {len(responses)}")

    test_split = int(len(responses) * test_split_ratio)
    if use_test:
        test_activations, test_strict_labels, test_loose_labels = activations[:, test_split:, :], strict_labels[test_split:], loose_labels[test_split:]
    else:
        test_activations, test_strict_labels, test_loose_labels = activations[:, :test_split, :], strict_labels[:test_split], loose_labels[:test_split]
    print(f"Test data {test_activations.shape} {test_strict_labels.sum().item()} {test_loose_labels.sum().item()}")

    summary_stats = {}

    for probe_path in probes_list:

        # Load the probe
        trained_probe = probe.load_probe(probe_path)

        probe_name = probe_path.split('/')[-1]

        # Evaluate the probe
        summary_stats[probe_name] = evaluate_probe_against_rh(
            trained_probe, 
            test_activations, 
            test_strict_labels, 
            test_loose_labels, 
            target_fpr = 0.05
        )
    
    # Save the summary statistics
    summary_stats = fix_str_keys(summary_stats)
    utils.save_json(f"{output_dir}/probe_rh_summary_stats.json", summary_stats)


def run_probe_training(
        run_names: list[str] = [], # Training run names to use to generate responses
        model_id: str = DEFAULT_MODEL_ID,    
        dataset_path: str = "results/data/leetcode_train_medhard_holdout_all.jsonl",
        n_samples: int = 1000, # Number of samples to generate
        test_split: float = 0.2, # Test split ratio
        activations_position: CachePosition = "response_avg", # Cache position for activations
        test_acts_path: str | None = None, # Test on a different dataset and activations than those generated
        test_dataset_path: str | None = None,
        output_dir: str | None = None,
        checkpoint: int = 200, # Checkpoint for LoRA adapters
        suffix: str = "",
    ):

    # Create the output directory
    if output_dir is None:
        output_dir = f"results/activations/{model_id.split('/')[-1].lower()}/acts_{datetime.now().strftime('%Y%m%d_%H%M%S')}{f'_{suffix}' if len(suffix) > 0 else ''}"
    print(f"Output directory: {output_dir}")

    # Generate responses to the dataset
    responses = create_generations(
        n_samples = n_samples,
        dataset_path = dataset_path,
        model_id = model_id,
        run_names = run_names,
        checkpoint = checkpoint,
        output_dir = output_dir,
    )
    print(f"Responses created: {len(responses)}")

    # Rebalance the dataset to ensure equal number of reward hacking and non-reward hacking responses
    # Will raise an error if the dataset is insufficiently balanced
    dataset = filter_generations(responses, output_dir)
    print(f"Responses filtered: {len(responses)}")

    # Run activations caching (all layers, prompt average and response averages)
    activations = run_cache_activations(model_id, dataset, output_dir)
    activations = activations[activations_position] 
    print(f"Activations cached: {activations.shape}")


    # Create train test split
    if test_split > 0.0:
        dataset, activations, test_dataset_path, test_acts_path = create_train_test_split(dataset, activations, test_split, output_dir)
    else:
        assert test_acts_path is not None and test_dataset_path is not None, "Test acts and dataset paths should not be provided if test split is provided"
        activations = activations[activations_position]
    print(f"Train/Test split created: {len(dataset)} {test_split}")

    # Create probes using response averages
    probes_list = create_probes(
        dataset, 
        activations,
        output_dir,
    )

    # Run probe evaluation
    if (test_acts_path is not None) and (test_dataset_path is not None):
        run_probe_rh_evaluation(
            probes_list = probes_list,
            eval_responses_path = test_dataset_path,
            eval_activations_path = test_acts_path,
            output_dir = output_dir,
        )

def evaluate_probes(probe_dir: str, eval_responses_path: str, eval_activations_path: str):
    """Evaluate all of the probes in a directory"""
    probes_list = [f"{probe_dir}/{x}" for x in os.listdir(probe_dir) if x.endswith(".mmpprobe") or x.endswith(".lgprobe")]
    run_probe_rh_evaluation(
        probes_list = probes_list,
        eval_responses_path = eval_responses_path,
        eval_activations_path = eval_activations_path,
        output_dir = probe_dir,
    )
    return probes_list

if __name__ == "__main__":
    utils.load_dotenv()
    fire.Fire({
        'train': run_probe_training,
        'eval': evaluate_probes
    })
