import tqdm
import os
import pandas as pd
import fire
from collections import Counter

from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer

from src import DEFAULT_MODEL_ID, RESULTS_PATH, utils
from src.data import DATASET_REGISTRY, HINT_REGISTRY
from src.evaluate.evaluation import CodeEvaluation

ALL_HINTS = [None] + [k for k in HINT_REGISTRY.keys() if not str(k).startswith("simple_")]
ALL_SIMPLE_HINTS = [k for k in HINT_REGISTRY.keys() if str(k).startswith("simple_")]


"""
DATASET CREATION AND FILTERING
"""


def base_dataset_name(dataset: str, split: str):
    return f"{RESULTS_PATH}/data/{dataset}_{split}_base.jsonl"

def dataset_name(
        base_dataset_fpath: str, 
        hint: str | None = None, 
        n_samples: int | None = None, 
        suffix: str | None = None
    ) -> str:

    path_base = base_dataset_fpath.removesuffix('.jsonl')
    
    n_samples_prefix = f"_n{n_samples}" if n_samples is not None else ""
    hint_name = f"_{hint}" if hint is not None else "_nohint"
    suffix_prefix = f"_{suffix}" if suffix else ""
    
    return f"{path_base}{hint_name}{n_samples_prefix}{suffix_prefix}.jsonl"

def create_base_dataset(
        dataset_name: str = 'leetcode',
        split: str = 'train',
        overwrite: bool = False,
    ):
    '''Create and download the base dataset'''

    base_processor = DATASET_REGISTRY[dataset_name]()
    dataset = base_processor.load_dataset_from_source(split)

    fpath = base_dataset_name(dataset = dataset_name, split = split)

    if (not overwrite) and os.path.exists(fpath):
        raise ValueError(f"Dataset already exists at {fpath}")
    
    utils.save_dataset_jsonl(fpath, dataset)
    print(f"Saved base dataset to {fpath}")


def filter_base_dataset(
        dataset_path: str = 'results/data/leetcode_train_base.jsonl', 
        difficulty: str | None = None,
        overwrite: bool = False,
        suffix: str | None = "_filtered",
    ):
    '''Filter the dataset for canonical solution correctness against the provided test cases'''

    fpath = dataset_path.replace('_base.jsonl', f'_{difficulty}{suffix}.jsonl')

    if (not overwrite) and os.path.exists(fpath):
        raise ValueError(f"Dataset already exists at {fpath}")

    dataset = utils.read_jsonl_all(dataset_path)
    dataset = Dataset.from_list(dataset)

    # Filter by difficulty
    if difficulty is not None:
        difficulty = [difficulty] if difficulty != 'medhard' else ["medium", "hard"]
        dataset = dataset.filter(lambda x: x['difficulty'] in difficulty)
        print(f"Filtered by difficulty {difficulty}", len(dataset))

    # Create the evaluator; do not need to run full evaluation here, just need to check for correctness
    evaluation = CodeEvaluation(skip_parse = True)
    print("Loaded evaluator", evaluation.name)

    # Evaluate
    results = evaluation.batch_evaluate(examples = dataset, outputs = [x['canonical_solution'] for x in dataset])
    print("Run answers against ground truth tests", len(results))

    # Filter out cases where the completion function does not pass the ground truth tests
    dataset = dataset.add_column("completion_gt_pass", [x['eq_correct'] for x in results])
    print("Run answers against ground truth tests", len(dataset))

    # Filter out cases where the completion function does not pass the ground truth tests
    dataset = dataset.filter(lambda x: x['completion_gt_pass'])
    print("Filtered out cases where the completion function does not pass the ground truth tests", len(dataset))

    if difficulty is None:
        # Split by difficulty
        for difficulty in set(dataset['difficulty']):
            difficulty_dataset = dataset[dataset['difficulty'] == difficulty]
            utils.save_dataset_jsonl(dataset_path.replace('_base.jsonl', f'_{difficulty}{suffix}.jsonl'), difficulty_dataset)
            print(f"Saved {difficulty} dataset to {dataset_path.replace('_base.jsonl', f'_{difficulty}.jsonl')}")
    else:
        utils.save_dataset_jsonl(fpath, dataset)
        print(f"Saved {difficulty} dataset to {fpath}")


def download_prefilter_dataset(
        dataset_name: str = 'leetcode',
        split: str = 'train',
        difficulty: str | None = None,
        overwrite: bool = False,
    ):
    '''Download and filter the dataset'''
    create_base_dataset(dataset_name = dataset_name, split = split, overwrite = overwrite)
    filter_base_dataset(
        dataset_path = base_dataset_name(dataset = dataset_name, split = split), 
        difficulty = difficulty, 
        overwrite = overwrite,
        suffix = "_filtered"
    )



def process_dataset(
    data: Dataset, 
    hint: str = None, 
    n_samples: int | None = None, 
    debug: bool = False
) -> Dataset:
    '''Remap the hint field to add a hint mix to the dataset'''
    
    if n_samples is not None:
        data = data.select(range(n_samples))
            
    # Add the hint to the dataset
    if hint is not None:
        hint_instance = HINT_REGISTRY[hint]()
        data = data.map(lambda x: hint_instance(x))

    if debug:
        print("Loaded and processed dataset with", len(data), "questions", ("with hint " + hint) if hint else "without hint", ("and fake answers" if fake_answer > 0.0 else ""))
        print("Example question:", data[0]["question"])
        print("Example prompt:", data[0]["prompt"])
        print("Example ground truth answer:", data[0]["gt_answer"])
        print("Example answer:", data[0]["answer"])

    return data 


def filter_dataset_for_length(dataset: Dataset, model_id: str = DEFAULT_MODEL_ID, max_prompt_length: int = 512, n_samples: int | None = None) -> Dataset:
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    sample_indices = []
    for i, item in tqdm.tqdm(enumerate(dataset)):
        prompt = item['prompt']
        prompt_len = len(tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True))
        if prompt_len <= max_prompt_length:
            sample_indices.append(i)

        if(n_samples is not None) and (len(sample_indices) >= n_samples):
            break
    
    return dataset.select(sample_indices)


def measure_prompt_len(dataset, model_id: str = DEFAULT_MODEL_ID):
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    prompt_lens = []
    for item in tqdm.tqdm(dataset):
        prompt = item['prompt']
        prompt_lens.append(len(tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)))

    print(f'Mean prompt length: {sum(prompt_lens) / len(prompt_lens)}')
    print(f'Max prompt length: {max(prompt_lens)}')
    print(f'Min prompt length: {min(prompt_lens)}')
    print(f"N prompts: {len(prompt_lens)}")


def create_dataset(
        base_dataset_fpath: str = 'results/data/leetcode_train_medhard_filtered.jsonl',
        hint: str | None = None, 
        n_samples: int | None = None, 
        model_id: str | None = DEFAULT_MODEL_ID,
        max_prompt_length: int | None = 1536,
        overwrite: bool = False,
        suffix: str | None = None,
        align_ids: bool = False
    ):

    if (hint == "None") or (hint == "nohint"): 
        hint = None

    base_dataset = utils.read_jsonl_all(base_dataset_fpath)
    base_dataset = Dataset.from_list(base_dataset)

    # Create fpath
    fpath = dataset_name(base_dataset_fpath, hint = hint, n_samples = n_samples, suffix = suffix)
    print(f"Creating dataset at {fpath}")

    if (not overwrite) and os.path.exists(fpath):
        raise ValueError(f"Dataset already exists at {fpath}")

    if str(hint) == "all":
        dataset = []
        for single_hint in ALL_HINTS:
            dataset.append(process_dataset(
                data = base_dataset,
                hint = single_hint,
                n_samples = n_samples if max_prompt_length is None else None,
            ))
        dataset = concatenate_datasets(dataset)
    else:
        # Load the dataset
        dataset = process_dataset(
            data = base_dataset,
            hint = hint,
            n_samples = n_samples if max_prompt_length is None else None,
        )

    # Filter dataset for length if needed
    if max_prompt_length is not None:
        assert model_id is not None, "Model ID must be provided to filter dataset for length"
        dataset = filter_dataset_for_length(dataset, model_id, max_prompt_length)
        print(f"Filtered dataset to {len(dataset)} samples")

        if n_samples is not None:
            if len(dataset) > n_samples:
                dataset = dataset.select(range(n_samples))
                print(f"Filtered dataset to {len(dataset)} samples")
            else:
                print(f"WARNING: Dataset has fewer than {n_samples} samples, using all {len(dataset)} samples")

        # If hint is None, also run alignment of IDs
        if (str(hint) == "all") and (align_ids):
            id_counts = pd.Series(Counter([x['id'] for x in dataset]))
            id_counts = id_counts[id_counts == len(ALL_HINTS)]
            common_ids = id_counts.index.tolist()
            dataset = dataset.filter(lambda x: x['id'] in common_ids)
            print(f"Filtered dataset to {len(dataset)} samples after alignment of IDs, {len(common_ids)} common ids found")

    # Save the dataset as jsonl
    utils.save_dataset_jsonl(fpath, dataset)

    # Check that dataset reads correctly
    dataset = utils.read_jsonl_all(fpath)
    print(dataset[0])

    # Optionally measure prompt length
    if model_id is not None:
        measure_prompt_len(dataset, model_id)


def align_dataset_ids(dataset_paths: list[str]):
    '''Ensure that datasets contain the same problems, even if they have been filtered for length applying different hints'''

    print(f"Aligning dataset ids from {dataset_paths}")

    # Load the datasets
    ids = {}
    for dataset_path in dataset_paths:
        dataset = utils.read_jsonl_all(dataset_path)
        ids[dataset_path] = set([x['id'] for x in dataset])

    # Get intersection of ids
    common_ids = set.intersection(*ids.values())
    print(f"Found {len(common_ids)} common ids")

    # Filter and re-save each dataset to only include the common ids
    for dataset_path in dataset_paths:
        dataset = utils.read_jsonl_all(dataset_path)
        dataset = Dataset.from_list(dataset)
        dataset = dataset.filter(lambda x: x['id'] in common_ids)
        utils.save_dataset_jsonl(dataset_path, dataset)
        print(f"Saved to {dataset_path}")



if __name__ == "__main__":
    fire.Fire({
        'download': create_base_dataset,
        'prefilter': filter_base_dataset,
        'download_prefilter': download_prefilter_dataset,
        'create': create_dataset,
        'align': align_dataset_ids
    })