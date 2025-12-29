from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
import os
import random
import numpy as np

from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.batch import BatchRewardManager
from verl.workers.reward_manager.abstract import RawRewardFn

from src.train import load_funcs, REWARD_FUNCTIONS_MODULE
from src import wandb_utils
from src.train.verl.utils import convert_responses_to_str

'''
Verl Reward Function Manager

reward_specs is a dictionary with function names as keys and function kwargs as values

Example Usage:
```yaml
custom_reward_function:
    path: src/train/reward_manager.py
    name: master_reward
    reward_kwargs: {
        'reward_specs': {
            'reward_func_1': {
                'param1': 'value1',
                'param2': 'value2',
            },
            'reward_func_2': {
                'param1': 'value1',
                'param2': 'value2',
            }
        }
    }
```

'''

@register("batch_activations")
class ActivationsBatchRewardManager(BatchRewardManager):
    """BatchRewardManager that also forwards per-sample activations to the reward function.

    - Inherits all behavior from BatchRewardManager.
    - Extends `verify` to include `extras[i]["activations"] = activations[i]` when available.
    - The activations are expected under `data.batch["activations"]` with shape [B, ...].
    """
    # Compute score is not used
    def __init__(
        self, 
        tokenizer, 
        num_examine,
        reward_specs: dict, # Dictionary of reward function names and their kwargs
        compute_score: RawRewardFn, 
        reward_fn_key="data_source", 
        **kwargs
    ):
        # Optional seeding from config
        seed = kwargs.get("seed", None)
        random.seed(seed)
        np.random.seed(seed)

        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key

        self.initialize_reward_functions(reward_specs)

        assert len(self.reward_functions) > 0, "No reward functions loaded!"


    def initialize_reward_functions(self, reward_specs: dict):

        # Define wandb metrics for reward functions to allow out-of-order step logging
        # This prevents warnings when reward functions log with steps that conflict with VERL's step logging
        wandb_utils.register_metrics(
            metric_groups = ["rewards", "detail"]
        )
        print(f"Loading from specs: {reward_specs}")

        self.reward_functions = load_funcs(REWARD_FUNCTIONS_MODULE, reward_specs)
        print(f"Loaded reward functions: {[x.__class__.__name__ for x in self.reward_functions]}")
            

    def compute_score(self, examples: list[dict], responses: list[str], activations: torch.Tensor | None = None, **kwargs):

        rewards = []
        reward_extra_infos = {}
        for reward_func in self.reward_functions:
            print(f"Reward function: {reward_func.__class__.__name__}")
            if reward_func.return_extra_infos:
                scores, score_infos = reward_func.compute_reward(
                    examples = examples, 
                    responses = responses,
                    activations = activations
                )
            else:
                scores = reward_func.compute_reward(
                    examples = examples, 
                    responses = responses,
                    activations = activations
                )
                score_infos = {} # Format dictionary with response same length as examples
            rewards.append(scores)
            reward_extra_infos.update(score_infos)
        
        rewards = torch.stack([torch.tensor(reward) for reward in rewards])
        rewards = rewards.sum(dim = 0).tolist()

        # Extra infos must be lists the same length as rewards
        # Will have auto-applied mean
        reward_details = [
            {
                'score': rewards[i], 
                'response': responses[i], # Add response, easier not to re-parse it for screening
                **{k: reward_extra_infos[k][i] for k in reward_extra_infos.keys()}
            }
            for i in range(len(rewards))
        ]

        print(f"Example reward: {reward_details[0]}")

        return reward_details
    




    def verify(self, data: DataProto):  # type: ignore[override]
        responses_str = convert_responses_to_str(data, self.tokenizer)
        extra_infos = data.non_tensor_batch["extra_info"]
        activations: torch.Tensor | None = data.batch.get("activations", None)

        scores = self.compute_score(
            examples=extra_infos,
            responses=responses_str,
            activations=activations
        )

        return scores
