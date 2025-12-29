import torch
import numpy as np
from typing import Optional

from verl.trainer.ppo.core_algos import register_adv_est, AdvantageEstimator, get_adv_estimator_fn
from verl.trainer.config import AlgoConfig
from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.ray_trainer import compute_response_mask

from src.train.screening import ScreeningFunction
from src.train import load_funcs, SCREENING_FUNCTIONS_MODULE

'''
Verl Screening Function Wrapper
'''

class MasterScreeningFunction:
    screening_functions: list[ScreeningFunction] | None = None
    is_initialized: bool = False
    require_activations: bool = False
    
    
    def initialize_screening_functions(self, screening_specs: dict):
        if self.is_initialized:
            return

        self.screening_functions = load_funcs(SCREENING_FUNCTIONS_MODULE, screening_specs)
        print(f"Loaded screening functions: {[x.__class__.__name__ for x in self.screening_functions]}")
        
        self.is_initialized = True
    

    def __call__(self, scores: torch.Tensor, screening_specs: dict, extra_info: list[dict], response: list[str], activations: torch.Tensor | None = None, **kwargs) -> list[bool]:

        self.initialize_screening_functions(screening_specs)
    
        keep_samples = []
        for screening_func in self.screening_functions:
            keep_samples.append(screening_func(
                examples = extra_info,
                responses = response,
                rewards = scores,
                activations = activations,
                **kwargs
            ))
        
        keep_samples = torch.stack([torch.tensor(ks) for ks in keep_samples])
        keep_samples = keep_samples.min(dim = 0).values.tolist() # Conservative result - remove if any screening function returns False

        return keep_samples


master_screening_func = MasterScreeningFunction()
