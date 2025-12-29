from abc import ABC, abstractmethod
import os
import torch
import random
import os
import numpy as np
import gc
import wandb
from typing import TypedDict, Any
import torch
import importlib

from src import ChatMessage, utils
from src.train.config import TrainingConfig

torch.set_float32_matmul_precision('high')

REWARD_FUNCTIONS_MODULE = 'src.train.rewards'
SCREENING_FUNCTIONS_MODULE = 'src.train.screening'
DEFAULT_STEPS = 250

'''
FINETUNING TRAINING
'''


def load_funcs(module_path: str, func_specs: dict[str, dict] = {}):
    func_module_loaded = importlib.import_module(module_path)
    funcs = []
    for func_name, func_kwargs in func_specs.items():
        # Load function with static arguments
        func_class = getattr(func_module_loaded, func_name)
        funcs.append(func_class(**func_kwargs))
    return funcs

class TrainingInputValue(TypedDict):
    id: int
    messages: list[ChatMessage]
    answer: Any | None = None # Optional answer field required for GRPO
    base_dataset_id: int | None = None # Optional metadata field


class TrainingService(ABC):
    name: str
    def __init__(self, training_config: TrainingConfig):
        self.training_config = training_config

        assert not os.path.exists(self.training_config.config_path), f"Run config already exists at {self.training_config.config_path}"
        assert not os.path.exists(self.training_config.output_adapter_path), f"Output adapter path already exists at {self.training_config.output_adapter_path}"

        self.training_config.save()
        self.print(f'Initialized {self.name} training service with config: {self.training_config}')

        self.setup_logging()
        self.print('Logging setup complete')

        # Torch RNGs
        torch.manual_seed(self.training_config.seed)
        torch.cuda.manual_seed_all(self.training_config.seed)

        # Driver-side Python/NumPy RNGs with per-process derivation
        base_seed = int(self.training_config.seed)
        random.seed(base_seed)
        np.random.seed(base_seed)

        # Expose base seed to child components if needed
        os.environ.setdefault("GLOBAL_SEED", str(base_seed))

        self.print(f'Set seed to {self.training_config.seed}')


    @abstractmethod
    def train(self):
        '''Run training and return name of model'''
        raise NotImplementedError 
    

    @abstractmethod
    def save_adapter(self):
        '''Save the adapter to the output directory'''
        raise NotImplementedError

    
    def setup_logging(self):
        self.logger = utils.create_logger(
            log_file = self.training_config.log_file,
            print_to_console = False
        )


    def print(self, *args):
        print(f'{self.training_config.run_id}:', *args)
        if hasattr(self, 'logger'):
            self.logger.info(f'{self.training_config.run_id}: {" ".join([str(x) for x in args])}')
    

    def run(self):
        '''Run the training process'''
        try:
            self.train()
        except BaseException as e:
            print("========================ERROR========================")
            print(f"ERROR: Training failed or interrupted for {self.training_config.run_id}: {e}")
            self.save_adapter()
            self.graceful_shutdown()
            raise e


    def graceful_shutdown(self):
        '''Gracefully shutdown the training process'''

        # Clear the cache
        gc.collect()
        torch.cuda.empty_cache()

        # Clear the cache
        try:
            torch.cuda.ipc_collect()
        except:
            pass

        try:
            # Then let PyTorch tear down the process group, if vLLM initialized it
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()  # or dist.shutdown() on recent PyTorch
        except AssertionError:
            pass
        
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except OSError: 
            pass

        # Remove wandb
        wandb.finish()
        wandb.teardown()

        self.print("Successfully deleted the llm pipeline and free the GPU memory!")

