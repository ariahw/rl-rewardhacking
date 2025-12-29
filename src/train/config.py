from pydantic import BaseModel
from typing import Literal

from src import utils, RESULTS_PATH


class TrainingConfig(BaseModel):
    """Generic training config to use across SFT and RL"""

    # CORE SETTINGS
    # NOTE: Add to exclusion list in .sfttrainer_config() if you do not want to pass to SFTTrainer
    run_id: str
    model_id: str # Base model to train
    dataset_path: str # Finetuning dataset
    eval_dataset_path: str | None = None # Optional eval dataset
    save_merged: bool = False # Save merged version after finetuning
    extra_metadata: dict | None = None 
    skip_save: bool = False # Skip saving the model to disk
    resume_from_checkpoint: bool = False # Resume from checkpoint

    # TRAINING SETTINGS
    seed: int = 1
    logging_steps: int = 1
    report_to: str = "wandb"

    # Unsloth only settings
    eval_strategy: str = 'epoch' # Will eval every epoch; Unsloth Only
    save_strategy: str = 'epoch' # Will save every epoch; Unsloth Only
    save_only_model: bool = True # Save only model (LoRA adapter), not optimizer state. Works for both Unsloth and Verl

    # Save settings
    save_total_limit: int | None = None # Prevent excessive saving
    save_steps: int = 50

    # Quantization settings - Unsloth only
    load_in_4bit: bool = False
    load_in_8bit: bool = False

    # LoRA Arguments
    # PEFT arguments can be taken from: https://huggingface.co/docs/peft/v0.17.0/en/package_reference/lora#peft.LoraConfig
    lora_rank: int = 32
    lora_alpha: int = 32

    # Unsloth-Only Arguments
    lora_dropout: float = 0.0 # Unsloth only; Not supported for verl
    lora_bias: Literal["none"] = "none"  # Unsloth only; Supports any, but = "none" is optimized
    use_rslora: bool = False # Unsloth only; Not supported by verl
    loftq_config: Literal[None] = None # Unsloth only; Not supported by verl

    @property
    def output_dir(self):
        return f"{RESULTS_PATH}/runs/{self.model_id.split('/')[-1].lower()}/{self.run_id}"
    
    @property
    def config_path(self):
        return f"{self.output_dir}/config.json"

    @property
    def output_adapter_path(self):
        return f"{self.output_dir}/adapter"
    
    @property
    def log_file(self):
        return f"{self.output_dir}/training.log"
    
    @property
    def base_kwargs(self):
        return [
            'run_id',
            'model_id',
            'dataset_path',
            'eval_dataset_path',
            'save_merged',
            'extra_metadata',
            'skip_save',
            'resume_from_checkpoint',
        ]
    
    @property
    def lora_kwargs(self):
        return [
            "lora_rank",
            "lora_alpha",
            "lora_dropout",
            "lora_bias",
            "use_rslora",
            "loftq_config",
        ]
    
    def training_args(self):
        return {k: v for k, v in self.model_dump().items() if k not in (self.base_kwargs + self.lora_kwargs)}
    
    def lora_args(self):
        return {k: getattr(self, k) for k in self.lora_kwargs}

    def save(self):
        utils.verify_path(self.config_path)
        
        with open(self.config_path, 'w') as f:
            f.write(self.model_dump_json(indent = 4))

    def peft_config(self) -> dict:
        return {
            'r': self.lora_rank,
            'lora_alpha': self.lora_alpha,
            'lora_dropout': self.lora_dropout,
            'target_modules': [ # NOTE: This is hardcoded for Verl RL
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            'bias': self.lora_bias,
            'use_rslora': self.use_rslora,
            'loftq_config': self.loftq_config,
        }


class GRPOConfig(TrainingConfig):
    '''https://huggingface.co/docs/trl/main/en/grpo_trainer#trl.GRPOConfig'''

    eval_strategy: str = 'steps' # Overriding default for GRPO
    save_strategy: str = 'steps' # Overriding default for GRPO
    save_steps: int = 50 # Overriding default for GRPO

    system_prompt: str | None = None # Added to all GRPO generations
    system_prompt_method: Literal['after', 'before', 'replace'] = 'replace' # Replace the system prompt or append AFTER
    
    reward_funcs_kwargs: dict[str, dict] = {} # Names of reward functions and their kwargs
    screening_funcs_kwargs: dict[str, dict] = {} # Arguments to pass to screening functions; NOT IMPLEMENTED IN VERL

    beta: float = 1e-3 # KL coefficient

    optim: Literal["adamw_8bit", "adamw"] = "adamw_8bit"
    learning_rate: float = 7e-5
    lr_scheduler_type: Literal["linear", "cosine", "constant"] = "cosine" # Note: Linear not supported by verl
    warmup_ratio: float | None = None
    warmup_steps: int = 10 # Will overwrite any setting for warmup ratio
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    max_grad_norm: float = 1.0

    num_train_epochs: int = 1 # NOTE: Implemented such that this will be overwritte by max_steps
    max_steps: int = 500 # This usually makes more sense for RL

    max_prompt_length: int | None = 1536 # Causes some issue
    max_completion_length: int = 1536
    dataloader_num_workers: int = 4 # Start with 4, increase to 8 if CPU allows

    # Batch Size
    # Per gradient update, run num_prompt x num_generations per prompt

    # per_device_train_batch_size = number of prompts selected each round
    # num_generations = number of generations per selected prompt
    # mini batch = per_device_train_batch_size * num_generations (total number of generations per step)
    # gradient_accumulation_steps = number of steps to accumulate gradients before updating the model
    # Logging "steps" = gradient updates
    num_generations: int = 16 # Number of rollouts per prompt
    num_prompts: int = 16 # Number of prompts to run on each device
    per_device_batch_size: int = 8 # Number of prompts to run on each device; for Verl Only (for Unsloth == num_generations); if auto_find_batch_size is True this is transformed to tokens via per_device_batch_size * (max_prompt_length + max_completion_length)
    auto_find_batch_size: bool = True # Recommend set to True at first, then restart + set to False

    enable_gradient_checkpointing: bool = True # Enable gradient checkpointing for the model
    gpu_memory_utilization: float = 0.6 # Reduce for verl, can set to 0.9 for unsloth

    # GRPO Generation config
    use_vllm: bool = True # Set to false only when need to run activation caching
    temperature: float = 0.7
    top_p: float = 0.95
    repetition_penalty: float = 1.0 # Default to 1.0, no effect on repetition
    generation_kwargs: dict = {} # Other arguments to pass to vLLM
    enable_thinking: bool = False # Default False, only applied if the model is a reasoning model

    # Activations caching settings
    cache_activations: bool = False
    cache_activations_layers: list[int] | None = [18] # Need to be layers that are in the model
    cache_activations_position: str | None = "response_avg" # Needs to be prompt_avg, prompt_last, response_avg

    # Screening-related
    fill_nan_global: bool = True # Fill nan's with global mean and std; useful when batch size is small

    # TRL Only Parameters
    log_completions: bool = True # Always true in verl
    dataloader_prefetch_factor: int = 2 # Prefetch batches to reduce GPU idle time
    dataloader_persistent_workers: bool = True # Keep workers alive between epochs
    dataloader_pin_memory: bool = True # Already default, but explicit for clarity

    @property
    def base_kwargs(self):
        return super().base_kwargs + [
            'system_prompt',
            'system_prompt_method',
            'reward_funcs_kwargs',
            'screening_funcs_kwargs',
        ]