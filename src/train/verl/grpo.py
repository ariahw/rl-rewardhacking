import os
import torch
import ray
import warnings

from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from datasets import Dataset

from verl.trainer.main_ppo import run_ppo

from src.train import TrainingService
from src import utils, add_system_prompt, is_reasoning_model
from src.train.verl.trainer import RHGRPOTaskRunner

'''
VERL GRPO TRAINING CLASS
'''


class VerlGRPO(TrainingService):
    name: str = "verl_grpo"

    def train_dataset_path(self):
        return f"{self.training_config.output_dir}/train_dataset.parquet"
    
    def validation_dataset_path(self):
        return f"{self.training_config.output_dir}/validation_dataset.parquet"
    
    def verl_config_path(self):
        return f"{self.training_config.output_dir}/verl_config.yaml"
    
    def verl_full_config_path(self):
        return f"{self.training_config.output_dir}/verl_full_config.yaml"

    def load_configure_datasets(self):
        
        # Load the dataset
        if not os.path.exists(self.training_config.dataset_path):
            raise ValueError(f"Dataset not found at {self.training_config.dataset_path}")

        # Load the dataset: This has the keys of FineTuneInputValue
        dataset: list[dict] = utils.read_jsonl_all(self.training_config.dataset_path)

        # Add system prompt if provided
        if self.training_config.system_prompt is not None:
            for data in dataset:
                data['prompt'] = add_system_prompt(data['prompt'], self.training_config.system_prompt, method = self.training_config.system_prompt_method)
            self.print("System prompt added to dataset")
            self.print(f"Example of dataset: {str(dataset[0]['prompt'])[:100]}...")

        # Convert to Dataset object
        dataset = Dataset.from_list(dataset)

        def map_to_verl_format(x: dict):
            '''Map a datasets example to a verl example'''

            return {
                'data_source': x['dataset'],
                'prompt': x['prompt'],
                'ability': "code",
                'reward_model': {'style': 'rule', 'ground_truth': x['answer']},
                'extra_info': x # Note: This is required to pass the answers and the problem IDs for better tracking, but increases data size
            }
    
        dataset = dataset.map(map_to_verl_format)
        dataset = dataset.select_columns(['data_source', 'prompt', 'ability', 'reward_model', 'extra_info'])

        # Save to parquet file at the training dataset location
        dataset.to_parquet(self.train_dataset_path())
        self.print(f"Saved dataset to {self.train_dataset_path()}")

        dataset.select(range(10)).to_parquet(self.validation_dataset_path())
        self.print(f"Saved dataset to {self.validation_dataset_path()}")
    

    def read_in_config(self, config_yaml: str):
        """
        Build the config the same way @hydra.main would, then optionally
        overlay/override with a user-provided YAML file.
        """
        config_path = os.path.abspath(os.getcwd())
        config_path = os.path.join(config_path, "src", "train", "verl", "config")
        self.print("Main verl config path: ", config_path)

        # This mirrors: @hydra.main(config_path="config", config_name="ppo_trainer") in verl/trainer/main_ppo.py
        with initialize_config_dir(config_dir=config_path, version_base=None, job_name="ppo_trainer"):
            # This gives you the same default-config as before
            cfg = compose(config_name="rh_trainer")
        
        # Prevent struct issue with new keys
        OmegaConf.set_struct(cfg.data.apply_chat_template_kwargs, False)
        OmegaConf.set_struct(cfg.actor_rollout_ref.rollout.engine_kwargs.vllm, False)
        OmegaConf.set_struct(cfg.reward_model, False)
        OmegaConf.set_struct(cfg.algorithm.screening_specs, False)

        # Merge YAML
        # Later arguments override earlier ones
        user_cfg = OmegaConf.load(config_yaml)
        cfg = OmegaConf.merge(cfg, user_cfg)

        return cfg


    def create_config(self):
        # Use jinja to create a yaml file corresponding to the very config

        assert not self.training_config.model_id.startswith("unsloth"), "Verl does not support unsloth models!"
        assert self.training_config.lr_scheduler_type in ["cosine", "constant"], "Linear scheduler not supported by verl"

        assert not self.training_config.use_rslora, "Verl does not support RSLoRA!"
        assert self.training_config.loftq_config is None, "Verl does not support LoFTQ!"
        assert self.training_config.lora_bias == "none", "Verl does not support LoRA bias!"
        assert self.training_config.lora_dropout == 0.0, "Verl does not support LoRA dropout!"

        # Check for bitsandbytes + FSDP2 incompatibility
        # verl uses fsdp2 by default, and bitsandbytes optimizer doesn't work with FSDP2/DTensor
        if self.training_config.optim == "adamw_8bit":
            self.print("WARNING: bitsandbytes optimizer (adamw_8bit) is incompatible with FSDP2/DTensor used by verl.")
            self.print("Automatically switching to adamw optimizer. This may use more memory but will work correctly.")
            # Override the optimizer setting to use regular AdamW instead
            self.training_config.optim = "adamw"

        # If activation caching is enabled, reserve 1 GPU for the activations worker.
        n_total = torch.cuda.device_count()
        reserve_for_activations = 1 if getattr(self.training_config, "cache_activations", False) else 0 # NOTE: If n_gpus is an odd number, a lot more difficult
        n_gpus = n_total - reserve_for_activations
        assert n_gpus >= 1, "No GPUs available for training!"

        if n_gpus % 2 != 0:
            warnings.warn("Odd number of GPUs detected, will remove 1 to preserve even number of GPUs for training")
            n_gpus -= 1
        
        print(f"n_gpus: {n_gpus} | activations_gpus: {reserve_for_activations}")

        full_batch_size = self.training_config.num_prompts * self.training_config.num_generations
        self.print(f"Optimizer update batch size: {full_batch_size}")

        # Calculate checkpoint save contents based on save_only_model
        # Format as YAML list string for the template
        if self.training_config.save_only_model:
            checkpoint_save_contents = "['model']"
        else:
            checkpoint_save_contents = "['model', 'optimizer', 'extra']"

        # Calculate total_epochs: if max_steps is specified, use a very large number
        # to ensure training continues until total_training_steps is reached, not stopping after 1 epoch
        if self.training_config.max_steps and self.training_config.max_steps > 0:
            total_epochs = 9999 # Set to very large number to ensure training continues until total_training_steps is reached, not stopping after 1 epoch
        else:
            total_epochs = self.training_config.num_train_epochs

        if self.training_config.auto_find_batch_size:
            ppo_max_token_len_per_gpu = self.training_config.per_device_batch_size * (self.training_config.max_prompt_length + self.training_config.max_completion_length)
        else:
            ppo_max_token_len_per_gpu = 32678

        utils.create_yaml(
            template_path = "src/train/verl/grpo_config.jinja2",
            template_kwargs = {
                **self.training_config.training_args(),
                **self.training_config.lora_args(),
                **{
                    'model_id': self.training_config.model_id,
                    'run_name': self.training_config.run_id,
                    'chat_template_kwargs': {
                        'enable_thinking': self.training_config.enable_thinking
                    } if is_reasoning_model(self.training_config.model_id) else {}, #WARNING: Only compatible with Qwen models
                    'wandb_project': os.getenv('WANDB_PROJECT'),
                    'n_gpus': n_gpus,
                    'output_dir': self.training_config.output_dir,
                    'reward_funcs_kwargs': self.training_config.reward_funcs_kwargs,
                    'screening_funcs_kwargs': self.training_config.screening_funcs_kwargs,
                    'train_batch_size': self.training_config.num_prompts,
                    'mini_batch_size': self.training_config.num_prompts, # batch size per optimizer step
                    'train_dataset_parquet_path': self.train_dataset_path(),
                    'validation_dataset_parquet_path': self.validation_dataset_path(),
                    'reward_func_path': "src/train/verl/rewards.py",
                    'reward_func_name': "master_reward",
                    'checkpoint_save_contents': checkpoint_save_contents,
                    'total_epochs': total_epochs,
                    'ppo_max_token_len_per_gpu': ppo_max_token_len_per_gpu, # Default value
                    'rollout_engine': "vllm" if self.training_config.use_vllm else "hf",
                    'use_dynamic_bsz': False,
                    'max_num_seqs': 1024 if not self.training_config.cache_activations else 512,
                    'max_num_batched_tokens': 16384 if not self.training_config.cache_activations else 8192,
                },
            },
            output_path = self.verl_config_path(),
        )
        self.print(f"Created config at {self.verl_config_path()}")

        config = self.read_in_config(self.verl_config_path())
        utils.save_yaml(self.verl_full_config_path(), config)

        return config
        

    def train(self):
        '''Run training and return name of model'''

        # Copies the dataset to output directory path ending in .parquet
        self.load_configure_datasets() 
        
        # Run config creation
        config = self.create_config()

        # Run training
        run_ppo(config, task_runner_class=RHGRPOTaskRunner)

        # Shut down
        self.graceful_shutdown()
    

    def save_adapter(self):
        # Verl does not support this
        pass
    


    def graceful_shutdown(self):

        ray.shutdown()

        super().graceful_shutdown()
