from __future__ import annotations

import os
from typing import Any, Dict, Optional

import ray
import torch

from src.activations import TransformersActivations
from src import ChatRequest
from src.train.verl.utils import convert_responses_to_str

from collections import OrderedDict
from typing import Any, Optional

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from peft.utils.save_and_load import set_peft_model_state_dict

from verl.utils.fsdp_utils import replace_lora_wrapper, collect_lora_params
from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker
from verl.single_controller.base.decorator import register, Dispatch
from verl import DataProto
from verl.single_controller.base import Worker

from src.utils import get_logger
from src.activations import BatchedTransformersActivations

logger = get_logger()


@ray.remote(num_cpus=1, num_gpus=1, name="activations_cache_worker")
class ActivationsWorker:
    """Ray actor that holds a HF model on a dedicated GPU and serves activation caching.

    Notes on GPU placement:
    - Ray assigns a distinct physical GPU to this actor via `num_gpus=1`.
    - Ensure the training config reserves one GPU for this worker (n_gpus_per_node reduced by 1)
      so rollout workers do not consume all GPUs.
    """

    def __init__(
        self,
        model_id: str,
        lora_config: dict,
        layers: list[int] = [],
        position: str = "response_avg",
        batch_size: int = 16,
        dtype: str | torch.dtype = torch.bfloat16,
        trust_remote_code: bool = False,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:

        # super().__init__()

        self.model_id = model_id
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.trust_remote_code = trust_remote_code
        self.tokenizer_kwargs = tokenizer_kwargs or {}
        self.chat_template_kwargs = chat_template_kwargs or {}

        self.layers = layers
        self.position = position # Currently only support one position at a time

        assert len(self.layers) > 0, "Layers must be a non-empty list"

        # Honor Ray's CUDA_VISIBLE_DEVICES: use device index 0 within the assigned namespace.
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # Load tokenizer and model on the assigned GPU.
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=self.trust_remote_code, **self.tokenizer_kwargs
        )

        model_cls = AutoModelForCausalLM
        model = model_cls.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            trust_remote_code=self.trust_remote_code,
            device_map={"": 0} if self.device.type == "cuda" else None,
        )

        self.lora_config = LoraConfig(**lora_config)
        peft_model = get_peft_model(model, self.lora_config)
        self.peft_config = peft_model.peft_config.get("default", None)  # type: ignore[assignment]

        self.activations_extractor = BatchedTransformersActivations(
            model=peft_model, 
            tokenizer=self.tokenizer, 
            progress_bar=True,
            batch_size=batch_size
        )

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self._last_lora_hash: Optional[str] = None

        logger.info(f"INITIALIZED ACTIVATIONS WORKER: layers {self.layers}, position {self.position}")
    
    @property
    def model(self):
        return self.activations_extractor.model


    def warmup(self) -> str:
        """Run a tiny forward to allocate memory and lazy init kernels."""
        try:
            text = "Hello"
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                _ = self.model(**inputs)
            return "ok"
        except Exception as e:
            return f"error: {e}"


    def update_lora(
        self,
        lora_params: Dict[str, Any],
        base_sync_done: bool = True,
        **kwargs
    ) -> None:

        if not lora_params:
            return

        new_hash = self.hash_state_dict(lora_params)
        if self._last_lora_hash == new_hash:
            return

        state_dict = lora_params
        if not base_sync_done:
            state_dict = {replace_lora_wrapper(k, self.peft_config): v for k, v in lora_params.items()}

        set_peft_model_state_dict(self.model, state_dict)

        self._last_lora_hash = new_hash
        return
    

    def hash_state_dict(self, d: dict[str, Any] | OrderedDict) -> str:
        """Create a stable hash for a state dict to detect changes quickly."""
        import hashlib

        hasher = hashlib.sha256()
        for k in sorted(d.keys()):
            hasher.update(k.encode("utf-8"))
            t = d[k]
            try:
                # Prefer tensor buffer if available
                buf = t.cpu().contiguous().view(-1).numpy().tobytes()  # type: ignore[attr-defined]
            except Exception:
                # Fallback to repr
                buf = repr(t).encode("utf-8")
            hasher.update(buf)
        return hasher.hexdigest()


    def cache_activations(self, data: DataProto) -> Dict[str, Any]:
        logger.info(f"Attempting to cache activations at {self.position} with layers {self.layers}")
        responses_str = convert_responses_to_str(data, self.tokenizer)
        prompts = [x['prompt'] for x in data.non_tensor_batch["extra_info"]]

        cache = self.activations_extractor.cache_activations(
            prompts=prompts, 
            responses=responses_str, 
            layers=self.layers, 
            position=[self.position]
        )[self.position].squeeze(0) # Remove the first dimesion so it is n_samples x hidden_size
        logger.info(f"Activations cached at {self.position} with layers {self.layers}, shape {cache.shape}")
        return {"activations": cache}

    def update_and_cache(
        self,
        lora_params: Dict[str, Any],
        data: DataProto,
        base_sync_done: bool = True,
        **kwargs
    ) -> Dict[str, Any]:

        self.update_lora(
            lora_params, 
            base_sync_done=base_sync_done, 
        )
        return self.cache_activations(
            data=data
        )


    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "dtype": str(self.dtype),
            "device": str(self.device),
            "has_peft": hasattr(self.model, "peft_config"),
        }

    def shutdown(self) -> str:
        try:
            import gc

            del self.model
            torch.cuda.empty_cache()
            gc.collect()
            return "shutdown"
        except Exception as e:
            return f"error: {e}"


class ExtendedActorRolloutRefWorker(ActorRolloutRefWorker):
    """Extended worker with method to extract LoRA parameters."""

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_lora_params(self, layered_summon: bool = False) -> Dict[str, Any]:
        """
        Extract LoRA parameters from actor_module_fsdp.

        This method allows external callers to retrieve the current LoRA parameters
        from the worker's actor module.

        Args:
            layered_summon: Whether to use layered summoning for FSDP2

        Returns:
            OrderedDict of LoRA parameters
        """
        return collect_lora_params(
            module=self.actor_module_fsdp,
            layered_summon=layered_summon,
            base_sync_done=self.base_sync_done,
        )

class ExtendedAsyncActorRolloutRefWorker(AsyncActorRolloutRefWorker):
    """Extended worker with method to extract LoRA parameters."""

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_lora_params(self, layered_summon: bool = False) -> Dict[str, Any]:
        """
        Extract LoRA parameters from actor_module_fsdp.

        This method allows external callers to retrieve the current LoRA parameters
        from the worker's actor module.

        Args:
            layered_summon: Whether to use layered summoning for FSDP2

        Returns:
            OrderedDict of LoRA parameters
        """
        return collect_lora_params(
            module=self.actor_module_fsdp,
            layered_summon=layered_summon,
            base_sync_done=self.base_sync_done,
        )