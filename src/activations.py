from abc import ABC, abstractmethod
from typing import Literal
import gc
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.generate import ChatRequest

"""
Activations Caching
"""

CachePosition = Literal["prompt_avg", "prompt_last", "response_avg", "prompt_all", "response_all"]

class ActivationsCache(ABC):
    name: str


    @abstractmethod
    def cache_activations(self, prompts: list[ChatRequest], responses: list[str], layers: list[int] | None = None):
        pass

    def cleanup(self):
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer

        try:
            torch.cuda.empty_cache()
            gc.collect()
        except:
            pass

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


class TransformersActivations(ActivationsCache):
    name = "transformers"
    def __init__(self, model_name: str | None = None, model = None, tokenizer = None, progress_bar: bool = True, debug: bool = False):
        '''Provide either model name or model and tokenizer'''

        assert (model is not None) or (model_name is not None)
        
        if model is None:
            self.model_name = model_name
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="flash_attention_2",
            )
        else:
            self.model_name = model.config._name_or_path
            self.model = model

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            self.tokenizer = tokenizer
        
        self.progress_bar = progress_bar
        self.debug = debug

    
    def log(self, msg: str):
        if self.debug:
            print(msg)

    def cache_activations(self, prompts: list[ChatRequest], responses: list[str], layers: list[int] | None = None, position: list[CachePosition] | None = None):

        self.log(f"Caching activations for {len(prompts)} prompts and {len(responses)} responses")
        self.log(f"Layers: {layers}")
        self.log(f"Position: {position}")
        self.log(f"Prompt: {prompts[0]}")
        self.log(f"Response: {responses[0]}")

        max_layer = self.model.config.num_hidden_layers
        if layers is None:
            layers = list(range(max_layer+1))
        
        if position is None:
            position = ["prompt_avg", "prompt_last", "response_avg"]
        position = set(position)

        cache = {k: [[] for _ in range(max_layer+1)] for k in position}

        if self.progress_bar:
            pbar = tqdm(total=len(prompts), desc="Caching activations")
        
        with torch.inference_mode():
            for prompt, response in zip(prompts, responses):
                # Convert all to chatml format
                full_text = prompt + [{'role': 'assistant', 'content': response}]
                full_chat_text = self.tokenizer.apply_chat_template(full_text, tokenize=False, add_generation_prompt=False)
                prompt_chat_text = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

                # Tokenize
                inputs = self.tokenizer(full_chat_text, return_tensors="pt", add_special_tokens=False).to(self.model.device)
                prompt_len = len(self.tokenizer.encode(prompt_chat_text, add_special_tokens=False))

                # NOTE: BATCH SIZE IS 1

                # Cache activations
                outputs = self.model(**inputs, output_hidden_states=True)
                for layer in layers:
                    if "prompt_avg" in position:
                        cache['prompt_avg'][layer].append(outputs.hidden_states[layer][:, :prompt_len, :].mean(dim=1).detach().cpu()) # (1, hidden_size)
                    if "response_avg" in position:
                        cache['response_avg'][layer].append(outputs.hidden_states[layer][:, prompt_len:, :].mean(dim=1).detach().cpu()) # (1, hidden_size)
                    if "prompt_last" in position:
                        cache['prompt_last'][layer].append(outputs.hidden_states[layer][:, prompt_len-1, :].detach().cpu()) # (1, hidden_size)
                    if "prompt_all" in position:
                        cache['prompt_all'][layer].append(outputs.hidden_states[layer][:, :prompt_len, :].detach().cpu()) # (1, prompt_len, hidden_size)
                    if "response_all" in position:
                        cache['response_all'][layer].append(outputs.hidden_states[layer][:, prompt_len:, :].detach().cpu()) # (1, response_len, hidden_size)
                del outputs

                if self.progress_bar:
                    pbar.update(1)
        
        if self.progress_bar:
            pbar.close()

        # Response all and prompt_all need to be padded to the same length
        # prompt_all and response_all will have seq_len as part of the shape
        for k in ["prompt_all", "response_all"]:
            if k in cache:
                x = cache[k]

                # Get max len across all layers
                max_len = max([max([x[l][i].shape[1] for i in range(len(x[l]))]) for l in layers])

                # Pad all sequences to have dim[1] = max_len
                for l in layers:
                    x[l] = [torch.nn.functional.pad(x[l][i].transpose(-1, -2), (0, max_len - x[l][i].shape[1]), value=torch.nan).transpose(-1, -2) for i in range(len(x[l]))]

        for k in position:
            layer = None
            try:
                # Cat all of the layers -> (n_samples, hidden_size) for each layer OR (n_samples, seq_len, hidden_size)
                for layer in layers:
                    cache[k][layer] = torch.cat(cache[k][layer], dim=0) # (n_samples, hidden_size) OR (n_samples, seq_len, hidden_size)
            except Exception as e:
                self.log(f"Error caching activations for layer {layer}: {e}")
                raise e
                
            # Stack all of the layers -> (n_layers, n_samples, hidden_size)
            cache[k] = torch.vstack([cache[k][l].unsqueeze(0) for l in layers]) # (n_layers, n_samples, hidden_size) OR (n_layers, n_sample, seq_len, hidden_size)

        # Returns: dict[str, torch.Tensor]
        # 'prompt_avg': (n_layers, n_samples, hidden_size)
        # 'prompt_last': (n_layers, n_samples, hidden_size)
        # 'response_avg': (n_layers, n_samples, hidden_size)
        # 'prompt_all': (n_layers, n_samples, seq_len, hidden_size) # seq_len is the maximum sequence length across all samples and layers
        # 'response_all': (n_layers, n_samples, seq_len, hidden_size) # seq_len is the maximum sequence length across all samples and layers
        return cache 


class BatchedTransformersActivations(TransformersActivations):
    name = "transformers_batched"

    def __init__(
        self,
        model_name: str | None = None,
        model=None,
        tokenizer=None,
        progress_bar: bool = True,
        debug: bool = False,
        batch_size: int = 8,
    ):
        """Same as TransformersActivations, but processes inputs in batches for speed.

        Args:
            model_name: HF repo ID if model/tokenizer not provided.
            model: Optional preloaded model (e.g., with LoRA applied).
            tokenizer: Optional preloaded tokenizer.
            progress_bar: Show a tqdm over batches.
            debug: Verbose logging.
            batch_size: Number of (prompt,response) pairs per forward pass.
        """
        super().__init__(model_name=model_name, model=model, tokenizer=tokenizer, progress_bar=progress_bar, debug=debug)
        self.batch_size = max(1, int(batch_size))
        try:
            self.model.eval()
        except Exception:
            pass

    def cache_activations(
        self,
        prompts: list[ChatRequest],
        responses: list[str],
        layers: list[int] | None = None,
        position: list[CachePosition] | None = None,
    ):
        """Batched variant of TransformersActivations.cache_activations.

        Returns the same structure and shapes as the base implementation, but
        executes the model on batches to reduce overhead.
        """

        self.log(f"Caching activations (batched) for {len(prompts)} items, batch_size={self.batch_size}")
        if len(prompts) == 0:
            return {}

        max_layer = self.model.config.num_hidden_layers
        if layers is None:
            layers = list(range(max_layer + 1))

        if position is None:
            position = ["prompt_avg", "prompt_last", "response_avg"]
        position = set(position)

        # Prepare storage: dict[pos][layer] -> list of tensors (later cat along samples)
        cache: dict[str, list[list[torch.Tensor]]] = {k: [[] for _ in range(max_layer + 1)] for k in position}

        # Pre-render chat templates and prompt lengths so we can batch tokenize
        full_texts: list[str] = []
        prompt_lens: list[int] = []
        for prompt, response in zip(prompts, responses):
            full_text = prompt + [{"role": "assistant", "content": response}]
            full_chat_text = self.tokenizer.apply_chat_template(
                full_text, tokenize=False, add_generation_prompt=False
            )
            prompt_chat_text = self.tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
            full_texts.append(full_chat_text)
            # Compute prompt token length per sample
            prompt_len = len(self.tokenizer.encode(prompt_chat_text, add_special_tokens=False))
            prompt_lens.append(prompt_len)

        total = len(full_texts)
        rng = range(0, total, self.batch_size)
        pbar = tqdm(total=len(rng), desc="Caching activations (batched)") if self.progress_bar else None

        with torch.inference_mode():
            for start in rng:
                end = min(start + self.batch_size, total)
                batch_texts = full_texts[start:end]
                batch_prompt_lens = prompt_lens[start:end]

                # Tokenize as a batch; use padding to longest to enable a single forward pass
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    add_special_tokens=False,
                    padding=True,
                    truncation=False,
                )
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

                outputs = self.model(**inputs, output_hidden_states=True)
                hs = outputs.hidden_states  # tuple(len = n_layers+1) of (B, L, H)

                # Per-sample extraction within the batch
                attn_mask = inputs.get("attention_mask")
                for j in range(end - start):
                    prompt_len = int(batch_prompt_lens[j])
                    # Actual (unpadded) sequence length for this item
                    if attn_mask is not None:
                        seq_len = int(attn_mask[j].sum().item())
                    else:
                        # Fallback: assume no padding
                        seq_len = int(hs[0].shape[1])

                    for layer in layers:
                        h = hs[layer][j]  # (L, H)
                        if "prompt_avg" in position:
                            cache["prompt_avg"][layer].append(
                                h[:prompt_len, :].mean(dim=0, keepdim=True).detach().cpu()
                            )
                        if "prompt_last" in position:
                            last_idx = max(0, prompt_len - 1)
                            cache["prompt_last"][layer].append(
                                h[last_idx, :].unsqueeze(0).detach().cpu()
                            )
                        if "response_avg" in position:
                            cache["response_avg"][layer].append(
                                h[prompt_len:seq_len, :].mean(dim=0, keepdim=True).detach().cpu()
                            )
                        if "prompt_all" in position:
                            cache["prompt_all"][layer].append(
                                h[:prompt_len, :].unsqueeze(0).detach().cpu()
                            )
                        if "response_all" in position:
                            cache["response_all"][layer].append(
                                h[prompt_len:seq_len, :].unsqueeze(0).detach().cpu()
                            )

                del outputs
                if pbar is not None:
                    pbar.update(1)

        if pbar is not None:
            pbar.close()

        # Pad variable-length sequences for *_all keys to a common length across samples per layer
        for k in ["prompt_all", "response_all"]:
            if k in cache:
                x = cache[k]
                if len(layers) == 0:
                    continue
                # Max length over all samples and selected layers
                try:
                    max_len = max(
                        [max([x[l][i].shape[1] for i in range(len(x[l]))]) for l in layers]
                    )
                except ValueError:
                    max_len = 0
                for l in layers:
                    x[l] = [
                        torch.nn.functional.pad(
                            x[l][i].transpose(-1, -2),
                            (0, max_len - x[l][i].shape[1]),
                            value=torch.nan,
                        ).transpose(-1, -2)
                        for i in range(len(x[l]))
                    ]

        # Concatenate per-layer lists across samples, then stack layers
        for k in position:
            layer = None
            try:
                for layer in layers:
                    cache[k][layer] = torch.cat(cache[k][layer], dim=0)
            except Exception as e:
                self.log(f"Error caching activations for layer {layer}: {e}")
                raise e
            cache[k] = torch.vstack([cache[k][l].unsqueeze(0) for l in layers])

        return cache
