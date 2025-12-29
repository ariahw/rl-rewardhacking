import os
import asyncio
import threading
import gc
import time
import requests
from concurrent.futures import Future
from typing import Literal
from abc import ABC, abstractmethod
from urllib.parse import urlparse
from tqdm import tqdm

import torch

from src import ChatRequest, SamplingParams, is_reasoning_model

torch.set_float32_matmul_precision('high')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EngineOptions = Literal["vllm", "openrouter"]

GENERATOR_REGISTRY: dict[str, type["LLMGenerator"]] = {}

def register_generator(cls: type["LLMGenerator"]) -> type["LLMGenerator"]:
    GENERATOR_REGISTRY[cls.name] = cls
    return cls

_bg_loop: asyncio.AbstractEventLoop | None = None
_bg_thread: threading.Thread | None = None


def _ensure_background_loop() -> asyncio.AbstractEventLoop:
    global _bg_loop, _bg_thread
    if _bg_loop is not None and _bg_loop.is_running():
        return _bg_loop

    def _loop_runner(loop: asyncio.AbstractEventLoop):
        asyncio.set_event_loop(loop)
        loop.run_forever()

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=_loop_runner, args=(loop,), daemon=True)
    thread.start()
    _bg_loop = loop
    _bg_thread = thread
    return loop


def run_coro_sync(coro):
    """Run an async coroutine from sync code safely using a persistent background loop."""
    loop = _ensure_background_loop()
    fut: Future = asyncio.run_coroutine_threadsafe(coro, loop)
    return fut.result()


class LLMGenerator(ABC):
    name: str

    @abstractmethod
    def batch_generate(self, prompts: list[ChatRequest], sampling_params: SamplingParams, **kwargs) -> list[str] | list[list[str]]:
        pass

    def turn_on_thinking(self):
        pass
    
    def turn_off_thinking(self):
        pass

    def respond(self, prompt: str, sampling_params: SamplingParams | None = None, **kwargs) -> str:
        if sampling_params is None:
            sampling_params = SamplingParams()
            sampling_params.n = 1 # Force = 1 for this sampling
        resp = self.batch_generate([[{'role': 'user', 'content': prompt}]], sampling_params, **kwargs)
        return resp[0]

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


@register_generator
class VLLMGenerator(LLMGenerator):
    name = "vllm"

    def __init__(self, model_name: str, lora_adapter_path: str | None = None, max_lora_rank: int = 64, **kwargs):

        self.model_name = model_name
    
        if lora_adapter_path is not None:
            from vllm.lora.request import LoRARequest
            self.lora_adapter_path = self.resolve_lora_adapter_path(lora_adapter_path)
            self.lora_request = LoRARequest("adapter", 1, self.lora_adapter_path)
            print("Loaded LoRA adapter:", self.lora_adapter_path)
        else:
            self.lora_request = None


        from vllm import LLM
        self.model = LLM(
            model=model_name,
            task="generate",
            max_lora_rank=max_lora_rank if lora_adapter_path is not None else None,
            enable_lora=lora_adapter_path is not None,
            **kwargs
        )
        print("Loaded VLLM model:", self.model_name)
        self.tokenizer = self.model.get_tokenizer()


        self.chat_template_kwargs = {}
    

    def turn_on_thinking(self):
        if is_reasoning_model(self.model_name):
            self.chat_template_kwargs['enable_thinking'] = True
            print(f"Turned on thinking for {self.model_name}: {self.chat_template_kwargs}")
    
    def turn_off_thinking(self):
        if is_reasoning_model(self.model_name):
            self.chat_template_kwargs['enable_thinking'] = False
            print(f"Turned off thinking for {self.model_name}: {self.chat_template_kwargs}")
    

    def resolve_lora_adapter_path(self, lora_adapter_path: str) -> str:
        if not os.path.exists(lora_adapter_path):
            raise FileNotFoundError(f"LoRA adapter path {lora_adapter_path} does not exist")
        
        # Check if adapter tensors exists
        if os.path.exists(os.path.join(lora_adapter_path, "adapter_config.json")): # Unsloth/TRL Location
            return lora_adapter_path
        elif os.path.exists(os.path.join(lora_adapter_path, "actor/lora_adapter/adapter_config.json")): # Verl Location
            return os.path.join(lora_adapter_path, "actor/lora_adapter")
        else:
            raise FileNotFoundError(f"LoRA adapter path {lora_adapter_path} does not contain adapter tensors")

    def batch_generate(self, prompts: list[ChatRequest], sampling_params: SamplingParams | None = None, **kwargs) -> list[str] | list[list[str]]:
        if sampling_params is None:
            sampling_params = SamplingParams()

        from vllm import SamplingParams as VLLMSamplingParams
        vllm_sampling_params = VLLMSamplingParams(**{
            'n': int(sampling_params.n),
            'temperature': sampling_params.temperature,
            'max_tokens': sampling_params.max_new_tokens,
            'top_p': sampling_params.top_p,
            'repetition_penalty': sampling_params.repetition_penalty
        })

        # Run batch inference
        responses = self.model.chat(
            messages = prompts,
            sampling_params = vllm_sampling_params,
            use_tqdm = True,
            lora_request = self.lora_request,
            chat_template_kwargs = self.chat_template_kwargs,
            **kwargs
        )

        if (sampling_params.n or 1) <= 1:
            return [y.outputs[0].text for y in responses]
        else:
            return [[out.text for out in y.outputs] for y in responses]


class AsyncLLMClientBackend(LLMGenerator):
    """Shared async client backend with bounded concurrency and optional RPM limit."""
    supports_native_n: bool = False

    def __init__(self, concurrent_requests: int = 10, max_rpm: int | None = None):
        self.semaphore = asyncio.Semaphore(max(1, concurrent_requests))
        self.max_rpm = max_rpm
        self.concurrent_requests = concurrent_requests
        self._request_times: list[float] = []
        self._rate_lock = asyncio.Lock()
        self.turn_off_thinking() # Turn off by default

    async def _rate_limit(self):
        if not self.max_rpm:
            return
        now = time.time()
        cutoff = now - 60
        async with self._rate_lock:
            self._request_times = [t for t in self._request_times if t > cutoff]
            if len(self._request_times) >= self.max_rpm:
                wait = max(0, 60 - (now - self._request_times[0]))
                if wait > 0:
                    await asyncio.sleep(wait)
            self._request_times.append(time.time())
    
    @abstractmethod
    async def _acomplete(self, prompt: ChatRequest, **kwargs) -> str:
        '''Complete a single prompt'''
        pass

    async def _run_single(self, prompt: ChatRequest, sampling_kwargs: dict):
        await self._rate_limit()
        async with self.semaphore:
            return await self._acomplete(prompt, **sampling_kwargs)

    async def run_batch_generate(self, prompts: list[ChatRequest], sampling_kwargs: dict) -> list[str] | list[list[str]]:
        n = int(sampling_kwargs.get("n", 1) or 1)
        if not self.supports_native_n:
            if (n > 1):
                prompts = [prompt for prompt in prompts for _ in range(n)]
            del sampling_kwargs['n']

        tasks = [asyncio.create_task(self._run_single(prompt, sampling_kwargs)) for prompt in prompts]
        pbar = tqdm(total=len(tasks), desc="Generating", leave=False)
        for task in tasks:
            task.add_done_callback(lambda *_: pbar.update(1))
        
        try:
            outputs = await asyncio.gather(*tasks)
        finally:
            pbar.close()

        if (not self.supports_native_n) or (n == 1):
            # _acomplete returns list[str] even for n=1, extract first element
            outputs = [out[0] if isinstance(out, list) and out else out for out in outputs]
        

        if (n > 1) and not self.supports_native_n:
            # Every output is a list, so it needs to be unpacked
            return [outputs[i * n:(i + 1) * n] for i in range(len(outputs) // n)]
        else:
            return outputs

    @abstractmethod
    def remaining_credits(self):
        '''Return the remaining credits for the engine'''
        pass

    def cleanup(self):
        if hasattr(self, "client"):
            async def _cleanup():
                close_fn = getattr(self.client, "close", None)
                if close_fn:
                    await close_fn()
                else:
                    subclient = getattr(self.client, "_client", None)
                    if subclient:
                        await subclient.aclose()
            try:
                run_coro_sync(_cleanup())
            finally:
                del self.client
                gc.collect()


class OpenAIClientBackend(AsyncLLMClientBackend):
    """Backend using OpenAI Client - base class for OpenAI-compatible APIs."""
    name = "openai_client_backend"
    
    def __init__(self, api_base: str, api_key: str, model_name: str, api_headers: dict = {}, concurrent_requests: int = 20, max_rpm: int | None = None, timeout: int = 30, **kwargs):
        from openai import AsyncOpenAI
        
        self.model_name = model_name
        self.api_key = (api_key or "").strip().strip('"').strip("'")
        assert self.api_key is not None, f"API key not provided."
        
        self.api_base = (api_base or "").strip()
        if not (self.api_base.startswith("http://") or self.api_base.startswith("https://")):
            raise ValueError(f"Invalid api_base for OpenAIClientBackend: {self.api_base!r}")
        urlparse(self.api_base)  # sanity check
        
        self.extra_headers = {k: v for k, v in api_headers.items() if v is not None}

        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            timeout=timeout,
            default_headers=self.extra_headers,
        )
        
        super().__init__(concurrent_requests=concurrent_requests, max_rpm=max_rpm)

    def turn_on_thinking(self):
        raise NotImplementedError(f"Thinking is not supported for {self.name}")

    def turn_off_thinking(self):
        raise NotImplementedError(f"Thinking is not supported for {self.name}")
    
    async def _acomplete(self, messages: list[dict], temperature: float | None = None, top_p: float | None = None, max_tokens: int | None = None, n: int = 1, reasoning: dict = {}, max_retries: int = 3, **kwargs) -> list[str]:
        for attempt in range(max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model = self.model_name,
                    messages = messages,
                    temperature = temperature,
                    top_p = top_p,
                    max_tokens = max_tokens,
                    n = n,
                    extra_body = reasoning,
                    **kwargs
                )
                choices = response.choices
                if reasoning.get('enabled', False):
                    return [(c.message.content.strip() if c.message.content else "", c.message.__dict__.get("reasoning", "").strip()) for c in choices]
                return [c.message.content.strip() if c.message.content else "" for c in choices]
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** (attempt + 1))
                    continue
                print(f"Async request failed: {e}")
                return [None for _ in range(n)]
        return [None for _ in range(n)]
    
    def batch_generate(self, prompts: list[ChatRequest], sampling_params: SamplingParams | None = None) -> list[str] | list[list[str]]:
        """Synchronous batch generation wrapper."""
        if sampling_params is None:
            sampling_params = SamplingParams()
        
        if sampling_params.with_reasoning:
            self.turn_on_thinking()
        else:
            self.turn_off_thinking()
        
        sampling_kwargs = {
            "temperature": sampling_params.temperature or 0.7,
            "top_p": sampling_params.top_p or 0.95,
            "max_tokens": sampling_params.max_new_tokens or 512,
            "n": int(sampling_params.n) if sampling_params.n is not None else 1,
            'reasoning': self.reasoning_kwargs,
        }
        
        # Not supported
        if "gpt-5" in self.model_name:
            del sampling_kwargs['temperature']
            del sampling_kwargs['top_p']
        
        return run_coro_sync(self.run_batch_generate(prompts, sampling_kwargs))


@register_generator
class OpenRouterGenerator(OpenAIClientBackend):
    name = "openrouter"
    supports_native_n = False
    
    def __init__(self, model_name: str, concurrent_requests: int = 100, timeout: int = 60, **kwargs):
        # Remove openrouter/ prefix if present (OpenRouter API expects just the model ID)
        clean_model_name = model_name.removeprefix("openrouter/") if model_name.startswith("openrouter/") else model_name

        assert os.environ["OPENROUTER_API_KEY"] is not None, "OPENROUTER_API_KEY is not set"
        
        super().__init__(
            api_base = "https://openrouter.ai/api/v1",
            api_key = os.environ["OPENROUTER_API_KEY"],
            api_headers = {
                "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER"),
                "X-Title": os.getenv("OPENROUTER_X_TITLE", "aria-wong")
            },
            model_name = clean_model_name,
            concurrent_requests = concurrent_requests,
            timeout = timeout,
            **kwargs
        )
    
    def turn_on_thinking(self):
        # See: https://openrouter.ai/docs/guides/best-practices/reasoning-tokens#controlling-reasoning-tokens
        self.reasoning_kwargs = {
            'enabled': True,
            'exclude': False
        }

        if "grok" in self.model_name:
            self.reasoning_kwargs['effort'] = "medium"
        elif "openai" in self.model_name:
            self.reasoning_kwargs['effort'] = "medium"
        elif "anthropic" in self.model_name:
            self.reasoning_kwargs['max_tokens'] = 1024 # Set here
        elif "gemini" in self.model_name:
            pass # No further action required
        elif "qwen" in self.model_name:
            pass # No further action required
        else:
            raise ValueError(f"Thinking is not supported for {self.model_name}")
    
    def turn_off_thinking(self):
        self.reasoning_kwargs = {
            'enabled': False,
            'exclude': True
        }

    def batch_generate(self, prompts: list[ChatRequest], sampling_params: SamplingParams | None = None) -> list[str] | list[list[str]]:
        if sampling_params is None:
            sampling_params = SamplingParams()

        if sampling_params.with_reasoning:
            self.turn_on_thinking()
        else:
            self.turn_off_thinking()

        n = int(sampling_params.n) if sampling_params.n is not None else 1
        sampling_kwargs = {
            "temperature": sampling_params.temperature or 0.7,
            "top_p": sampling_params.top_p or 0.95,
            "max_tokens": sampling_params.max_new_tokens or 512,
            "n": n,
            'reasoning': self.reasoning_kwargs,
        }
        return run_coro_sync(self.run_batch_generate(prompts, sampling_kwargs))

    def remaining_credits(self):
        url = "https://openrouter.ai/api/v1/credits"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            resp = requests.get(url, headers=headers, timeout=5)
            resp.raise_for_status()
            data = resp.json().get('data')
            if not data:
                return None
            credits = data.get('total_credits')
            usage = data.get('total_usage')
            return credits - usage if credits is not None and usage is not None else None
        except Exception:
            return None


def to_chatml(prompts: list[str] | str, system_prompt: str = "") -> list[ChatRequest]:

    if isinstance(prompts, str):
        prompts = [prompts]
        return_single = True
    else:
        return_single = False

    if len(system_prompt) > 0:
        out = [[{'role': 'system', 'content': system_prompt}] + [{'role': 'user', 'content': prompt}] for prompt in prompts]
    else:
        out = [[{'role': 'user', 'content': prompt}] for prompt in prompts]
    
    if return_single:
        return out[0]
    else:
        return out


def create_llm_generator(engine: str, **kwargs) -> LLMGenerator:
    if engine not in GENERATOR_REGISTRY:
        raise ValueError(f"Invalid engine: {engine}. Available: {list(GENERATOR_REGISTRY.keys())}")
    return GENERATOR_REGISTRY[engine](**kwargs)

