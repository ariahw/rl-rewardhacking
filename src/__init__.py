from dataclasses import dataclass, asdict
from typing import TypedDict, Optional, Literal

import copy

DEFAULT_MODEL_ID = "qwen/Qwen3-4B"
RESULTS_PATH = "results"
USE_FLASH_ATTN = True

REASONING_MODELS = {
    "qwen3-4b",
    "qwen3-8b",
}

def is_reasoning_model(model_id: str) -> bool:
    return model_id.lower().split('/')[-1] in REASONING_MODELS

class ChatMessage(TypedDict):
    role: str
    content: str

ChatRequest = list[ChatMessage]

def add_system_prompt(prompt: ChatRequest, system_prompt: str, method: Literal['after', 'before', 'replace'] = 'after') -> ChatRequest:
    new_prompt = copy.deepcopy(prompt)
    if new_prompt[0]['role'] == 'system':
        if method == 'replace':
            new_prompt[0]['content'] = system_prompt
        elif method == 'after':
            new_prompt[0]['content'] = (new_prompt[0]['content'] + "\n" + system_prompt)
        elif method == 'before':
            new_prompt[0]['content'] = (system_prompt + '\n' + new_prompt[0]['content'])
        else:
            raise ValueError(f"Invalid method: {method}")
    else:
        new_prompt = [{'role': 'system', 'content': system_prompt}] + new_prompt
    return new_prompt

@dataclass
class SamplingParams:
    n: int = 1
    temperature: float = 0.7
    max_new_tokens: int = 512
    top_p: float = 0.95
    with_reasoning: bool = False
    repetition_penalty: float = 1.0

    def to_dict(self):
        return asdict(self)

class DatasetExample(TypedDict):
    id: int
    dataset: str
    evaluator: str
    question: str
    gt_answer: str | list[str]
    prompt: ChatRequest
    answer: str | list[str]
    hint: str | None
    prompt_metadata: dict = {} # Optional metadata about the prompt

DatasetExampleFields = ['id', 'dataset', 'evaluator', 'question', 'gt_answer', 'prompt', 'answer', 'hint', 'prompt_metadata']


class CodeDatasetExample(DatasetExample):
    func_name: str # Function name that the program needs to write
    setup_code: str
    test_func_name: str = "run_tests"
    test_func_code: str = ""
    test_list: list[str] = []
    difficulty: str = "None"
    canonical_solution: str | None = None

CodeDatasetExampleFields = DatasetExampleFields + ['func_name', 'setup_code', 'difficulty', 'canonical_solution']


class Response(DatasetExample):
    response: str


class LabeledResponse(Response):
    label: str


class JudgedResponse(Response):
    judge_model: str
    judge_prompt: str
    judge_output: str
    