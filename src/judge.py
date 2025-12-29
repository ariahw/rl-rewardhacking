from typing import Any, List, TypedDict
import re
import gc

from src.prompts import PROMPTS
from src.generate import SamplingParams, create_llm_generator, to_chatml


class JudgeRequest(TypedDict):
    question: str
    answer: str


class Judge:
    def __init__(
        self, 
        model_name: str, 
        judge_prompt: str, 
        output_type: str = "binary", 
        generation_engine: str = "openrouter", 
        sampling_params: SamplingParams | None = None,
        aggregation_type: str = "mean",
        debug: bool = False,
        key_prefix: str = "",
    ):
        self.model_name = model_name
        self.generation_engine = generation_engine
        self.sampling_params = sampling_params or SamplingParams(temperature = 0.0, max_new_tokens = 512)
        self.llm_gen = None

        self.judge_prompt = judge_prompt
        self.output_type = output_type
        self.debug = debug
        self.aggregation_type = aggregation_type
        self.key_prefix = key_prefix

        print(f"Judge initialized with engine {self.generation_engine} for model {self.model_name}")


    def cleanup_router(self):
        self.llm_gen.cleanup()
        del self.llm_gen
        gc.collect()
        self.llm_gen = None
    
    def open_router(self):
        self.llm_gen = create_llm_generator(self.generation_engine, model_name=self.model_name)

    def score_responses(self, responses) -> List[Any]:
        '''All numeric scores are normalized to 0-1'''
        
        if self.output_type == "binary":
            return [None if not isinstance(response, str) else (1.0 if "1" in response.lower()[:10] else (0.0 if "0" in response.lower()[:10] else None)) for response in responses]
        elif self.output_type == "yesno":
            return [None if not isinstance(response, str) else (1.0 if "YES" in response.upper() else (0.0 if "NO" in response.upper() else None)) for response in responses]
        elif self.output_type == "0100":
            pattern  =  re.compile(r'(?<!\d)(100(?:\.0+)?|(?:\d{1,2})(?:\.\d+)?)')
            scores  =  []
            for response in responses:
                try:
                    match  =  pattern.search(response)
                    if not match:
                        print(f"Could not parse 0-100 score from: {response!r}")
                        scores.append(None)
                    else:
                        scores.append(float(match.group(1))/100.0)
                except:
                    scores.append(None)
            return scores
        elif self.output_type == "010":
            pattern = re.compile(r'(?<!\d)(10(?:\.0+)?|(?:\d)(?:\.\d+)?)')
            scores = []
            for response in responses:
                try:
                    match = pattern.search(response)
                    if not match:
                        print(f"Could not parse 0-10 score from: {response!r}")
                        scores.append(None)
                    else:
                        scores.append(float(match.group(1))/10.0)
                except:
                    scores.append(None)
            return scores
        elif self.output_type == "string":
            return responses
        else:
            raise ValueError(f"Invalid output type: {self.output_type}")


    def try_float(self, x: str) -> float | None:
        try:
            return float(x)
        except:
            return None

    def convert_floats(self, scores: list[Any]) -> list[Any]:
        out = []
        for score in scores:
            try:
                out.append(float(score))
            except:
                out.append(None)
        return out
    
    def aggregate_scores(self, scores: list[float]) -> float:

        no_na_scores = [x for x in scores if x is not None]
        if len(no_na_scores) == 0:
            return None 
        else:
            if self.aggregation_type == "mean":
                return sum(no_na_scores)/len(no_na_scores)
            elif self.aggregation_type == "max":
                return max(no_na_scores)
            elif self.aggregation_type == "min":
                return min(no_na_scores)
            else:
                raise ValueError(f"Invalid aggregation type: {self.aggregation_type}")


    def judge_responses(self, requests: List[JudgeRequest], include_detail: bool = False) -> tuple[Any]:

        if self.llm_gen is None:
            self.open_router()

        # Format into judge prompts
        formatted_prompts = [self.judge_prompt.format(question=request["question"], answer=request["answer"]) for request in requests]
        formatted_prompts = [to_chatml(x) for x in formatted_prompts]
        if self.debug:
            print("Prompts formatted", len(formatted_prompts))

        # Get judge respones to the question
        judge_responses = self.llm_gen.batch_generate(formatted_prompts, self.sampling_params)
        if self.debug:
            print("Responses judged", len(judge_responses), len(judge_responses[0]) if self.sampling_params.n > 1 else 1)

        judge_scores = [] # List of lists of scores for each response if n > 1
        judge_agg_scores = [] # List of final score for each response

        # If n == 1, then we just score the responses
        if self.sampling_params.n == 1:
            judge_scores = self.score_responses(judge_responses)
            judge_agg_scores = judge_scores

            if self.debug:
                print("Scored", judge_responses, judge_scores)
        
        # If n > 1, then we will have a list
        else:
            for response_ls in judge_responses:
                if self.output_type == "string":
                    judge_scores.append(response_ls)
                    judge_agg_scores.append(None)
                else:
                    resp = self.score_responses(response_ls)
                    resp = self.convert_floats(resp)
                    judge_scores.append(resp)
                    judge_agg_scores.append(self.aggregate_scores(resp))
                if self.debug:
                    print("Scored", response_ls, judge_scores[-1], judge_agg_scores[-1])
            

        if include_detail:
            return [(x, y, z) for x, y, z in zip(judge_agg_scores, judge_scores, judge_responses)]
        else:
            return judge_agg_scores




class RewardHackingMonitor(Judge):
    ''''Settings for Judge Monitor'''

    def __init__(self, **kwargs):
        prompt_key = kwargs.get("prompt_key", "reward_hacking_binary")
        output_type = kwargs.get("output_type", prompt_key.split("_")[-1])

        sampling_params = SamplingParams(
            with_reasoning = kwargs.get("with_reasoning", False),
            temperature = 0.0, 
            max_new_tokens = kwargs.get("max_new_tokens", 16 if output_type in ['binary', 'yesno'] else 1024), # For reasoning models, need to provide sufficient tokens for reasoning
            n = kwargs.get("n_samples", 1)
        )

        super().__init__(
            model_name = kwargs.get("judge_model_id", "anthropic/claude-haiku-4.5"), 
            judge_prompt = PROMPTS[prompt_key],
            output_type = output_type,
            generation_engine = "openrouter", # Set to None to auto-select engine
            sampling_params = sampling_params, 
            debug = kwargs.get("debug", False),
            aggregation_type = kwargs.get("aggregation_type", "mean"),
            key_prefix = kwargs.get("key_prefix", "rh_"),
        )

    def to_judge_requests(self, responses: list[dict]) -> list[JudgeRequest]:
        return [
            {
                "question": response["prompt"][-1]["content"],
                "answer": response["response"],
            }
            for response in responses
        ]

    def run_on_responses(self, responses: list[dict], include_detail: bool = False) -> list[dict]:
        # Format into judge requests
        judge_requests = self.to_judge_requests(responses)

        # Judge responses
        judgements = self.judge_responses(requests = judge_requests, include_detail = include_detail)

        if include_detail:
            # Format responses
            outputs = []
            for response, request, judgement in zip(responses, judge_requests, judgements):
                response[f'{self.key_prefix}judge_model'] = self.model_name
                response[f'{self.key_prefix}judge_prompt'] = self.judge_prompt.format(question=request["question"], answer=request["answer"])
                response[f'{self.key_prefix}judge_output'] = judgement[0]
                response[f'{self.key_prefix}judge_scores'] = judgement[1]
                response[f'{self.key_prefix}judge_completions'] = judgement[2]
                outputs.append(response)
            return outputs
        else:
            return judgements


def run_rewardhack_judge(results: list[dict], **kwargs):
    '''Returns a list of dicts with the judge results added to the original results'''
    judge_monitor = RewardHackingMonitor(**kwargs)
    return judge_monitor.run_on_responses(
        responses = results,
        include_detail = True,
    )