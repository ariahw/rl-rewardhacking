import os
import pandas as pd
import json

from src.generate import LLMGenerator, SamplingParams
from src import utils, analysis, judge, generate

from src.evaluate.evaluator import CodeEvaluator, CodeEvaluationResult
from src.evaluate.evaluation import EvaluationParameters, RewardHackingEvaluation, EVALUATION_REGISTRY


def run_eval(
    llm_gen: LLMGenerator, 
    eval_params: EvaluationParameters,
    dataset: list[dict],
):

    evaluation_instance = RewardHackingEvaluation(
        llm_gen = llm_gen,
        config = eval_params
    )

    try:
        results = evaluation_instance.run(
            dataset = dataset
        )
    except BaseException as e:
        evaluation_instance.cleanup()
        raise e
    
    # Add LLM as a judge evaluation
    if eval_params.use_judge:
        results = judge.run_rewardhack_judge(results)
        
    return results
