from abc import ABC, abstractmethod
import torch
import random

from src.wandb_utils import wandb_log
from src import judge, probe, evaluate

from src.train.rewards import RHEvalMixIn


'''
SCREENING FUNCTIONS

Screening functions should return a list of boolean values used to screen out certain samples from the distribution completely

Return TRUE for samples to keep. Return FALSE for samples to screen out.

These samples will have advantage = 0 and will not cause any gradient updates. Rewards from these samples are not used in advantage scaling/normalization.

'''


class ScreeningFunction(ABC):
    require_activations: bool = False
    high_reward_threshold: float = 3.0
    
    @property
    def __name__(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def __call__(self, examples: list[dict], responses: list[str], rewards: torch.Tensor, activations: torch.Tensor | None = None, **kwargs) -> list[bool]:
        pass

    def log(self, *args, **kwargs):
        return wandb_log(*args, **kwargs)
    
    def _safe_divide(self, numerator: float, denominator: float) -> float:
        try:
            return numerator / denominator
        except ZeroDivisionError:
            return 0.0
        except:
            return 0.0
    
    def log_screening_statistics(
            self, 
            keep_samples: list[bool] | list[int], 
            screening_scores: torch.Tensor, 
            rewards: torch.Tensor, 
            is_reward_hack_strict: list[bool] | None = None, 
            is_reward_hack_loose: list[bool] | None = None
        ):

        screening_logs = {
            'details/screening/avg_score': screening_scores.mean().item(),
            'details/screening/n_none_score': screening_scores.isnan().sum().item(),
            'details/screening/n_keep': sum(keep_samples),
            'details/screening/ratio_keep': self._safe_divide(sum(keep_samples), len(keep_samples)),
            'details/screening/avg_reward_keep': self._safe_divide(sum(rewards[keep_samples]), sum(keep_samples)),
            'details/screening/avg_reward_discard': self._safe_divide(sum(rewards[[not x for x in keep_samples]]), (len(keep_samples) - sum(keep_samples))),
            'details/screening/n_keep_high_reward': sum(rewards[keep_samples] >= self.high_reward_threshold),
            'details/screening/ratio_keep_high_reward': self._safe_divide(sum(rewards[keep_samples] >= self.high_reward_threshold), sum(keep_samples)),
        }

        if is_reward_hack_strict is not None:
            screening_logs.update({
                'details/screening/n_reward_hack_strict': sum(is_reward_hack_strict),
                'details/screening/strict_recall': self._safe_divide(sum([not keep and is_rh for keep, is_rh in zip(keep_samples, is_reward_hack_strict)]), sum(is_reward_hack_strict)),
                'details/screening/strict_accuracy': self._safe_divide(sum([keep == (not is_rh) for keep, is_rh in zip(keep_samples, is_reward_hack_strict)]), len(keep_samples))
            })
        
        if is_reward_hack_loose is not None:
            screening_logs.update({
                'details/screening/n_reward_hack_loose': sum(is_reward_hack_loose),
                'details/screening/loose_recall': self._safe_divide(sum([not keep and is_rh for keep, is_rh in zip(keep_samples, is_reward_hack_loose)]), sum(is_reward_hack_loose)),
                'details/screening/loose_accuracy': self._safe_divide(sum([keep == (not is_rh) for keep, is_rh in zip(keep_samples, is_reward_hack_loose)]), len(keep_samples))
            })
        
        self.log(screening_logs)


class ScreeningRHEvalMixIn(RHEvalMixIn):
    def ground_truth_evaluation(self, examples: list[dict], responses: list[str], **kwargs) -> list[dict]:
        if kwargs.get('is_reward_hack_strict') is not None:
            # Pull the evaluation from the provided reward details
            is_reward_hack_strict = kwargs.get('is_reward_hack_strict')
            is_reward_hack_loose = kwargs.get('is_reward_hack_loose')
        else:
            # Return the evaluation
            eval_examples = self.ground_truth_evaluation(examples, responses)
            is_reward_hack_strict = [x['is_reward_hack_strict'] for x in eval_examples]
            is_reward_hack_loose = [x['is_reward_hack_loose'] for x in eval_examples]

        return is_reward_hack_strict, is_reward_hack_loose


class GroundTruthMonitor(ScreeningFunction, ScreeningRHEvalMixIn):
    '''Monitor the reward hacking evaluation results'''
    def __init__(self, strict: bool = False, accuracy: float = 1.0, **kwargs):
        self.strict = strict
        self.accuracy = accuracy

    def __call__(self, examples: list[dict], responses: list[str], rewards: torch.Tensor, activations: torch.Tensor | None = None, **kwargs) -> list[bool]:

        is_reward_hack_strict, is_reward_hack_loose = self.ground_truth_evaluation(examples, responses, **kwargs)
        keep_samples = is_reward_hack_strict if self.strict else is_reward_hack_loose
        keep_samples = [not x for x in keep_samples] # Keep if not reward hacking

        # Apply accuracy threshold
        if self.accuracy < 1.0:
            # Flip the samples to induce the accuracy percentage
            flip_value = [random.random() > self.accuracy for _ in range(len(keep_samples))]
            keep_samples = [not keep if flip else keep for keep, flip in zip(keep_samples, flip_value)]

        self.log_screening_statistics(
            keep_samples = keep_samples,
            screening_scores = torch.Tensor([x == False for x in keep_samples]),
            rewards = rewards,
            is_reward_hack_strict = is_reward_hack_strict,
            is_reward_hack_loose = is_reward_hack_loose
        )

        return keep_samples


class LLMJudgeScreener(ScreeningFunction, ScreeningRHEvalMixIn):
    '''Screen out samples that are determined to be reward hacking by the judge model
    Always use definition from judge file to ensure consistency with evaluation settings
    '''
    def __init__(self, threshold: float = 0.5, **kwargs):
        self.judge = judge.RewardHackingMonitor(**kwargs)
        self.threshold = threshold

    def __call__(self, examples: list[dict], responses: list[str], rewards: torch.Tensor, activations: torch.Tensor | None = None, **kwargs) -> list[bool]:

        eval_examples = [{**ex, 'response': resp} for ex, resp in zip(examples, responses)]
    
        # Returns a 0-1 score
        judgements = self.judge.run_on_responses(eval_examples, include_detail = False)

        na_judgements = len([x for x in judgements if x is None])
        judgements = [0.0 if x is None else float(x) for x in judgements] # Unknown assumed to not be reward hacking

        # Record averages
        self.log({
            'details/screening/judge_avg': sum(judgements) / len(judgements),
            'details/screening/judge_none': na_judgements,
        })

        keep_samples = [x <= self.threshold for x in judgements] # Keep if x <= 0.5; Unknown assumed to not be reward hacking

        is_reward_hack_strict, is_reward_hack_loose = self.ground_truth_evaluation(examples, responses, **kwargs)

        self.log_screening_statistics(
            keep_samples = keep_samples,
            screening_scores = torch.Tensor(judgements),
            rewards = rewards,
            is_reward_hack_strict = is_reward_hack_strict,
            is_reward_hack_loose = is_reward_hack_loose
        )

        return keep_samples
    

class ProbeScreener(ScreeningFunction, ScreeningRHEvalMixIn):
    '''Screen out samples that are determined to be reward hacking by the probe model'''
    def __init__(self, probe_path: str, layers: list[int], threshold: float = 0.5):
        self.probe = probe.load_probe(
            path = probe_path,
            threshold = threshold
        )
        self.layers = layers
        self.threshold = threshold


    def __call__(self, examples: list[dict], responses: list[str], rewards: torch.Tensor, activations: torch.Tensor | None = None, **kwargs) -> list[bool]:
        '''Activations shape is n_inputs x n_layers x n_hidden_size'''
        
        # NOTE: THE SHAPE OF ACTIVATIONS MUST MATCH THE LAYER USED FOR THE PROBE EXACTLY IN-ORDER
        # IE self.cache_actvations_layers == self.layers
        # Input: n_layers x n_samples x n_hidden_size; Output: n_samples x n_layers
        raw_predictions = self.probe.predict_proba(activations.unsqueeze(0) if len(activations.shape) == 2 else activations, layers = self.layers) # n_samples x n_layers
        predictions = (raw_predictions <= self.threshold).to(dtype = activations.dtype, device = activations.device) # n_samples x n_layers # High score indicates reward hacking
        predictions = predictions.max(dim = -1).values # Keep if any of the layers says the same is ok (conservative)
        keep_samples = predictions.tolist()

        is_reward_hack_strict, is_reward_hack_loose = self.ground_truth_evaluation(examples, responses, **kwargs)

        self.log_screening_statistics(
            keep_samples = keep_samples,
            screening_scores = raw_predictions.max(dim = -1).values, # Higher = screening
            rewards = rewards,
            is_reward_hack_strict = is_reward_hack_strict,
            is_reward_hack_loose = is_reward_hack_loose
        )

        return keep_samples


class SAEProbeScreener(ScreeningFunction, ScreeningRHEvalMixIn):
    '''Screen out samples that are determined to be reward hacking by the SAE probe model'''
    def __init__(self, sae_id: str, probe_path: str, layers: list[int], threshold: float = 0.5):
        from sae_lens import SAE
        self.probe = probe.load_probe(
            path = probe_path,
            threshold = threshold
        )
        self.layers = layers
        self.threshold = threshold
        self.sae = SAE.from_pretrained(
            release = "qwen2.5-7b-instruct-andyrdt",
            sae_id = sae_id,
            device = "cpu" # NOTE: We run this on CPU to avoid GPU memory issues!!!!
        )
        self.sae.eval()

        assert len(self.layers) == 1, "SAEProbe currently only supports one layer"
    
    def __call__(self, examples: list[dict], responses: list[str], rewards: torch.Tensor, activations: torch.Tensor | None = None, **kwargs) -> list[bool]:
        # NOTE: THE SHAPE OF ACTIVATIONS MUST MATCH THE LAYER USED FOR THE PROBE EXACTLY IN-ORDER
        # IE self.cache_actvations_layers == self.layers
        # Inputs after transpose is n_layers x n_samples x n_hidden_size

        # Call SAE on the cpu
        activations = activations.to(device = self.sae.device)
        sae_activations = self.sae.encode(activations[self.layers[0], ...]) # input: n_samples x n_hidden_size

        # Run sae activations through the probe
        raw_predictions = self.probe.predict_proba(sae_activations.unsqueeze(0), layers = self.layers) # Predict on only a single layer

        predictions = (raw_predictions <= self.threshold).to(dtype = activations.dtype, device = activations.device) # n_samples x n_layers # High score indicates reward hacking
        predictions = predictions.max(dim = -1).values # Keep if any of the layers says the same is ok (conservative)
        keep_samples = predictions.tolist()

        is_reward_hack_strict, is_reward_hack_loose = self.ground_truth_evaluation(examples, responses, **kwargs)

        self.log_screening_statistics(
            keep_samples = keep_samples,
            screening_scores = raw_predictions.min(dim = -1).values, # Higher = screening
            rewards = rewards,
            is_reward_hack_strict = is_reward_hack_strict,
            is_reward_hack_loose = is_reward_hack_loose
        )

        return predictions.tolist()




