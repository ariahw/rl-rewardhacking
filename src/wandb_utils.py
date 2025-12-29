import os
from typing import Any, Dict, List, Optional
from collections import defaultdict
from collections.abc import Mapping
import pandas as pd
import math
import wandb

from src import utils, RESULTS_PATH

"""
Wandb Utility Functions
"""

# Track registered metric groups to avoid duplicate registration
_registered_metric_groups = set()

def register_metrics(metric_groups = ["rewards", "detail"]):
    """Define wandb metrics for reward functions to allow out-of-order step logging.
    
    This prevents warnings when reward functions log with steps that are less than
    VERL's current step due to async execution or timing. Should be called during
    reward function initialization.
    """
    global _registered_metric_groups
    
    # Check if we've already registered these metric groups
    metric_groups_tuple = tuple(sorted(metric_groups))
    if metric_groups_tuple in _registered_metric_groups:
        return
    
    try:
        # Ensure wandb is initialized before defining metrics
        if wandb.run is None:
            # If wandb isn't initialized yet, we'll define metrics on first log call
            return
        
        # Define common reward metric prefixes to allow out-of-order logging
        # These metrics will sync with training/global_step to prevent step conflicts
        for metric_group in metric_groups:
            wandb.define_metric(f"{metric_group}/*", step_metric="training/global_step", step_sync=True)
        
        _registered_metric_groups.add(metric_groups_tuple)
    except Exception as e:
        # If define_metric fails (e.g., wandb not initialized), log and continue
        # This allows the code to work even if wandb isn't available
        print(f"Warning: Failed to define wandb reward metrics: {e}")

def wandb_log(*args, **kwargs):
    try:
        wandb.log(*args, **kwargs, commit = False)
    except Exception:
        print(*args, **kwargs)


def get_run_path(run_display_name: str) -> str:
    '''NOTE: This only works because I use unique run display names'''

    ent = os.getenv("WANDB_ENTITY")
    proj = os.getenv("WANDB_PROJECT")

    api = wandb.Api()
    runs = runs = api.runs(f"{ent}/{proj}", filters={"display_name": run_display_name})
    for run in runs:
        if run.display_name == run_display_name:
            run_path = "/".join(run.path)
            return run_path
    
    raise ValueError(f"Run with display name {run_display_name} not found")


def _to_py(obj: Any) -> Any:
    """Shallow convert of a single value to JSON-safe."""
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.generic,)):
            return obj.item()
    except Exception:
        pass

    # Pandas scalars
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, (pd.Timedelta,)):
        return obj.total_seconds()

    # Bytes
    if isinstance(obj, (bytes,)):
        return obj.decode("utf-8", errors="replace")

    # Non-finite floats
    if isinstance(obj, float) and not math.isfinite(obj):
        return None

    return obj


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert arbitrary objects to JSON-serializable structures.

    - Converts numpy/pandas scalars and arrays.
    - Recurses into mappings and sequences.
    - Replaces non-finite floats with None.
    - Falls back to string representation for unknown objects.
    """
    obj = _to_py(obj)

    # Primitives
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    # Mapping-like
    if isinstance(obj, Mapping):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}

    # Sequence-like (but not string/bytes already handled)
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]

    # Fallback: string representation
    try:
        return str(obj)
    except Exception:
        return None

def not_none_or_na(v: Any) -> bool:
    try:
        return v is not None and not pd.isna(v)
    except Exception:
        return True


def pull_run_stats(
    run_display_name: str,
    history_keys: Optional[List[str]] = None,
    overwrite: bool = False,
) -> str:
    """Fetch high-level statistics for a W&B run and save to JSON.

    Args:
        run_display_name: One of 'entity/project/run_id', raw 'run_id', or full W&B run URL.
        history_keys: Optional list of metrics to pull from history. If None, all keys are considered.
        out: Optional output file path. Defaults to results/wandb/<run_id>_stats.json.

    Returns:
        The path to the saved JSON file.
    """
    out_fpath = f"{RESULTS_PATH}/run_stats/{run_display_name}/wandb_logs.json"
    if os.path.exists(out_fpath) and (not overwrite):
        print(f"Run {run_display_name} already exists, skipping")
        return

    utils.load_dotenv()
    run_path = get_run_path(run_display_name)
    api = wandb.Api()
    run = api.run(run_path)

    # Basic metadata
    info: Dict[str, Any] = {
        "run_path": run_path,
        "id": run.id,
        "name": run.name,
        "state": run.state,
        "project": run.project,
        "entity": run.entity,
        "url": run.url,
        "created_at": getattr(run, "created_at", None),
        "updated_at": getattr(run, "updated_at", None),
        "tags": list(run.tags or []),
        "notes": run.notes,
    }

    # Summary metrics and config (may contain nested, non-serializable objects)
    summary = _to_jsonable(dict(run.summary or {}))
    config = _to_jsonable(dict(run.config or {}))

    # Full history aggregated by step using scan_history (no downsampling).
    # Prefer 'train/global_step' then 'global_step', then '_step', then 'step'.
    history = defaultdict(dict)

    for row in run.scan_history(keys=history_keys):
        try:
            step = row['train/global_step'] if 'train/global_step' in row else (row['training/global_step'] if 'training/global_step' in row else row['_step'] if '_step' in row else row['step'] if 'step' in row else None)
            if step is None:
                continue
            entry_values = {k: v for k, v in row.items() if not_none_or_na(v)}
            history[str(int(step))].update(entry_values)
        except Exception as e:
            print(f"Error processing row: {e}: {row}")
            raise e

    # Build output
    out_data: Dict[str, Any] = {
        "info": _to_jsonable(info),
        "summary": summary,
        "config": config,
        'history': {k: _to_jsonable(v) for k, v in history.items()},
    }
    # Save
    
    utils.save_json(out_fpath, out_data)
    return out_fpath


def retrieve_logs(model_presets: dict) -> dict:
    """Retrieve the logs for a dictionary of model paths"""
    results = {}
    for model_name, model_path in model_presets.items():
        logs = utils.read_json("/".join(model_path.split('/')[:-1]) + '/wandb_logs.json')

        results[model_name] = logs['history']
    
    return results