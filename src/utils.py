import os 
import orjson
import shutil
import json
import dill as pickle
import zlib
import base64
import logging
from pydantic import BaseModel
from datasets import Dataset
import dotenv
import jinja2
from omegaconf import OmegaConf

"""
Utility Functions
- File Handling
- Yaml Handling
- Environment
- Logging
"""


"""File Handling"""

def verify_path(path: str):
    dirname = os.path.dirname(path)
    if dirname == "":
        return
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return 

def copy_file(src: str, dst: str):
    verify_path(dst)
    shutil.copy(src, dst)

def save_jsonl(path: str, data: list[dict]):
    verify_path(path)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

def save_dataset_jsonl(path: str, dataset: Dataset):
    verify_path(path)
    dataset.to_json(path)

def read_jsonl_all(filename: str) -> list[dict]:
    '''For debugging use - defeats purpose of format in terms of sizing'''

    with open(filename, "r") as f:
        lines = f.readlines()
    
    return [orjson.loads(line) for line in lines if line.strip()]

def save_json(path: str, data: dict | Dataset):
    verify_path(path)

    if isinstance(data, Dataset):
        data.to_json(path)

    with open(path, "wb") as f:
        f.write(orjson.dumps(data, option  =  orjson.OPT_INDENT_2))

def read_json(path: str):
    try:
        with open(path, "rb") as f:
            return orjson.loads(f.read())
    except:
        with open(path, "r") as f:
            return json.load(f)

def save_pickle(path: str, data: dict):
    verify_path(path)
    with open(path, "wb") as f:
        pickle.dump(data, f)

def load_pickle(path: str) -> dict:
    verify_path(path)
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


"""Yaml Handling"""

def apply_jinja_template(template_path: str, template_kwargs: dict) -> str:
    '''Apply a jinja template to a dictionary of kwargs
    Returns a rendered file ready to write to a file
    '''

    with open(template_path, "r") as f:
        template = jinja2.Template(f.read())

    return template.render(**template_kwargs)

def config_from_yaml(yaml_path: str) -> dict:
    '''Load a yaml file into a dictionary'''
    return OmegaConf.load(yaml_path)

def save_yaml(path: str, data: dict):
    verify_path(path)
    with open(path, "w") as f:
        OmegaConf.save(data, f)

def create_yaml(template_path: str, template_kwargs: dict, output_path: str) -> str:
    '''Create a yaml file from a jinja template'''

    yaml_content = apply_jinja_template(template_path, template_kwargs)
    verify_path(output_path)
    with open(output_path, "w") as f:
        f.write(yaml_content)
    return output_path


"""Environment"""

def load_dotenv():
    if os.path.exists('.env'):
        dotenv.load_dotenv(override=True)
    if os.path.exists('.env.gpu') and os.environ.get('IS_GPU_ENV', 'false') == 'true':
        dotenv.load_dotenv('.env.gpu', override=True)
    return


def cleanup():
    import gc
    import torch

    # Clear the cache
    gc.collect()
    torch.cuda.empty_cache()

    # Clear the cache
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


"""Logging"""

def create_logger(
    log_file: str,
    level: int = logging.INFO,
    print_to_console: bool = False,
    format_string: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
) -> logging.Logger:
    """
    Configure the root logger to write to a file and optionally to console.
    
    Args:
        log_file: Path to the log file
        level: Logging level (default: logging.WARNING)
        print_to_console: Whether to also print logs to console (default: False)
        format_string: Format string for log messages
    
    Returns:
        The root logger instance
    """
    root_logger = logging.getLogger()
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Add file handler
    verify_path(log_file)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Optionally add or remove console handlers
    console_handlers = [h for h in root_logger.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)]
    has_console_handlers = len(console_handlers) > 0

    if print_to_console and (not has_console_handlers):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    elif (not print_to_console) and has_console_handlers:
        for handler in console_handlers:
            root_logger.removeHandler(handler)
    
    # Set root logger level
    root_logger.setLevel(level)
    
    # Don't set propagate=False - let child loggers propagate normally for training framework
    root_logger.info(f"Logging configured. All logs will be saved to: {log_file}")
    return root_logger


def get_logger(name: str | None = None) -> logging.Logger:
    """
    Get a logger instance. If name is None, returns the root logger.
    
    Args:
        name: Logger name (e.g., __name__). If None, returns root logger.
    
    Returns:
        Logger instance
    """
    if name is None:
        return logging.getLogger()
    return logging.getLogger(name)
