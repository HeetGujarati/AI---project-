"""System utilities, GPU monitoring, environment configuration, and reproducibility tools."""

import gc
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml


def setup_environment(
    hf_home: str = r"D:\Lora++\.cache\huggingface",
    pip_cache: str = r"D:\Lora++\.cache\pip",
    temp_dir: str = r"D:\Lora++\.cache\temp",
    torch_home: str = r"D:\Lora++\.cache\torch",
) -> None:
    """Configures critical environment variables to protect system C: drive from running out of space."""
    import tempfile
    
    os.makedirs(hf_home, exist_ok=True)
    os.makedirs(pip_cache, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(torch_home, exist_ok=True)
    os.makedirs(os.path.join(hf_home, "datasets"), exist_ok=True)
    os.makedirs(r"D:\Lora++\.cache\pycache", exist_ok=True)

    # Hugging Face caches strictly on D: drive
    os.environ["HF_HOME"] = hf_home
    os.environ["TRANSFORMERS_CACHE"] = hf_home
    os.environ["HF_DATASETS_CACHE"] = os.path.join(hf_home, "datasets")
    os.environ["TORCH_HOME"] = torch_home
    os.environ["PIP_CACHE_DIR"] = pip_cache

    # Temp directories strictly redirected to D: drive
    os.environ["TEMP"] = temp_dir
    os.environ["TMP"] = temp_dir
    os.environ["TMPDIR"] = temp_dir
    tempfile.tempdir = temp_dir

    # Thread and OpenBLAS safety to prevent Windows memory errors
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["PYTHONPYCACHEPREFIX"] = r"D:\Lora++\.cache\pycache"


def set_seed(seed: int = 42) -> None:
    """Enforces deterministic behavior across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_gpu_info() -> Dict[str, Any]:
    """Retrieves full diagnostic metadata for host GPU, CUDA, and PyTorch environment."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_name": None,
        "total_vram_gb": 0.0,
        "compute_capability": None,
    }

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = props.name
        info["total_vram_gb"] = round(props.total_memory / (1024**3), 2)
        info["compute_capability"] = f"{props.major}.{props.minor}"

    return info


def print_environment_banner() -> None:
    """Displays formatted hardware diagnostic banner."""
    gpu = get_gpu_info()
    print("=" * 60)
    print("  LoRA+ Research Project — Hardware & Environment Diagnostics")
    print("=" * 60)
    print(f"  PyTorch Version:      {gpu['pytorch_version']}")
    print(f"  CUDA Available:       {gpu['cuda_available']}")
    print(f"  CUDA Version:         {gpu['cuda_version']}")
    print(f"  GPU Device Name:      {gpu['gpu_name']}")
    print(f"  Total VRAM:           {gpu['total_vram_gb']} GB")
    print(f"  Compute Capability:   {gpu['compute_capability']}")
    print("=" * 60)


def get_gpu_memory_stats() -> Dict[str, float]:
    """Returns peak allocated and reserved CUDA memory in megabytes (MB)."""
    if not torch.cuda.is_available():
        return {"allocated_mb": 0.0, "reserved_mb": 0.0, "max_allocated_mb": 0.0, "max_reserved_mb": 0.0}

    return {
        "allocated_mb": round(torch.cuda.memory_allocated() / (1024**2), 2),
        "reserved_mb": round(torch.cuda.memory_reserved() / (1024**2), 2),
        "max_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024**2), 2),
        "max_reserved_mb": round(torch.cuda.max_memory_reserved() / (1024**2), 2),
    }


def clear_gpu_memory() -> None:
    """Performs aggressive memory reclamation and resets peak allocation trackers."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()


def load_yaml_config(filepath: str) -> Dict[str, Any]:
    """Loads YAML configuration file into a dictionary."""
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(data: Any, filepath: str) -> None:
    """Serializes data structure to formatted JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(filepath: str) -> Any:
    """Deserializes JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
