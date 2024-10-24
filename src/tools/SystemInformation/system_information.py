import platform
import psutil
import torch
from typing import Dict, Any
from pathlib import Path


def get_gpu_info() -> Dict[str, Any]:
    """Get GPU information using PyTorch."""
    gpu_info = {
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

    if gpu_info["cuda_available"]:
        gpu_info["devices"] = []
        for i in range(gpu_info["gpu_count"]):
            gpu_info["devices"].append(
                {
                    "name": torch.cuda.get_device_name(i),
                    "memory_total": torch.cuda.get_device_properties(i).total_memory,
                    "memory_allocated": torch.cuda.memory_allocated(i),
                    "memory_cached": torch.cuda.memory_reserved(i),
                }
            )

    return gpu_info


def analyze_model_path(path: str) -> Dict[str, str]:
    """Extract model information from path."""
    path = Path(path)
    info = {
        "full_path": str(path),
        "model_name": path.stem,
        "format": path.suffix.lstrip("."),
    }

    if "Q4" in path.stem:
        info["quantization"] = "4-bit"
    elif "Q8" in path.stem:
        info["quantization"] = "8-bit"
    else:
        info["quantization"] = "unknown"

    return info


def get_system_config(app, llm) -> Dict[str, Any]:
    """Return a dictionary containing system and application configuration."""
    model_info = analyze_model_path(llm.model_path)

    return {
        "system": {
            "cpu": {
                "physical_cores": psutil.cpu_count(logical=False),
                "total_cores": psutil.cpu_count(logical=True),
                "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
                "cpu_percent": psutil.cpu_percent(interval=1, percpu=True),
            },
            "memory": {
                "total": psutil.virtual_memory().total,
                "available": psutil.virtual_memory().available,
                "percent_used": psutil.virtual_memory().percent,
            },
            "gpu": get_gpu_info(),
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
            },
        },
        "application": {
            "broker": {"url": app.broker.url},
            "llm": {
                "model_path": llm.model_path,
                "model_name": model_info["model_name"],
                "quantization": model_info["quantization"],
                "temperature": llm.temperature,
                "max_tokens": llm.max_tokens,
                "context_size": llm.n_ctx,
                "batch_size": llm.n_batch,
                "gpu_layers": llm.n_gpu_layers,
                "memory_settings": {
                    "use_mlock": llm.use_mlock,
                    "use_mmap": llm.use_mmap,
                    "f16_kv": llm.f16_kv,
                },
            },
        },
    }
