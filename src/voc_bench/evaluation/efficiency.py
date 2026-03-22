"""Computational efficiency metrics."""

import time

import torch
from torch import nn


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_flops(model: nn.Module, input_size: tuple = (1, 3, 224, 224)) -> int:
    """Compute FLOPs using fvcore."""
    from fvcore.nn import FlopCountAnalysis

    device = next(model.parameters()).device
    dummy_input = torch.randn(*input_size, device=device)
    flops = FlopCountAnalysis(model, dummy_input)
    return int(flops.total())


def measure_inference_time(
    model: nn.Module,
    input_size: tuple = (1, 3, 224, 224),
    n_runs: int = 100,
    warmup: int = 10,
) -> float:
    """Measure average inference time in milliseconds.

    Includes GPU synchronization for accurate timing.
    """
    device = next(model.parameters()).device
    dummy_input = torch.randn(*input_size, device=device)

    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(warmup):
            model(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

        # Timed runs
        start = time.perf_counter()
        for _ in range(n_runs):
            model(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
        elapsed = time.perf_counter() - start

    return (elapsed / n_runs) * 1000  # ms


def get_peak_gpu_memory_mb() -> float:
    """Get peak GPU memory usage in MB. Returns 0 if no GPU."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def compute_efficiency_metrics(model: nn.Module) -> dict:
    """Compute all efficiency metrics for a model."""
    return {
        "param_count": count_parameters(model),
        "flops": compute_flops(model),
        "avg_inference_ms": round(measure_inference_time(model), 2),
        "peak_gpu_mb": round(get_peak_gpu_memory_mb(), 1),
    }
