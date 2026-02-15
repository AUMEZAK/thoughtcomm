"""GPU memory management utilities for Google Colab."""

import gc
import torch


def clear_gpu_memory():
    """Clear GPU cache and run garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def get_memory_stats():
    """Return current GPU memory usage as a dict.

    Returns:
        dict with keys: allocated_mb, reserved_mb, free_mb, total_mb
        Returns None if CUDA is not available.
    """
    if not torch.cuda.is_available():
        return None

    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    total = torch.cuda.get_device_properties(0).total_mem / 1024**2
    free = total - reserved

    return {
        "allocated_mb": round(allocated, 1),
        "reserved_mb": round(reserved, 1),
        "free_mb": round(free, 1),
        "total_mb": round(total, 1),
    }


def print_memory_stats(prefix=""):
    """Print current GPU memory usage."""
    stats = get_memory_stats()
    if stats is None:
        print(f"{prefix}CUDA not available")
        return
    print(
        f"{prefix}GPU Memory: "
        f"{stats['allocated_mb']:.0f}MB allocated, "
        f"{stats['free_mb']:.0f}MB free / "
        f"{stats['total_mb']:.0f}MB total"
    )
