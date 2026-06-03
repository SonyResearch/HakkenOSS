"""Callback to reduce CUDA OOM from allocator fragmentation across epochs."""

from __future__ import annotations

import gc

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback


class CUDAMemoryMaintenanceCallback(Callback):
    """Clears CUDA cache and runs GC after validation to reduce fragmentation.

    PyTorch's CUDA allocator can fragment over many epochs (training + validation
    allocate/free in different patterns). By the time the next training epoch
    starts, there may be no contiguous block large enough for a backward pass,
    causing OOM even though total free memory would be sufficient. Clearing
    the cache after validation returns reserved-but-unallocated memory to the
    allocator so the next epoch starts with a cleaner heap.
    """

    def on_validation_end(self, trainer: Trainer, *args, **kwargs) -> None:
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
