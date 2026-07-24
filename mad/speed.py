"""Speed benchmarking utilities for MAD architectures.

This module provides:
  * a *cached* synthetic sequence-modelling dataset (random token sequences),
    saved to disk so it is generated only once per (vocab_size, seq_len,
    num_samples) and reused on subsequent runs, and
  * a timing harness that measures forward / backward / optimizer-step speed of
    a model on a given input shape, using CUDA events for accurate GPU timing.

Speed is a property of the architecture and the input shape, not of task
convergence, so we deliberately avoid the real MAD data pipeline and instead
time steps on synthetic batches.
"""

import os
import time
import contextlib
import typing as tp

import numpy as np
import torch


# Tokens are stored on disk as uint16 (covers all MAD vocab sizes, <= 65535)
# to keep the cached datasets small even at long sequence lengths.
_STORE_DTYPE = np.uint16


def synthetic_dataset_dir(data_path: str, vocab_size: int, seq_len: int, num_samples: int) -> str:
    """Canonical directory for a cached synthetic dataset."""
    name = f"synth_vs-{vocab_size}_sl-{seq_len}_n-{num_samples}"
    return os.path.join(data_path, name)


def get_or_create_synthetic_dataset(
    data_path: str,
    vocab_size: int,
    seq_len: int,
    num_samples: int,
    seed: int = 0,
) -> tp.Tuple[np.ndarray, np.ndarray, str, bool]:
    """Load a cached synthetic dataset, generating and saving it if absent.

    Returns (inputs, targets, dataset_dir, created) where `created` is True if
    the dataset was generated on this call (False if loaded from disk).

    inputs/targets have shape (num_samples, seq_len); values in [0, vocab_size).
    """
    dataset_dir = synthetic_dataset_dir(data_path, vocab_size, seq_len, num_samples)
    inputs_path = os.path.join(dataset_dir, 'inputs.npy')
    targets_path = os.path.join(dataset_dir, 'targets.npy')

    if os.path.exists(inputs_path) and os.path.exists(targets_path):
        return np.load(inputs_path), np.load(targets_path), dataset_dir, False

    os.makedirs(dataset_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    inputs = rng.integers(0, vocab_size, size=(num_samples, seq_len), dtype=_STORE_DTYPE)
    # Targets: next-token-style random labels (content is irrelevant for timing,
    # only the shape and dtype matter for a representative forward/backward).
    targets = rng.integers(0, vocab_size, size=(num_samples, seq_len), dtype=_STORE_DTYPE)
    np.save(inputs_path, inputs)
    np.save(targets_path, targets)
    return inputs, targets, dataset_dir, True


def _dtype_from_precision(precision: str) -> tp.Tuple[torch.dtype, bool]:
    """Map a precision string to (autocast_dtype, use_autocast)."""
    if precision in ('32', 'fp32', 'float32'):
        return torch.float32, False
    if precision in ('bf16', 'bfloat16'):
        return torch.bfloat16, True
    if precision in ('16', 'fp16', 'float16'):
        return torch.float16, True
    raise ValueError(f"invalid precision: {precision}")


def time_model(
    model: torch.nn.Module,
    inputs: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    *,
    device: str = 'cuda',
    precision: str = 'bf16',
    warmup: int = 5,
    iters: int = 20,
    train: bool = True,
) -> tp.Dict[str, tp.Any]:
    """Time a model on synthetic batches.

    Runs `warmup` untimed steps, then `iters` timed steps of
    forward (+ backward + optimizer step if `train`). Uses CUDA events on GPU
    and falls back to wall-clock timing on CPU.

    Returns a dict of measurements (per-step latency, throughput, peak memory).
    """
    use_cuda = device.startswith('cuda') and torch.cuda.is_available()
    autocast_dtype, use_autocast = _dtype_from_precision(precision)

    model = model.to(device)
    model.train(mode=train)

    inputs_t = torch.from_numpy(inputs.astype(np.int64))
    targets_t = torch.from_numpy(targets.astype(np.int64))
    n = inputs_t.shape[0]
    assert n >= batch_size, f"num_samples ({n}) must be >= batch_size ({batch_size})"

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4) if train else None

    def get_batch(i: int) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        # Cycle through the dataset so we don't always time the exact same batch.
        start = (i * batch_size) % (n - batch_size + 1)
        x = inputs_t[start:start + batch_size].to(device, non_blocking=True)
        y = targets_t[start:start + batch_size].to(device, non_blocking=True)
        return x, y

    def one_step(i: int) -> None:
        x, y = get_batch(i)
        ctx = (
            torch.autocast(device_type='cuda', dtype=autocast_dtype)
            if (use_autocast and use_cuda)
            else contextlib.nullcontext()
        )
        with ctx:
            out = model(x)
            # cast logits to fp32 for a numerically stable, dtype-safe loss
            loss = loss_fn(out.reshape(-1, out.size(-1)).float(), y.reshape(-1))
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    grad_ctx = contextlib.nullcontext() if train else torch.no_grad()
    with grad_ctx:
        for i in range(warmup):
            one_step(i)

        if use_cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()
            for i in range(iters):
                one_step(warmup + i)
            end_evt.record()
            torch.cuda.synchronize()
            elapsed_ms = start_evt.elapsed_time(end_evt)
            peak_mem_mb = torch.cuda.max_memory_allocated() / 1e6
        else:
            t0 = time.perf_counter()
            for i in range(iters):
                one_step(warmup + i)
            elapsed_ms = (time.perf_counter() - t0) * 1e3
            peak_mem_mb = float('nan')

    step_ms = elapsed_ms / iters
    seq_len = inputs_t.shape[1]
    tokens_per_step = batch_size * seq_len
    return {
        'status': 'ok',
        'step_ms': step_ms,
        'tokens_per_s': tokens_per_step / (step_ms / 1e3),
        'samples_per_s': batch_size / (step_ms / 1e3),
        'peak_mem_mb': peak_mem_mb,
        'params': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'batch_size': batch_size,
    }
