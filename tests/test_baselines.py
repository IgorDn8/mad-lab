"""Tests for the model families used in the paper, including the `fla`-backed
baselines (Mamba2, DeltaNet, DeltaProduct).

For one representative registered layer per family (the d=128 / ~1M
iso-parameter tier):

  * the layer builds from the registry and runs forward + backward on a random
    (batch, T, dim) input, in the precision the paper uses for that family
    (bf16 autocast for the fla baselines, fp32 for BD-LRU / H-LRU / PDSSM /
    LSTM); outputs and gradients must be finite;
  * for DeltaProduct, the `hh{r}` tag / `rank` config entry must reach fla's
    layer as `num_householder = r` (fla ignores unknown keyword arguments
    silently, so a misnamed knob would fall back to its default of 2);
  * a short `train()` run on the memorization task and on the group-S task
    completes and reports finite metrics.

Families whose optional dependency is missing (`fla`) are skipped, not failed.
These tests need a CUDA GPU.

Run with:  uv run python -m tests.test_baselines
"""

import os
import shutil

import torch
import yaml

from tests import run_tests
from mad.configs import MADConfig, MADModelConfig
from mad.helpers import compute_vocab_size
from mad.paths import get_base_path
from mad.registry import layer_registry
from train import train

# representative layer per family -> dtype used for that family in the paper
PAPER_LAYERS = {
    'bdlru-sel-wd4-d128-iso1m': torch.float32,
    'hlru-sel-wd4-d128-iso1m': torch.float32,
    'pdssm-d128-iso1m': torch.float32,
    'lstm-d128-iso1m': torch.float32,
    'mamba2-fla-d128-iso1m': torch.bfloat16,
    'dnet-d128-iso1m': torch.bfloat16,
    'dproduct-hh2-d128-iso1m': torch.bfloat16,
    'dproduct-hh4-d128-iso1m': torch.bfloat16,
    'dproduct-hh8-d128-iso1m': torch.bfloat16,
}
FLA_LAYERS = tuple(n for n in PAPER_LAYERS if n.split('-')[0] in ('mamba2', 'dnet', 'dproduct'))
CACHE_PATH = os.path.join('.', '.testing')


def _fla_available() -> bool:
    try:
        import fla  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def _cuda_ready() -> bool:
    if not torch.cuda.is_available():
        print('... CUDA unavailable, skipping')
        return False
    return True


def _build_layer(name: str, dim: int = 128, max_length: int = 128) -> torch.nn.Module:
    reg = layer_registry[name]
    with open(os.path.join(get_base_path(), reg['cfg'])) as f:
        cfg = yaml.safe_load(f)
    cfg['dim'] = dim
    cfg['max_length'] = max_length
    return reg['module'](**cfg)


def _layers_to_test():
    names = list(PAPER_LAYERS)
    if not _fla_available():
        skipped = [n for n in names if n in FLA_LAYERS]
        print(f'  SKIPPED (fla not installed): {skipped}')
        names = [n for n in names if n not in FLA_LAYERS]
    return names


def test_paper_layers_forward_backward():
    """Each family's layer runs forward + backward with finite outputs and gradients."""
    print('Testing forward/backward of one layer per paper family...')
    if not _cuda_ready():
        return
    batch_size, dim, max_length = 2, 128, 128
    for name in _layers_to_test():
        dtype = PAPER_LAYERS[name]
        torch.manual_seed(0)
        layer = _build_layer(name, dim, max_length).cuda()
        x = torch.randn(batch_size, max_length, dim, device='cuda', requires_grad=True)
        with torch.autocast(device_type='cuda', dtype=dtype, enabled=dtype != torch.float32):
            y = layer(x)
        assert y.shape == x.shape, f'{name}: output shape {tuple(y.shape)}'
        assert torch.isfinite(y).all(), f'{name}: non-finite output'
        y.float().square().mean().backward()
        assert torch.isfinite(x.grad).all(), f'{name}: non-finite input gradient'
        missing = [n for n, p in layer.named_parameters() if p.requires_grad and p.grad is None]
        assert not missing, f'{name}: parameters without gradient: {missing[:5]}'
        bad = [n for n, p in layer.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
        assert not bad, f'{name}: non-finite parameter gradients: {bad[:5]}'
        n_params = sum(p.numel() for p in layer.parameters())
        print(f'  {name:28s} params={n_params:>9,}  dtype={str(dtype).split(".")[-1]}  ok')
    print('... passed!')


def test_dproduct_num_householder():
    """`dproduct-hh{r}-*` layers reach fla with num_householder = r."""
    print('Testing DeltaProduct Householder count...')
    if not _cuda_ready():
        return
    if not _fla_available():
        print('  SKIPPED (fla not installed)')
        return
    sizes = {}
    for r in (2, 4, 8):
        name = f'dproduct-hh{r}-d128-iso1m'
        layer = _build_layer(name)
        got = layer.layer.num_householder
        assert got == r, f'{name}: fla layer has num_householder={got}, expected {r}'
        sizes[r] = sum(p.numel() for p in layer.parameters())
        print(f'  {name:28s} num_householder={got}  params={sizes[r]:,}')
    # all three are solved to the same ~1M budget, so they must be within a few % of each other
    lo, hi = min(sizes.values()), max(sizes.values())
    assert hi / lo < 1.1, f'iso-parameter dproduct tiers differ by more than 10%: {sizes}'
    print('... passed!')


def _train_small(layers, mad_config: MADConfig, vocab_size: int, tag: str):
    model = MADModelConfig(layers=layers, vocab_size=vocab_size, max_length=mad_config.seq_len)
    model = model.build_model_from_registry()
    results = train(
        model=model,
        mad_config=mad_config,
        log_path=os.path.join(CACHE_PATH, tag),
        save_checkpoints=False,
    )
    numeric = results.select_dtypes('number')
    assert not numeric.empty and numeric.notna().all().all(), f'{tag}: non-finite metrics\n{results}'
    return results


def test_train_paper_families():
    """A short training run per family on memorization, in the family's precision."""
    print('Testing a short training run per paper family (memorization)...')
    if not _cuda_ready():
        return
    for name in _layers_to_test():
        precision = 'bf16' if PAPER_LAYERS[name] == torch.bfloat16 else '32'
        mad_config = MADConfig(
            task='memorization', vocab_size=16, seq_len=32,
            num_train_examples=16, num_test_examples=8, batch_size=8, epochs=1,
            precision=precision, data_path=os.path.join(CACHE_PATH, 'data'),
        )
        print(f'  {name} (precision={precision})')
        _train_small((name, 'swiglu'), mad_config, vocab_size=16, tag=f'mem-{name}')
    if os.path.isdir(CACHE_PATH):
        shutil.rmtree(CACHE_PATH)
    print('... passed!')


def test_train_group_task():
    """BD-LRU (fp32) and DeltaNet (bf16) train on the group-S state-tracking task."""
    print('Testing a short training run on group-S...')
    if not _cuda_ready():
        return
    cases = [('bdlru-sel-wd4-d128-iso1m', '32')]
    if _fla_available():
        cases.append(('dnet-d128-iso1m', 'bf16'))
    else:
        print('  SKIPPED dnet (fla not installed)')
    for name, precision in cases:
        n = 3  # degree of the symmetric group: S3 has 6 elements
        mad_config = MADConfig(
            task='group-S', vocab_size=n, seq_len=16,
            num_train_examples=32, num_test_examples=8, batch_size=8, epochs=1,
            precision=precision, data_path=os.path.join(CACHE_PATH, 'data'),
        )
        vocab = compute_vocab_size('group-S', n)
        print(f'  {name} on S{n} (|G|={vocab}, precision={precision})')
        _train_small((name, 'swiglu'), mad_config, vocab_size=vocab, tag=f'gs-{name}')
    if os.path.isdir(CACHE_PATH):
        shutil.rmtree(CACHE_PATH)
    print('... passed!')


if __name__ == '__main__':
    run_tests()
