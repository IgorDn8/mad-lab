import os
import shutil
import torch
import yaml
from train import train
from mad.configs import MADConfig, MADModelConfig
from mad.registry import layer_registry
from tests import run_tests
import fnmatch


def _selected_layers():
    """Registered layers to test. Set MAD_TEST_LAYERS to a comma-separated list of
    glob patterns (e.g. '*-d128-iso1m,swiglu') to restrict the (large) registry;
    layers whose optional dependencies are missing (module is None) are skipped."""
    patterns = [p for p in os.environ.get('MAD_TEST_LAYERS', '*').split(',') if p]
    selected, skipped = [], []
    for name, reg in layer_registry.items():
        if not any(fnmatch.fnmatch(name, p) for p in patterns):
            continue
        (selected if reg['module'] is not None else skipped).append(name)
    if skipped:
        print(f'...skipping {len(skipped)} layers whose dependencies are missing: {skipped[:8]}{" ..." if len(skipped) > 8 else ""}')
    return selected


def test_layer_forward_pass():
    print('Testing forward pass for each registered layer type...')
    batch_size = 2
    dim = 128
    max_length = 128
    failures = {}
    for layer_name in _selected_layers():
        layer_reg = layer_registry[layer_name]
        print(f'...now testing: {layer_name}')
        try:
            with open(layer_reg['cfg'], 'r') as f:
                layer_cfg = yaml.safe_load(f)
            layer_cfg['dim'] = dim
            layer_cfg['max_length'] = max_length
            layer = layer_reg['module'](**layer_cfg)
            inputs = torch.randn(batch_size, max_length, dim)
            # move to GPU if available
            if torch.cuda.is_available():
                fp32 = any(k in layer_name for k in ('hyena', 'pdssm'))  # pdssm is complex-valued
                inputs = inputs.to(torch.float if fp32 else torch.bfloat16).cuda()
                layer = layer.to(torch.float if fp32 else torch.bfloat16).cuda()
            outputs = layer(inputs)
            assert outputs.shape == inputs.shape, f'output shape {tuple(outputs.shape)}'
            assert torch.isnan(outputs).sum() == 0, 'NaN in output'
        except Exception as exc:  # noqa: BLE001 - collect every failing layer, report at the end
            failures[layer_name] = f'{type(exc).__name__}: {exc}'
            print(f'   FAILED: {failures[layer_name]}')
    assert not failures, f'{len(failures)} layer(s) failed the forward pass:\n' + '\n'.join(
        f'  {k}: {v}' for k, v in failures.items())
    print('... passed!')


# families run in fp32 in the paper (PDSSM is complex-valued and cannot run under
# bf16 autocast; BD-LRU / H-LRU / LSTM are trained in fp32); everything else bf16
_FP32_FAMILIES = ('pdssm', 'bdlru', 'hlru', 'lstm')


def _precision_for(layer_name: str) -> str:
    return '32' if layer_name.split('-')[0] in _FP32_FAMILIES else 'bf16'


def test_train_layers():
    print('Testing training for each registered layer type...')
    cache_path = os.path.join('.', '.testing')
    failures = {}
    for layer in _selected_layers():
        precision = _precision_for(layer)
        print(f'...now testing: {layer} (precision={precision})')
        mad_config = MADConfig(
            task='memorization',
            vocab_size=16,
            seq_len=64,
            num_train_examples=10,
            num_test_examples=10,
            batch_size=2,
            epochs=1,
            precision=precision,
            data_path=os.path.join(cache_path, 'data'),
        )
        try:
            model_config = MADModelConfig(layers=(layer,))
            model = model_config.build_model_from_registry()
            train(
                model=model,
                mad_config=mad_config,
                log_path=os.path.join(cache_path, layer),
                save_checkpoints=False,
            )
        except Exception as exc:  # noqa: BLE001 - collect every failing layer, report at the end
            failures[layer] = f'{type(exc).__name__}: {exc}'
            print(f'   FAILED: {failures[layer]}')
    if os.path.exists(cache_path):
        shutil.rmtree(cache_path)
    assert not failures, f'{len(failures)} layer(s) failed to train:\n' + '\n'.join(
        f'  {k}: {v}' for k, v in failures.items())
    print('... passed!')


if __name__ == '__main__':
    run_tests()
