"""Layer registry imports.

Several layers are backed by optional CUDA-only kernels (flash-attn, mamba-ssm,
causal-conv1d, custom CUDA/Triton ops). Those packages live in the optional
``cuda`` extra and can be hard to build on some toolchains. To keep pure-PyTorch
layers (e.g. ``LSTM``) usable without that stack, every layer is imported
defensively here: if a layer's dependencies are missing, its name is still
defined but set to ``None`` (and a single warning is emitted listing what was
skipped). Installing the ``cuda`` extra makes all layers available again.
"""

import importlib
import warnings

# (module path, [exported names]) for every layer this package exposes.
_LAYER_IMPORTS = [
    # channel mixers:
    ("mad.model.layers.mlp", ["Mlp", "SwiGLU", "MoeMlp"]),
    ("mad.model.layers.rwkv", [
        "channel_mixer_rwkv5_wrapped", "channel_mixer_rwkv6_wrapped",
        "time_mixer_rwkv5_wrapped_bf16", "time_mixer_rwkv6_wrapped_bf16",
    ]),
    # sequence mixers:
    ("mad.model.layers.attention", ["Attention"]),
    ("mad.model.layers.attention_linear", ["LinearAttention"]),
    ("mad.model.layers.attention_gated_linear", ["GatedLinearAttention"]),
    ("mad.model.layers.hyena", [
        "HyenaOperator", "MultiHeadHyenaOperator", "HyenaExpertsOperator",
    ]),
    ("mad.model.layers.mamba", ["Mamba"]),
    ("mad.model.layers.rg_lru_attn", ["rgLRUattn"]),
    ("mad.model.layers.kernel_rnn", ["kernelRNN"]),
    ("mad.model.layers.kernel_rnn_o2", ["kernelRNNo2"]),
    ("mad.model.layers.attention_orig", ["AttentionOrig"]),
    ("mad.model.layers.memnet", ["MemNet"]),
    ("mad.model.layers.attention_o1", ["AttentionO1"]),
    ("mad.model.layers.hlru_o0", ["HLRU_o0"]),
    ("mad.model.layers.hlru_sel", ["HLRU_sel"]),
    ("mad.model.layers.hlru_o2", ["HLRU_o2"]),
    ("mad.model.layers.base_lru", ["baseLRU"]),
    ("mad.model.layers.hlru_o1_ag", ["HLRUag_o1"]),
    ("mad.model.layers.hlru_o3_v2", ["HLRUv2_o3"]),
    ("mad.model.layers.hlru_o0_v2", ["HLRU_o0_v2"]),
    ("mad.model.layers.hlru_o1_v2", ["HLRU_o1_v2"]),
    ("mad.model.layers.hlru_o2_v2", ["HLRU_o2_v2"]),
    ("mad.model.layers.minlstm", ["minLSTM"]),
    ("mad.model.layers.hlru_o0_d", ["HLRU_o0_d"]),
    ("mad.model.layers.hlru_o0_dp", ["HLRU_o0_dp"]),
    ("mad.model.layers.mingru", ["minGRU"]),
    ("mad.model.layers.rnn_sq", ["rnnsq"]),
    ("mad.model.layers.hssm_o0", ["HSSM_o0"]),
    ("mad.model.layers.hssm_o1", ["HSSM_o1"]),
    ("mad.model.layers.bdlru_nonsel", ["BDLRU_nonsel"]),
    ("mad.model.layers.bdlru_sel", ["BDLRU_sel"]),
    ("mad.model.layers.conv", ["Conv"]),
    ("mad.model.layers.hlru_o1_mlp", ["HLRU_o1_mlp"]),
    ("mad.model.layers.lstm", ["LSTM"]),
    ("mad.model.layers.hlru_o1_v1", ["HLRU_o1_v1"]),
    ("mad.model.layers.hlru_nonsel", ["HLRU_nonsel"]),
    ("mad.model.layers.deltaproduct", ["dproduct"]),
    ("mad.model.layers.deltanet", ["dnet"]),
    ("mad.model.layers.mambaV2", ["MambaV2"]),
]

_unavailable = {}
for _module_path, _names in _LAYER_IMPORTS:
    try:
        _module = importlib.import_module(_module_path)
        for _name in _names:
            globals()[_name] = getattr(_module, _name)
    except Exception as _exc:  # noqa: BLE001 - report and continue on any import failure
        for _name in _names:
            globals()[_name] = None
        _unavailable[_module_path] = f"{type(_exc).__name__}: {_exc}"

if _unavailable:
    _details = "; ".join(f"{_m} ({_e})" for _m, _e in _unavailable.items())
    warnings.warn(
        "Some MAD layers are unavailable and were set to None (install the "
        f"'cuda' extra to enable them): {_details}",
        stacklevel=2,
    )
