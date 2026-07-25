# $\text{MAD}$-Lab

This repository is a fork of the original MAD-Lab repository [Mechanistic Design and Scaling of Hybrid Architectures](https://arxiv.org/abs/2403.17844). This repository provides code for the paper "Improved State Mixing in Higher-order and Block Diagonal Linear Recurrent Networks".

### What this fork adds (on top of original MAD-Lab)
Everything in this section is new relative to upstream MAD-Lab; the rest of the README documents the original framework.
- **New models:** BD-LRU (`bdlru-sel-*`), H-LRU (`hlru-sel-*`), PDSSM (`pdssm-d128-h*`), Mamba2 (`mamba2-fla-d128`), DeltaNet (`dnet-*`) and DeltaProduct (`dproduct-*`). See [Provided architecture primitives](#provided-architecture-primitives).
- **New synthetic task:** a permutation / state-tracking task on the symmetric group (`group-S`, plus `group-Z` / `group-A`).
- **Multiple scan implementations** for the linear-recurrent layers (`orig`, PyTorch `associative_scan`, custom `hopscan`, and hand-written Triton kernels with analytic backward), selectable at run time via `--layer-overrides implementation=...`.
- **A speed-benchmarking harness** to compare these implementations across sequence length, block size, batch size and hidden size. See [Speed benchmarking](#speed-benchmarking-fork-addition).
- **`uv`-based environment** (`pyproject.toml`) and vendored finite-group utilities for the `group-*` tasks.

## Running the new models
Training runs are launched through [train.py](train.py). Layer names are defined in [mad/registry.py](mad/registry.py) and usually match files in [configs/layers/](configs/layers/) without the `.yml` suffix.

For example, to train the selective BD-LRU variant on the new permutation/state-tracking task:
```bash
python -m train --task group-S --layers bdlru-sel-wd1-d128-h32 swiglu bdlru-sel-wd1-d128-h32 swiglu
```

To train the selective H-LRU variant on the same task:
```bash
python -m train --task group-S --layers hlru-sel-wd1-d128-h32 swiglu hlru-sel-wd1-d128-h32 swiglu
```

Use the `group-S` task for the permutation task based on the symmetric group. You can replace the layer names with other registered variants, such as larger `bdlru-sel-*` or `hlru-sel-*` configs, or with non-selective variants when corresponding configs are registered.

Iso-parameter tiers (e.g. `pdssm-d128-iso1m`, `bdlru-sel-wd4-d128-iso1m`) are generated configs for ~matched parameter budgets and work in `train.py` the same way as manually sized layers. Regenerate them with `uv run python -m scripts.gen_iso_param_sweeps` if needed. Example:

```bash
uv run python -m train --task group-S --vocab-size 5 --seq-len 16 \
  --layers pdssm-d128-iso1m swiglu pdssm-d128-iso1m swiglu \
  --precision 32 --log-base-path logs_gs_pdssm
```

`train.py` infers `--dim` from the `-d<width>-` tag in layer names and sets `max_length` from `--seq-len`. Use `--layer-overrides implementation=associative_scan` to change scan implementations on supported layers.

<div align="center">
<img src="./assets/title-image.png" alt="title" width=750"/>
    
$\rightarrow$ A laboratory to improve and accelerate deep learning architecture protoyping using simple synthetic tasks.
</div>


**Mechanistic Architecture Design (in-short MAD)** represents a simple framework to accelerate the deep learning architecture design process. MAD uses simple synthetic tasks that can be implemented quickly and without much compute to predict how well new candidate architectures will perform at scale in sequence modeling. Each synthetic task is specifically designed to probe skills of a model relevant for sequence modeling, such as compression and recall.

<u>*Why do we care about MAD?*</u> ...because we find that it accurately indicates compute-optimal perplexity of language models at scale:

<div align="center">
<img src="./assets/mad_to_scale.png" alt="mad-to-scale" width="700"/>
</div>

In addition to providing an implementation of the MAD synthetic task suite, this repository provides an implementation of several state-of-the-art layer primitives to allow for easy architecture prototyping, such as attention, Hyena, Mamba, and Gated Linear Attention.

For all details on the MAD synthetic tasks and pipeline, see our recent paper on:
[Mechanistic Design and Scaling of Hybrid Architectures](https://arxiv.org/abs/2403.17844)



## Contents
- [Quickstart](#quickstart)
    - [Setup](#setup)
    - [Training](#training)
    - [Benchmarking](#benchmarking)
        - [Speed benchmarking (fork addition)](#speed-benchmarking-fork-addition)
- [Repository overview](#repository-overview)
- [The MAD synthetic task suite](#the-mad-synthetic-tasks)
- [The MAD protocol](#the-mad-protocol)
- [Provided architecture primitives](#provided-architecture-primitives)
- [How to contribute](#how-to-contribute)
    - [Architectures](#architecture-primitives)
    - [Synthetic Tasks](#synthetic-tasks)
- [Example Uses](#example-uses)


## Quickstart
### Setup
We use [uv](https://docs.astral.sh/uv/) to manage the Python environment and dependencies. If you don't have `uv` yet, install it with:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then create the environment and install the base dependencies (this reads `pyproject.toml` and creates a `.venv`):
```bash
uv sync
```

The CUDA-only kernels ([FlashAttention](https://github.com/Dao-AILab/flash-attention), `causal-conv1d`, `mamba-ssm`) are kept in an optional `cuda` extra because they build against your local CUDA toolkit and require a GPU. To install them as well:
```bash
uv sync --extra cuda
```
To train models on MAD, you will require access to cuda-capable hardware as many of the architecture components provided in this repository are specifically designed to run on cuda GPUs (such as [FlashAttention](https://github.com/Dao-AILab/flash-attention)). Note that most entrypoints (including `train.py` and the data-generation scripts) import the layer registry, so they require the `cuda` extra to be installed.

#### DeltaNet / DeltaProduct baselines (`fla`)
The `dnet-*` (DeltaNet) and `dproduct-*` (DeltaProduct) layers are backed by [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) (`fla`). Install the optional `fla` extra:
```bash
uv pip install "flash-linear-attention>=0.5.1" "tilelang>=0.1.12"
```
These are pure-python / prebuilt wheels and do **not** change your installed `torch`/`triton`. Two important caveats:
- **Precision:** DeltaProduct requires **bf16** (`float32` is unsupported); DeltaNet also runs in bf16 in practice. Run these layers with `--precision bf16`.
- **Hopper GPUs (H100/H200) + Triton ≥ 3.4:** `fla`'s gated chunk backward is only numerically correct via [`tilelang`](https://github.com/tile-ai/tilelang) (see [fla#640](https://github.com/fla-org/flash-linear-attention/issues/640)); without it, `fla` raises at runtime. `tilelang` is therefore part of the `fla` extra above.

#### Mamba2 baseline (why `fla`/Triton, not `mamba_ssm` CUDA)
The `mamba2-fla-d128` layer is backed by `fla.layers.Mamba2` (Triton), installed with the same `fla` extra. We deliberately do **not** depend on the official `mamba_ssm` package, for two reasons:
- **There is no CUDA scan kernel to gain.** Mamba-2's core is the SSD (state-space-duality) chunked scan, which is written in **Triton** in the official `mamba_ssm` repo too — the hand-written CUDA `selective_scan` kernel is Mamba-**1** only. So `mamba_ssm` would not give us a faster scan than the Triton path; it would only add a fused `causal_conv1d` CUDA kernel and a fused gated-RMSNorm. Since the SSD scan dominates runtime (~80–90% at the sequence lengths we benchmark, and both paths run it in Triton), the expected throughput gap is roughly **~10% at long `T`, up to ~30% at short `T`**, concentrated in the depthwise conv rather than the scan.
- **`mamba_ssm` / `causal_conv1d` do not install cleanly here.** They must build CUDA extensions against the local toolkit; on this stack (custom `torch 2.13.0+cu130`, CUDA 13.0) the build does not succeed, and resolving them via `uv` would downgrade `tilelang`/`protobuf` and break the DeltaNet/DeltaProduct baselines. The `fla`/Triton path avoids all of this and is a faithful, competitive stand-in for the Mamba-2 baseline.

#### Compatibility (tested versions)
The layers in this fork (BD-LRU / H-LRU / PDSSM Triton scans, and the `fla` DeltaNet/DeltaProduct baselines) were validated on the following stack:

| Component | Tested version |
|-----------|----------------|
| Python | ≥ 3.11 |
| PyTorch | 2.13.0 (CUDA 13.0 build, `+cu130`) |
| Triton | 3.7.1 |
| GPU / driver | NVIDIA H100 80GB, driver 580.x |
| flash-linear-attention (`fla`) | 0.5.1 |
| tilelang | 0.1.12 |
| transformers | 5.14.1 (pulled in by `fla`) |
| einops | 0.8.2 |
| numpy | 2.5.1 |

Notes:
- The custom-recurrence Triton scans require **Triton ≥ 3.0** and a modern PyTorch (`torch.compile` / `associative_scan` support); they were developed against PyTorch 2.13 + Triton 3.7.
- **Do not** run `uv sync`/`uv sync --extra ...` on a machine that uses a **custom local torch build** (such as a `+cuXXX` wheel not on PyPI): re-resolution can replace your torch. On such machines, add the `fla`/`cuda` packages with `uv pip install <pkg>` (which leaves the existing torch/triton in place), as shown above. `uv sync --extra fla` is only recommended when resolving the whole project fresh from PyPI.

Run any command inside the environment with `uv run`, e.g.:
```bash
uv run python -m train --task group-S --layers bdlru-sel-wd1-d128-h32 swiglu bdlru-sel-wd1-d128-h32 swiglu
```

> **Group tasks (`group-S` / `group-Z` / `group-A`)** rely on the finite-group utilities from [alreich/abstract_algebra](https://github.com/alreich/abstract_algebra) (MIT). Since that project isn't published on PyPI and isn't packaged as an importable `abstract_algebra` module, the required modules are vendored under [`abstract_algebra/`](abstract_algebra/) and their only extra runtime dependency (`sympy`) is included in the base install. No additional setup is needed.


### Training
The [train.py](train.py) script provides an entrypoint for training models on individual MAD synthetic tasks.

#### Using the command-line:
To train models built from architecture primitives provided in this repository, you can run the [train.py](train.py) script with the command line:
```bash
python -m train --task in-context-recall --layers mh-attention swiglu mh-attention swiglu
```
This will train a 4-layer model, composed of multi-head attention, SwiGLU, multi-head attention, SwiGLU in the [in-context recall task](#the-mad-synthetic-tasks). By default, each layer is configured according to our default layer configurations in [configs/layers/](configs/layers/)). For an overview of all provided command-line arguments, see the `get_args()` function in [train.py](train.py).

#### Using the code:
If you want to train architectures using components not provide in this repository, you can do so with the `train(...)` function provided in the [train.py](train.py) script:
```python
from train import train
from mad.configs import MADConfig

# create your PyTorch model:
model = ... 

# configure MAD:
mad_config = MADConfig(task='in-context-recall')
# for an overview of the MAD settings,
# see MADConfig in mad/configs.py

# train your model:
results = train(
    model=model,
    mad_config=mad_config, 
    log_path='./logs/{insert-your-model-name}',
)
```
This will train your model in the [in-context recall task](#the-mad-synthetic-tasks) and return a dataframe with an overview of the final training and evaluation performance of your model. 


### Benchmarking
Beyond training models in individual tasks, this repository provides a means to easily benchmark architectures across the entire [MAD protocol](#the-mad-protocol) with the [benchmark.py](benchmark.py) script.

#### Downloading the data:
To benchmark an architecture, you will first need to download our benchmark data, which is hosted at: [https://zenodo.org/records/10843663](https://zenodo.org/records/10843663).
Please download the data and place it in this directory under the following path: `./benchmark/data` (as shown in our [repository overview](#repository-overview)).

#### Using the command-line:
To benchmark an architecture composed of the architecture primitives provided in this repository, you can run:
```bash
python -m benchmark --layers hyena swiglu hyena swiglu
```
This will run a simple 4-layer model, composed of Hyena, SwiGLU, Hyena, and SwiGLU (configured according to our default layer configurations in [configs/layers/](configs/layers/)), through the MAD protocol. For an overview of all provided command-line arguments, see the `get_args()` function in [benchmark.py](benchmark.py).

#### Using the code:
If you want to benchmark new architectures, using components not provided in this repository, you can do so with the `benchmark(...)` function of the [benchmark.py](benchmark.py) script.

As several task variables are varied throughout the MAD benchmark, such as input sequence length and vocabulary size, you will first need to set up a function that creates the model you want to benchmark based on these task settings. This is an example of such a function:
```python
from torch import nn
from mad.model.model import LanguageModel, AutoEncoder
from mad.model.layers import hyena, swiglu

def make_model_fn(
    task: str,
    vocab_size: int,
    max_length: int,
) -> nn.Module:

    # a list of the layer modules composing your architecture:
    layers = [hyena, swiglu, hyena, swiglu]
    # these need to be torch.nn.Module instances!
    
    # setup a config for each of your layer types:
    hyena_config = {'dim': 128, 'max_length': max_length}
    swiglu_config = {'dim': 128, 'max_length': max_length}
    # (128 is the default model width of the MAD protocol)
    # and concatenate them into a list, with one entry for each layer:
    layer_configs = [hyena_config, swiglu_config, hyena_config, swiglu_config]
    
    # select the correct model backbone for your layers:
    backbone = LanguageModel if task not in {'compression'} else AutoEncoder
    # we recommend that you use these 2 backbones to make your results
    # comparable to the results provided in our paper
    
    return backbone(
        vocab_size=vocab_size,
        max_length=max_length,
        layers=layers,
        layer_cfgs=layer_configs
    )
```
As you can see, the function should accept three keyword arguments:
- `task` (str): the name for the MAD synthetic task. Here, we only use this variable to select an appropriate model backbone for your layers ... but who knows maybe there are other things in your architecture depending on it as well?
- `vocab_size` (int): the size of the vocabulary used in the current task setting.
- `max_length` (int): the maximum sequence length encountered in the current task setting.

Once you have created a function that creates your model given these arguments, you can go ahead and run it through the MAD protocol:
```python
mad_scores = benchmark(make_model_fn=make_model_fn, model_id='{insert-your-model-name}')
```
This will return a pandas series (`mad_scores`) with a MAD score for your model in each synthetic task of the MAD protocol.

### Speed benchmarking (fork addition)
In addition to the accuracy-oriented MAD protocol above, this fork adds a **throughput / latency / memory** harness for comparing sequence-mixer implementations. It measures forward+backward step time (median + IQR over repeats), tokens/s, samples/s and peak memory, sweeping over sequence length, block size, batch size or hidden size, with automatic OOM back-off, CSV output under `results/<name>/` and figures under `figures/<name>/`.

Single sweep (one layer, one implementation):
```bash
uv run python speed_benchmark.py \
  --layers bdlru-sel-wd1-d128-h128 \
  --layer-overrides implementation=triton_persistent \
  --sweep-key seq_len --sweep-values 512 1024 2048 4096 8192 16384 \
  --precision 32 --name bdlru-triton-persistent
```
Full multi-regime comparison (sequence length, block size, batch size, hidden size, inference, cross-model), resumable — reruns only fill in missing cells:
```bash
uv run python -m scripts.run_comparison_suite --regime all
```

#### Models available for speed benchmarking
Any registered layer works with `speed_benchmark.py`, including all original MAD primitives (`attention`, `hyena`, `mamba`, `gated-linear-attention`, `rwkv{5,6}`, …). The table below lists the models this fork focuses on. "Scan impls" are passed via `--layer-overrides implementation=<name>`.

| Model (layer name) | Origin | Scan implementations | Precision | Extra deps |
|--------------------|--------|----------------------|-----------|------------|
| `bdlru-sel-wd{1,2,4,8,16,32,64}-d128-h*` | fork | `orig`, `affine_scan_torch_impl`, `hopscan_custom`, `custom_hopscan_autotune`, `triton_sequential`, `triton_persistent`, `triton_parallel_blelloch`, `triton_chunked`, `triton_auto` | `32` | — |
| `hlru-sel-*` | fork | same set as BD-LRU (higher-order; still being refined) | `32` | — |
| `pdssm-d128-h{64,128,256}` | fork | `sequential`, `associative_scan` | `32` (complex) | — |
| `mamba2-fla-d128` | fork | `fla` Triton SSD (see [note](#mamba2-baseline-why-flatriton-not-mamba_ssm-cuda)) | `bf16` | `fla` |
| `dnet-*` (DeltaNet) | fork | `fla` chunked | `bf16` | `fla` |
| `dproduct-*` (DeltaProduct) | fork | `fla` chunked | `bf16` | `fla` |
| `lstm-d128-h*` | fork | cuDNN LSTM baseline | `bf16`/`32` | — |
| `mamba`, `attention`, `hyena`, `gated-linear-attention`, `rwkv{5,6}`, … | original | n/a | `bf16` | some need `cuda` extra |

Notes:
- **Block size** for BD-LRU/H-LRU is the `wd*` in the layer name, or set at run time with `--layer-overrides window_dim=<m>` on a `wd1` layer (used by the suite to sweep blocks without new configs).
- `orig` is the reference implementation and is the slowest at long `T`; the suite caps it at moderate `T` for tractability.
- **`triton_auto` is the recommended Triton variant.** `triton_sequential` / `triton_persistent` / `triton_parallel_blelloch` all run one program per row, so their only parallelism is `batch * hidden_dim` and they leave the GPU idle at small batch (as low as 18 GB/s of ~3350 available on an H100). `triton_auto` picks per shape between those and `triton_chunked` (see [`triton_auto_scan.py`](mad/model/layers/ops/scans/triton_auto_scan.py)). It matches the best fixed kernel everywhere measured and beats it by up to **3.3x on full layer fwd+bwd**, and by these factors on the scan kernel alone at `T=8192`:

  | `m` | 1 | 2 | 4 | 8 | 16 |
  |---|---|---|---|---|---|
  | speedup vs. best fixed kernel | 8.8–11.6x | 8.9–21.0x | 3.2–11.3x | 1.0–1.5x | 1.0–2.9x |

  Narrow blocks (`m ≤ 4`) get a log-depth `tl.associative_scan` over contiguous time slabs ([`triton_slab_scan.py`](mad/model/layers/ops/scans/triton_slab_scan.py)); `m=2` reaches 2455 GB/s (73% of peak) where the fixed kernels manage 277. `m=8` is the documented cutoff for that technique — it spills registers and falls back.
- Precision: the `m ≤ 4` path is exact to ~2e-7 vs. the `orig` reference. For wider blocks `triton_chunked` reassociates the recurrence into fp32 matrix products, like `triton_parallel_blelloch`, so both carry the same looser tolerance (see `tests/test_scans.py`); this only affects `m ≥ 8` at low `batch * hidden_dim`, where `triton_auto` selects the chunked path.
- Correctness/precision of the scan implementations (fwd + bwd error vs. a high-precision reference, in fp32 and bf16) is checked separately by [`scripts/verify_scans.py`](scripts/verify_scans.py), not by the speed harness.

#### Iso-parameter benchmark sets (systematic, all families)
For fair cross-architecture comparison, every family is normalized to the same **sequence-mixer parameter budget** at a fixed model width `d`. Token embed/unembed params are identical for a fixed `(vocab, d)`, so only the mixer differs — this is what we equalize. Every family's size knob is *solved* to hit the budget (via binary search over the knob, verified by direct `parameters()` count), so the whole set is generated from one script and new tiers/families are one edit away:
```bash
uv run python -m scripts.gen_iso_param_sweeps          # solve every knob, write configs + index
uv run python -m scripts.gen_iso_param_sweeps --check   # solve + print (params, %err) only
```
This writes one YAML per config under `configs/layers/` and an index `configs/iso_param_sweeps.json` that `mad/registry.py` reads to auto-register them.

**Tiers** (tag → width & budget):

| Tag | `d` | Budget | Notes |
|-----|-----|--------|-------|
| `iso033m` | 128 | ~0.33M | smallest |
| `iso1m` | 128 | ~1.0M | ~3× larger, same width |
| `iso1m` | 1024 | ~1.0M | wide, shallow state (Mamba2 excluded — floor ~3.4M) |
| `iso10m` | 1024 | ~10M | Mamba2 fits (`expand≈3`, near-canonical) |
| `iso100m` | 1024 | ~100M | largest |

**Naming** (uniform across all tiers; `{tag}` from the table, `{d}` = width):

| Family | Layer name(s) | Solved knob |
|--------|---------------|-------------|
| LSTM | `lstm-d{d}-{tag}` | `hidden_dim` |
| BD-LRU (blocks 1,2,4,8,16) | `bdlru-sel-wd{m}-d{d}-{tag}` | `hidden_dim` per block |
| H-LRU (blocks 1,2,4,8,16) | `hlru-sel-wd{m}-d{d}-{tag}` | `hidden_dim` per block |
| PDSSM | `pdssm-d{d}-{tag}` | `hidden_dim` |
| Mamba2 | `mamba2-fla-d{d}-{tag}` | `expand` (state_size=128, head_dim=64) |
| DeltaNet | `dnet-d{d}-{tag}` | `num_heads` (head_dim=16) |
| **DeltaProduct (2/4/8 Householders)** | `dproduct-hh{2,4,8}-d{d}-{tag}` | `num_heads` per rank (head_dim=16) |

So e.g. the `d=1024`/~100M tier is `lstm-d1024-iso100m`, `bdlru-sel-wd{1,2,4,8,16}-d1024-iso100m`, `hlru-sel-wd{1,2,4,8,16}-d1024-iso100m`, `pdssm-d1024-iso100m`, `mamba2-fla-d1024-iso100m`, `dnet-d1024-iso100m`, and `dproduct-hh{2,4,8}-d1024-iso100m` (17 configs). Measured budgets are within ±2% except a few tight corners (`bdlru-sel-wd16-d128-iso033m` −5%, `bdlru-sel-wd16-d1024-iso1m` −6.5%, DeltaProduct at `d=1024`/~1M within −7%) where integer knob steps are coarse relative to the budget.

**DeltaProduct at 2/4/8 Householders in every tier.** The number of Householder reflections (`rank`) is DeltaProduct's analog of BD-LRU/H-LRU block size, so it is a first-class comparison axis: each tier ships three independent DeltaProduct models (`rank ∈ {2,4,8}`), each parameter-matched to the budget. Rank changes parameters (and compute) but *not* recurrent state, so at fixed budget the head width shrinks slightly as rank grows.

Notes / caveats:
- **Precision differs by family** (BD-LRU / H-LRU / PDSSM / LSTM run at `--precision 32`; Mamba2 / DeltaNet / DeltaProduct at `--precision bf16`), so iso-parameter does *not* imply iso-precision — keep this in mind when reading throughput.
- **Non-linear knobs:** LSTM/PDSSM params grow ~quadratically in `hidden_dim` and BD-LRU/H-LRU ~quadratically in block size `m` (`Linear(d, N·m·(m+1))` gate), so the solved sizes are far from a naive linear scaling of the budget.
- **Mamba2 at `d=128`** needs an atypically large `expand` (≈6 at ~0.33M, ≈20 at ~1.0M) because a canonical `expand=2` is only ~0.13M there; those rows are parameter-matched, not architecturally canonical. At `d=1024`/~1M Mamba2 is dropped entirely (its `expand=1` floor is ~3.4M).
- The layer name and YAML both carry `d`, so run `d=1024` configs with `--dim 1024` (the speed harness and `param_count` honor it).

### Iso-hidden-state benchmark sets
Complementary to the iso-*parameter* sets above, these tiers match the **recurrent state size** — the total number of scalars carried across time (the model's fixed memory) — rather than the parameter count. This is the axis that governs recall/associative capacity.

State size per family (the quantity we hold fixed):

| Family | Recurrent state size | Knob to hit state `S` |
|--------|----------------------|-----------------------|
| LSTM | `hidden_dim` (cell state) | `hidden_dim = S` |
| BD-LRU / H-LRU | `hidden_dim × window_dim` (block-diagonal, **N·m**) | `window_dim = m`, `hidden_dim = S/m` |
| PDSSM | `hidden_dim` (complex diagonal modes) | `hidden_dim = S` |
| Mamba2 | `d_inner × state_size = (expand·d)·state_size` | `expand=2, head_dim=64, state_size = S/(2d)` |
| DeltaNet | `num_heads × head_dim²` (`expand_v=1`) | `head_dim=16, num_heads = S/256` |
| DeltaProduct (2/4/8 Householders) | `num_heads × head_dim²` (`expand_v=1`; **independent of #householders/rank**) | `head_dim=16, num_heads = S/256`, `rank ∈ {2,4,8}` |

Tiers generated: **d=128** at state ∈ {512, 1024, 2048, 4096}, and **d=1024** at state 4096 only. BD-LRU and H-LRU are emitted at block sizes `m ∈ {1, 2, 4, 8, 16}` (with `hidden_dim = S/m`) so block size can be compared at matched state. DeltaProduct is emitted at Householders `rank ∈ {2, 4, 8}` (`dproduct-hh{r}-d{d}-s{S}`); because rank does not change the state, all three sit at the same state `S` — the direct #householders-vs-block-size comparison against BD-LRU/H-LRU. Layer names follow `…-d{d}-s{state}`, e.g. `bdlru-sel-wd4-d128-s2048` (block 4, N=512), `dnet-d128-s1024`, `dproduct-hh4-d128-s2048`, `mamba2-fla-d1024-s4096`.

These configs are generated from a single spec (`mad.registry.iso_state_layer_specs`) — regenerate/verify with:
```bash
uv run python -m scripts.gen_iso_state_configs          # write the 85 YAMLs
uv run python -m scripts.gen_iso_state_configs --check   # print implied state sizes only
```

Important caveats:
- **Iso-state does *not* control parameters** — parameter counts vary by orders of magnitude across families at the same state. Because LSTM/PDSSM params grow ~quadratically in hidden size, they become very large at high state (e.g. `pdssm-d128-s4096` ≈ 205M params, `lstm-d128-s4096` ≈ 70M) and may OOM; the speed harness will back off batch size.
- At **low** state, the matrix-state baselines become degenerate: `mamba2-*-s512` uses `state_size=2` and `dnet-*-s512` uses `num_heads=2` — parameter-faithful for state matching but not architecturally canonical. Interpret the small-state Mamba2/Delta rows with care.
- Precision caveat unchanged (fp32 for LRU/PDSSM, bf16 for Mamba2/DeltaNet/DeltaProduct).


## Repository Overview
```
/benchmark
┗ 📂 data
┃    ┗ 📂 t-* -> one directory for each benchmark task setting
┃    ┃    ┗ 📂 test
┃    ┃    ┃    ┗ inputs.npy 
┃    ┃    ┃    ┗ targets.npy 
┃    ┃    ┗ 📂 train
┃    ┃         ┗ inputs.npy 
┃    ┃         ┗ targets.npy 
┃    ┗ 📂 t-*
┃    ┗ ...

/configs -> default layer and task settings
┣ 📂 layers
┃    ┗ *.yml
┣ 📂 tasks
┃    ┗ *.yml

/mad
┣ 📂 data
┃   ┗ __init__.py
┃   ┗ dataset.py -> creating datasets given a function generates instances of a task
┃   ┗ instances.py -> generating instances of each task
┣ 📂 model
┃    ┣ 📂 layers
┃    ┃    ┗ 📂 featurization -> everything used to featurize layers
┃    ┃    ┗ 📂 ops
┃    ┃    ┗ 📂 rwkv
┃    ┃    ┗ __init__.py
┃    ┃    ┗ attention_gated_linear.py 
┃    ┃    ┗ attention_linear.py
┃    ┃    ┗ attention.py
┃    ┃    ┗ hyena.py
┃    ┃    ┗ mamba.py
┃    ┃    ┗ mlp.py
┃    ┃
┃    ┣ __init__.py
┃    ┣ auto_encoder.py
┃    ┣ language_model.py
┃    ┗ pl_model_wrapper.py
┃
┣ __init__.py
┣ analysis.py -> tools to analyze benchmark results
┣ configs.py -> dataclasses we use to configure MAD
┣ metrics.py -> metrics for training / evaluation
┣ paths.py -> some tools to make and parse paths
┗ registry.py -> registry for all layers and tasks of this repository

/abstract_algebra -> vendored finite-group utilities (MIT) used by the group-* tasks

benchmark.py -> benchmarking models on MAD
train.py -> training a model on individual tasks
.gitignore.py
README.md
pyproject.toml -> uv project definition (dependencies + optional cuda extra)
requirements.txt -> legacy dependency list (kept in sync with pyproject.toml)
```



## The MAD synthetic tasks
MAD spans six simple token manipulation tasks. We provide a brief overview of each task in the following. For more details, please see our paper.

### `in-context-recall`
<img src="./assets/recall.png" alt="recall" width="300"/>
To answer a prompt well, language models must be able to understand and learn from new information presented in the prompt (so-called in-context learning). A wealth of empirical work has demonstrated that the associative recall task is well-suited to test the in-context learning skill. MAD uses a multi-query variant of this task: Given an input sequence of key-value pairs, models are tasked with retrieving all values from the input sequence associated with keys that were already shown in the input sequence. Note that while the mapping from keys to values is consistent within an input sequence, it is randomly shuffled between sequences.

### `fuzzy-in-context-recall`
<img src="./assets/fuzzy_recall.png" alt="fuzzy_recall" width="300"/>
In language, semantic units are often spread out over multiple adjacent tokens (e.g., "blue sky" vs "gray sky"). To test how capable a model is of semantically grouping together adjacent tokens, MAD utilizes a variant of in-context recall, in which keys and values are composed of a variable number of adjacent tokens. Specifically, for each sequence, variable length keys and values are randomly drawn from the vocabulary and then assigned into pairs. Since the structure of key/value lengths in a sequence, as well as the mapping from keys to values, change between sequences, fuzzy recall can be treated as a more challenging variant of in-context recall.

### `noisy-in-context-recall`
<img src="./assets/noisy_recall.png" alt="noisy_recall" width="300"/>
To answer a prompt well, language models must be able to ignore irrelevant information of the input. To test this skill, MAD uses another adaptation of in-context recall, in which irrelevant information, represented by tokens from a distinct vocabulary, is added in an arbitrary and variable pattern in between the key-value pairs.
Note that this adds a memorization aspect to the task, as models need to learn during training to ignore tokens from the noise vocabulary.

### `selective-copying`
<img src="./assets/selective_copy.png" alt="selective_copy" width="300"/>
In addition to ignoring irrelevant information of an input, language models must be able to selectively remember relevant information of an input. To test this skill, MAD uses a selective copying task, in which models are tasked with copying tokens from one position of an input sequence to a later position of the sequence, while ignoring irrelevant noise tokens that are randomly inserted into the sequence. Importantly, tokens are always copied in their order of occurrence. Models thereby need to not just remember the tokens that are to be copied but also their specific order of occurrence in the sequence.

### `compression`
<img src="./assets/compression.png" alt="compression" width="300"/>
Recent findings in the mechanistic interpretability literature indicate that a key skill of language models is "token concatenation", where early attention layers assemble information that is spread across multiple tokens in an input onto another token so that the assembled information can then be decoded well by subsequent MLPs. To test the ability of a model to perform token concatenation, even without attention and MLP, MAD utilizes a compression task. In this task, models are trained to compress a random sequence of input tokens into a single aggregation token so that the original input sequence can be fully recovered from the aggregation token by a subsequent MLP.

### `memorization`
<img src="./assets/memorization.png" alt="memorization" width="300"/>
In addition to manipulating and retrieving information from an input sequence, language modeling requires the memorization of factual knowledge. To test this skill, MAD utilizes a memorization task, in which models are tasked with learning a fixed key-value mapping (resembling facts in language) from the training data. Unlike recall, the mapping requires no in-context computation as the ground-truth mapping is constant across samples. 



## The MAD Protocol
MAD follows a two-step procedure, starting from the design of a new candidate architecture, followed by its systematic evaluation according to the following key principles: 

1. For each synthetic task, a MAD score is obtained by averaging architecture performances across a range of task difficulty levels. To manipulate difficulty, MAD independently varies a set of relevant experimental variables: length of the input sequence, size of the vocabulary, and size of the training set. Some tasks have additional variables such as the ratio of noise tokens in the noisy recall and selective copying tasks. For an overview of the changes applied to each task, see the changes entry in each task config in [configs/tasks/](configs/tasks/).

2. Fixed-state architectures need to be normalized to an iso-state and iso-parameter setting, including models featuring sparsely activated layers such as Mixture-of-Experts. For details on this, please see our paper!

3. To ensure that model performance estimates are not dependent on a specific optimization setting, MAD sweeps each architecture in each task setting over a 3 x 2 grid of learning rate and weight decay values (learning rates: $0.0001, 0.0005, 0.001$, weight decays: $0., 0.1$). MAD scores are based on the best runs from this sweep.

4. Model performances are always evaluated in an independent evaluation dataset, specific to each task setting.



## Provided architecture primitives
For an overview of all provided layer types, see the [mad/model/layers/](mad/model/layers/) directory as well as our layer registry in [mad/registry.py](mad/registry.py). For an overview of the default layer configurations, see their respective configurations in [configs/layers/](configs/layers/).

### Channel-mixing:
- `mlp`: Standard expanding multi-layer perceptron, [implementation](mad/model/layers/mlp.py)
- `swiglu`: Swish-Gated-Linear Unit, [implementation](mad/model/layers/mlp.py)
- `moe-mlp`: A Mixture-of-Experts MLP variant, [implementation](mad/model/layers/mlp.py)

### Sequence-mixing:
- `attention`: [paper](https://arxiv.org/abs/2307.08691), [implementation](mad/model/layers/attention.py)
    - `sliding-attention`: implements a sliding-window variant of attention
- `hyena`: [paper](https://arxiv.org/abs/2302.10866), [implementation](mad/model/layers/hyena.py)
    - `hyena-experts`: as proposed in the MAD paper, [implementation](mad/model/layers/hyena.py)
- `mamba`: [paper](https://arxiv.org/abs/2312.00752), [implementation](mad/model/layers/mamba.py)
- `linear-attention`: [paper](https://arxiv.org/abs/2006.16236), [implementation](mad/model/layers/attention_linear.py)
- `gated-linear-attention`: [paper](https://arxiv.org/abs/2312.06635), [implementation](mad/model/layers/attention_gated_linear.py)
- `rwkv{5,6}`: [paper](https://arxiv.org/abs/2305.13048), [implementation](mad/model/layers/rwkv)

Note that we also provide respective multi-head variants for layers supporting this type of state expansion, which you can call by simply adding `mh-` (for multi-head) to the layer's name, eg: `mh-hyena`. 

This fork additionally provides (see [mad/registry.py](mad/registry.py) for all registered configs):
- `bdlru-sel-*` / `hlru-sel-*`: block-diagonal / higher-order LRU variants (multiple scan implementations: `orig`, `affine_scan_torch_impl`, `hopscan_custom`, `custom_hopscan_autotune`, and Triton `triton_sequential` / `triton_persistent` / `triton_parallel_blelloch` / `triton_chunked` / `triton_auto`, selected via `--layer-overrides implementation=...`). Run at `--precision 32`.
- `pdssm-d128-h*`: Permutation-Dictionary SSM ([implementation](mad/model/layers/pdssm.py)) with `sequential` and `associative_scan` recurrences (`--layer-overrides implementation=...`). Complex-valued, so run at `--precision 32`.
- `dnet-*` (DeltaNet) / `dproduct-*` (DeltaProduct): `fla`-backed baselines; require the `fla` extra and `--precision bf16` (see [Setup](#setup)).



## How to contribute
We are hoping that MAD will grow as an open source tool for easy and compute-efficient architecture prototyping. We therefore welcome contributions of new architecture primitives and synthetic tasks.

### Architecture primitives:
To contribute a new architecture primitive, please proceed as follows:
0. Fork this repository.
1. Add an implementation of your layer primitive to [mad/model/layers](mad/model/layers/); Ideally, your implementation is self-contained in one script. For featurizations and ops used by your layer, we also provide the [ops](mad/model/layers/ops/) and [featurization](mad/model/layers/featurization/) directories.
2. Add a default configuration for your layer to [configs/layers](configs/layers/). Please make sure that your layer configuration is normalized to the iso-state and iso-parameter setting we use for MAD (see our paper for details on this).
3. Add an import for your layer to [mad/model/layers/__init__.py](mad/model/layers/__init__.py) and create an entry for your layer in our layer registry in [mad/registry.py](mad/registry.py).
4. Verify that your layer is fully compatible with our [train.py](train.py) and [benchmark.py](benchmark.py) scripts.
5. Once you have completed these steps, create a [pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests), which we will then review.

### Synthetic tasks:
To contribute a new synthetic task, please proceed as follows:
0. Fork this repository.
1. Write a function that creates a single instance of the synthetic task you would like to add and add it to [mad/data/instances.py](mad/data/instances.py). This function needs to return two numpy arrays: inputs and targets. Take a look at our current implementations, as well as the MADConfig in [mad/configs.py](mad/configs.py), to get an overview of all relevant keyword arguments.
2. Add a configuration for your task to [configs/tasks](configs/tasks/), making sure that it contains all relevant arguments for the baseline setting, as well as all relevant entries to manipulate task difficulty (indicated by "changes"). Use the existing task configurations as a guideline.
3. Make sure to add an entry for your to the layer registry in [mad/registry.py](mad/registry.py).
4. Verify that the models implemented in this repository can be trained on your task.
5. Once you have completed these steps, create a [pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests), which we will then review.



## Example Uses
We collect scripts showing basic usecases for MAD in [scripts/](scripts/).

### Architecture Improvement:
[scripts/architecture_improvement.py](scripts/architecture_improvement.py) shows an example demonstrating how to compare the performance of multiple variants of a base architecture.


## Thank you! :pray:
All of this work would not be possible without the many amazing open source implementations that we are building on, such as [FlashAttention](https://github.com/Dao-AILab/flash-attention), [flash attention linear](https://github.com/sustcsonglin/flash-linear-attention), [rwkv](https://github.com/BlinkDL/RWKV-LM/tree/main), [mamba](https://github.com/state-spaces/mamba), and [hyena](https://github.com/HazyResearch/safari).
