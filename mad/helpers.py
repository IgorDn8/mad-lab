from abstract_algebra.finite_algebras import (
    FiniteAlgebra,
    generate_cyclic_group,
    generate_symmetric_group,
)
from collections import OrderedDict
import re

_LAYER_DIM_RE = re.compile(r'-d(\d+)-')


def infer_dim_from_layer_names(layer_names: list[str]) -> int | None:
    """Return the unique model width encoded in layer names (e.g. ``d128``), if any."""
    dims = set()
    for name in layer_names:
        match = _LAYER_DIM_RE.search(name)
        if match:
            dims.add(int(match.group(1)))
    if len(dims) > 1:
        raise ValueError(
            f'Layers imply conflicting widths {sorted(dims)}: {layer_names}. '
            'Use layers that share the same `-d<width>-` tag or pass a consistent --dim.'
        )
    return next(iter(dims)) if dims else None


def _parse_override_value(value: str):
    if value.lower() in {'true', 'false'}:
        return value.lower() == 'true'
    try:
        if '.' in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def parse_layer_overrides(items: list[str]) -> dict:
    """Parse ``KEY=VALUE`` strings (same format as speed_benchmark.py)."""
    overrides = {}
    for item in items:
        if '=' not in item:
            raise ValueError(f'--layer-overrides expects KEY=VALUE, got {item!r}')
        key, value = item.split('=', 1)
        overrides[key] = _parse_override_value(value)
    return overrides

def generate_group(g: (str, int)) -> FiniteAlgebra:
    """Generate an group from a string identifier."""
    if g[0] == "S":
        return generate_symmetric_group(g[1])
    elif g[0] == "Z":
        return generate_cyclic_group(g[1])
    elif g[0] == "A":
        s_n = generate_symmetric_group(g[1])
        a_n = s_n.commutator_subalgebra()
        a_n.name = f"A{g[1]}"
        return a_n
    else:
        raise ValueError("Group must be one of S, Z, or A")

def compute_vocab_size(task: str,
                       vocab_size: int):
    """Helper function to compute vocabuary size for group task."""
    group_prod = generate_group((task[-1],vocab_size))

    num_elements = len(group_prod.elements)
    
    return num_elements 

def clean_state_dict(state_dict, prefix_to_strip="model."):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith(prefix_to_strip):
            k = k[len(prefix_to_strip):]
        new_state_dict[k] = v
    return new_state_dict

def substitute_field(filename: str, key: str, new_value: int) -> str:
    """
    Substitute the number after `key-` with a new number in the filename.

    Args:
        filename (str): Original filename string.
        key (str): The key whose value to replace (e.g., 'nvs').
        new_value (int): The new value to substitute.

    Returns:
        str: Updated filename string.
    """
    pattern = rf"({key})(\d+)"
    return re.sub(pattern, rf"\1{new_value}", filename)


import time
import pytorch_lightning as pl

class EpochTimeTracker(pl.Callback):
    """
    A callback to measure and store the duration of a specific training epoch.
    """
    def __init__(self, epoch_index: int = 2):
        super().__init__()
        self.epoch_index = epoch_index
        self.start_time = None
        self.epoch_duration = None  # <-- ADD THIS: Attribute to store the result

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.current_epoch == self.epoch_index:
            self.start_time = time.perf_counter()

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.current_epoch == self.epoch_index and self.start_time is not None:
            duration = time.perf_counter() - self.start_time
            self.epoch_duration = duration  # <-- ADD THIS: Store the calculated duration
            print(f"--- ⏱️ Epoch {self.epoch_index + 1} training time: {self.epoch_duration:.2f} seconds ---")

class AllEpochsTimeTracker(pl.Callback):
    """A callback to measure and store the duration of ALL training epochs."""
    def __init__(self):
        super().__init__()
        self.epoch_times = {}  # Use a dictionary to store times
        self._start_time = None

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Record the start time of a training epoch."""
        self._start_time = time.perf_counter()

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Calculate and store the epoch duration."""
        if self._start_time is None:
            return
        
        duration = time.perf_counter() - self._start_time
        epoch_num = trainer.current_epoch
        self.epoch_times[epoch_num] = duration
        print(f"--- ⏱️ Epoch {epoch_num + 1} training time: {duration:.2f} seconds ---")
