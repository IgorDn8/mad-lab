"""Dataset configurations of the paper's synthetic experiments (``configs/tasks/*.yml``).

Each task file holds a ``baseline`` mapping of ``MADConfig`` task settings and either

* ``changes``: a mapping ``setting -> [values]``; every value is applied on its own to
  the baseline (the MAD protocol), so ``baseline`` plus ``n`` listed values give
  ``n + 1`` datasets; or
* ``configs``: an explicit list of mappings, each merged over the baseline.

``load_task_configs`` expands a file into the list of settings dictionaries, which
``scripts/generate_synth_data.py`` turns into datasets and which can be passed to
``MADConfig(**spec)``.
"""

from __future__ import annotations

import os

import yaml

from mad.paths import get_base_path
from mad.registry import task_registry


def task_config_path(task: str) -> str:
    return os.path.join(get_base_path(), 'configs', 'tasks', f'{task}.yml')


def load_task_configs(task: str) -> list[dict]:
    """Expand ``configs/tasks/<task>.yml`` into the list of dataset settings."""
    if task not in task_registry:
        raise KeyError(f'unknown task {task!r}; registered: {list(task_registry)}')
    path = task_config_path(task)
    if not os.path.exists(path):
        raise FileNotFoundError(f'no dataset configuration for task {task!r} at {path}')
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = dict(cfg.get('baseline') or {})
    base['task'] = task
    if 'configs' in cfg:
        return [{**base, **(c or {})} for c in cfg['configs']]
    specs = [dict(base)]
    for key, values in (cfg.get('changes') or {}).items():
        if key not in base:
            raise ValueError(f'{path}: change key {key!r} is not set in the baseline')
        for value in values:
            specs.append({**base, key: value})
    return specs


def paper_tasks() -> list[str]:
    """Tasks that have a dataset configuration file, in registry order."""
    return [t for t in task_registry if os.path.exists(task_config_path(t))]
