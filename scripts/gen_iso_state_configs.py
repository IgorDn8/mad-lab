"""Write the iso-hidden-state layer config YAMLs from the registry spec.

Single source of truth is `mad.registry.iso_state_layer_specs()`; this script
just materializes the corresponding YAML files under configs/layers/ so the
registered layers can be built. Re-run after changing the spec.

    uv run python -m scripts.gen_iso_state_configs           # write files
    uv run python -m scripts.gen_iso_state_configs --check    # verify state sizes only
"""

import os
import argparse

import yaml

from mad.registry import iso_state_layer_specs
from mad.paths import get_base_path


def _state_size(name: str, cfg: dict) -> int:
    """Recurrent state size implied by a config (for verification/printing)."""
    if name.startswith('lstm'):
        return cfg['hidden_dim']
    if name.startswith(('bdlru', 'hlru')):
        return cfg['hidden_dim'] * cfg['window_dim']
    if name.startswith('pdssm'):
        return cfg['hidden_dim']
    if name.startswith('mamba2'):
        return cfg['expand'] * cfg['dim'] * cfg['state_size']
    if name.startswith(('dnet', 'dproduct')):
        return cfg['num_heads'] * cfg['head_dim'] ** 2 * cfg.get('expand_v', 1)
    raise ValueError(name)


def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--check', action='store_true',
                   help='only verify the implied state sizes, do not write files')
    return p.parse_args()


def main():
    args = get_args()
    base = get_base_path()
    specs = iso_state_layer_specs()

    written = 0
    print(f"{'layer':40s} {'dim':>5s} {'state':>7s}")
    print('-' * 56)
    for name, _module, cfg_path, _short, cfg in specs:
        state = _state_size(name, cfg)
        print(f"{name:40s} {cfg['dim']:5d} {state:7d}")
        if not args.check:
            path = os.path.join(base, cfg_path)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as f:
                yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
            written += 1
    if not args.check:
        print(f"\nwrote {written} config files under configs/layers/")


if __name__ == '__main__':
    main()
