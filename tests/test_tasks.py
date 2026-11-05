import os
import shutil
from train import train
from mad.configs import MADConfig, MADModelConfig
from mad.registry import task_registry, layer_registry
from mad.helpers import compute_vocab_size
from mad.task_configs import load_task_configs, paper_tasks
from tests import run_tests 


def test_instance_generation():
    print('Testing instance generation for each registered task....')
    test_settings = {
        'vocab_size': (16, 128),
        'seq_len': (64, 128),
        'k_motif_size': (1, 2),
        'v_motif_size': (1, 2),
        'frac_noise': (0.0, 0.5),
        'multi_query': ("True", "False"),
        'noise_vocab_size': (2,)
    }
    for task in task_registry:
        settings = dict(test_settings)
        if task.startswith('group'):
            settings['vocab_size'] = (3, 4)  # group degree; S16 / S128 are intractable
        mad_config = MADConfig(task=task, num_tokens_to_copy=16)
        mad_config.update_from_kwargs({k:v[0] for k,v in settings.items()})
        for k, vs in settings.items():
            for v in vs:
                for train in [True, False]:
                    print(f'...now testing: {task} with {k}={v} and is_training={train}')
                    setattr(mad_config, k, v)
                    kwargs = mad_config.instance_fn_kwargs
                    kwargs['is_training'] = train
                    inputs, targets = mad_config.instance_fn(**kwargs)
                    assert kwargs['seq_len']-1 <= inputs.size < kwargs['seq_len']+1,\
                        f'task={task} with {k}={v}: inputs.size={inputs.size} but kwargs[seq_len]={kwargs["seq_len"]}'
                    assert kwargs['seq_len']-1 <= targets.size < kwargs['seq_len']+1,\
                        f'task={task} with {k}={v}: targets.size={targets.size} but kwargs[seq_len]={kwargs["seq_len"]}'
    print('... passed!')


def test_train_tasks():
    print('Testing training for each registered task....')
    cache_path = os.path.join('.', '.testing')
    # mh-attention needs the `cuda` extra (flash-attn); fall back to BD-LRU otherwise
    mixer = 'mh-attention' if layer_registry['mh-attention']['module'] is not None else 'bdlru-sel-wd1-d128-h32'
    for task in task_registry.keys():
        print(f'...now testing: {task} with default args.')
        group = task.startswith('group')
        mad_config = MADConfig(
            task=task,
            vocab_size=4 if group else 16,  # group degree for group-* tasks
            num_train_examples=10,
            num_test_examples=10,
            batch_size=2,
            epochs=1,
            noise_vocab_size=2,
            num_tokens_to_copy=16,
            data_path=os.path.join(cache_path, 'data'),  # keep test datasets out of ./data
        )
        # as in train.py: the model's vocabulary for group tasks is the group order
        model_vocab = compute_vocab_size(task, mad_config.vocab_size) if group else mad_config.vocab_size
        model_config = MADModelConfig(layers=[mixer, 'swiglu'], vocab_size=model_vocab)
        model = model_config.build_model_from_registry()
        _ = train(
            model=model,
            mad_config=mad_config,
            log_path=os.path.join(cache_path, task)
        )
    shutil.rmtree(cache_path)
    print('... passed!')


def test_paper_task_configs():
    """Every configs/tasks/*.yml expands to valid settings and generates an instance."""
    print('Testing the dataset configurations in configs/tasks/...')
    expected = {  # task -> number of datasets described in the paper
        'in-context-recall': 4, 'selective-copying': 4, 'compression': 4,
        'fuzzy-in-context-recall': 4, 'noisy-in-context-recall': 4,
        'memorization': 3, 'group-S': 5,
    }
    tasks = paper_tasks()
    for task, n in expected.items():
        assert task in tasks, f'missing configs/tasks/{task}.yml'
    for task in tasks:
        specs = load_task_configs(task)
        if task in expected:
            assert len(specs) == expected[task], f'{task}: {len(specs)} settings, expected {expected[task]}'
        for spec in specs:
            mad_config = MADConfig(**spec)
            kwargs = mad_config.instance_fn_kwargs
            kwargs['is_training'] = True
            inputs, targets = mad_config.instance_fn(**kwargs)
            # the MAD generators produce seq_len or seq_len - 1 tokens (same bound as above)
            for name, arr in (('inputs', inputs), ('targets', targets)):
                assert spec['seq_len'] - 1 <= arr.size <= spec['seq_len'], \
                    f'{task} {spec}: {name}.size={arr.size}'
        print(f'...{task}: {len(specs)} dataset settings ok')
    print('... passed!')


if __name__ == '__main__':
    run_tests()
