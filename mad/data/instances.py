import typing as tp
import numpy as np
from itertools import permutations


# utils:

def exists(obj):
    return obj is not None and obj != ''

def generate_vocab_permutations(vocab, token_motif_size: int = 1, rng = None, *args, **kwargs):
    """
    Generate all possible permutations for a given vocabulary.

    Args:
        vocab (list): The vocabulary.
        token_motif_size (int): The size of the token motif. Defaults to 1.
        rng (np.random.Generator, optional): If given, generated tokens are shuffled. 

    Returns:
        list: A list of all possible permutations of the vocabulary.
    """
    values = list(permutations(vocab, token_motif_size))
    if exists(rng):
        rng.shuffle(values)
    return values


# in-context recall:

def generate_in_context_recall_instance(
    vocab_size: int = 16,
    seq_len: int = 128,
    is_training: bool = True,
    rng: np.random.Generator = None,
    target_ignore_idx: int = -100,
    multi_query: bool = False,
    noise_vocab_size: int = 16,
    frac_noise: float = 0.,
    *args, **kwargs
) -> tp.Tuple[np.array, np.array]:
    """
    Generate an instance of the in-context recall task.

    Args:
        vocab_size (int, optional): The size of the vocabulary.
        seq_len (int, optional): The length of the generated sequence.
        is_training (bool, optional): Whether to generate a training or test instance.
        rng (np.random.Generator, optional): The random number generator to use if provided.
        target_ignore_idx (int, optional): Index used in targets to indicate which entries to ignore.
        multi_query (bool, optional): Whether to probe the values for multiple keys.
        noise_vocab_size (int, optional): The size of the noise vocabulary.
        frac_noise (float, optional): The fraction of noise tokens in the sequence.

    Returns:
        tuple: Inputs and targets.
    """

    if not exists(rng):
        rng = np.random.default_rng()

    # generate keys and values:
    copy_prefix = vocab_size - 1
    non_special_vocab_size = vocab_size - 1 if not multi_query else vocab_size
    non_special_vocab_size -= noise_vocab_size
    key_vocab = np.arange(non_special_vocab_size//2)
    value_vocab = np.arange(non_special_vocab_size//2, non_special_vocab_size)

    # generate noise vocab:
    assert frac_noise >= 0 and frac_noise < 1, "frac_noise must be 0 =< frac_noise < 1"
    if frac_noise > 0:
        assert noise_vocab_size > 0, "noise_vocab_size must be >0 if frac_noise >0"
        noise_vocab = np.arange(non_special_vocab_size, non_special_vocab_size+noise_vocab_size)

    # generate inputs/targets:
    kv_map = {}
    inputs, targets = [], []
    keys_presented = {}
    kv_motif_size = 2
    assert seq_len % kv_motif_size == 0, "seq_len must be an even number"
    num_kv_pairs = seq_len // kv_motif_size
    not_noise_idx = rng.choice(num_kv_pairs) # make sure we have at least one key-value pair in the sequence
    for i in range(num_kv_pairs-1): # subtract one to account for final key-value pair
        
        # determine if noise or kv pair collected:
        is_noise = rng.random()<frac_noise if i!=not_noise_idx and frac_noise>0 else False

        # collect noise:
        if is_noise:
            noise = rng.choice(noise_vocab, size=kv_motif_size, replace=True)
            inputs += list(noise)
            targets += [target_ignore_idx]*kv_motif_size
        
        # collect kv pair:
        else:
            # key:
            k = rng.choice(key_vocab)
            # value:
            if k not in kv_map:
                v = rng.choice(value_vocab)
                kv_map[k] = v
            else:
                v = kv_map[k]
    
            inputs.append(k)
            inputs.append(v)
            
            targets.append(target_ignore_idx)
            if k not in keys_presented:
                targets.append(target_ignore_idx)
            else:
                if multi_query:
                    targets.append(v) # probe value if key has been presented before
                else:
                    targets.append(target_ignore_idx)

            keys_presented[k] = v   

    # add a final key-value pair to the sequence as well as a copy-prefix:
    k_probe = rng.choice(list(keys_presented.keys()))
    v_probe = keys_presented[k_probe]
    
    if not multi_query:
        inputs.append(copy_prefix)
    inputs.append(k_probe)
    inputs.append(v_probe)
    
    if not multi_query:
        targets.append(-100) # copy prefix
    targets.append(-100) # k_probe
    targets.append(v_probe)
                   
    inputs = np.array(inputs).astype(int)
    targets = np.array(targets).astype(int)

    if is_training:
        # autoregressive shift
        return inputs[:-1], inputs[1:] # use shifted inputs as targets for training
    else:
        return inputs[:-1], targets[1:]


# noisy in-context recall:

def generate_noisy_in_context_recall_instance(
    vocab_size: int = 32,
    seq_len: int = 128,
    noise_vocab_size: int = 16,
    frac_noise: float = 0.2,
    is_training: bool = True,
    rng: np.random.Generator = None,
    target_ignore_idx: int = -100,
    multi_query: bool = False,
    *args, **kwargs
) -> tp.Tuple[np.array, np.array]:
    """
    Generate an instance of the noisy in-context recall task.
    
    Args:
        vocab_size (int, optional): The size of the vocabulary.
        seq_len (int, optional): The length of the generated sequence.
        noise_vocab_size (int, optional): The size of the noise vocabulary (will be subtracted from vocab_size).
        frac_noise (float, optional): The fraction of noise tokens in the sequence.
        is_training (bool, optional): Whether to generate a training or test instance.
        rng (np.random.Generator, optional): The random number generator to use if provided.
        target_ignore_idx (int, optional): Index used in targets to indicate which entries to ignore.
        multi_query (bool, optional): Whether to probe the values for multiple keys.
        
    Returns:
        tuple: Inputs and targets.
    """
    return generate_in_context_recall_instance(
        vocab_size=vocab_size,
        seq_len=seq_len,
        is_training=is_training,
        rng=rng,
        target_ignore_idx=target_ignore_idx,
        multi_query=multi_query,
        noise_vocab_size=noise_vocab_size,
        frac_noise=frac_noise
    )


# fuzzy in-context recall:

def generate_fuzzy_in_context_recall_instance(
    vocab_size: int = 16,
    seq_len: int = 128,
    k_motif_size: int = 3,
    v_motif_size: int = 3,
    is_training: bool = True,
    rng: np.random.Generator = None,
    target_ignore_idx: int = -100,
    multi_query: bool = False,
    noise_vocab_size: int = 0,
    frac_noise: float = 0,
    *args, **kwargs
) -> tp.Tuple[np.array, np.array]:
    """
    Generate an instance of the fuzzy in-context recall task.

    Args:
        vocab_size (int, optional): The size of the vocabulary.
        seq_len (int, optional): The length of the generated sequence.
        k_motif_size (int, optional): The maximum number of adjacent tokens used to represent a key.
        v_motif_size (int, optional): The maximum number of adjacent tokens used to represent a value.
        is_training (bool, optional): Whether to generate a training or test instance.
        rng (np.random.Generator, optional): The random number generator to use if provided.
        target_ignore_idx (int, optional): Index used in targets to indicate which entries to ignore.
        multi_query (bool, optional): Whether to probe the values for multiple keys.
        noise_vocab_size (int, optional): The size of the noise vocabulary (will be subtracted from vocab_size).
        frac_noise (float, optional): The fraction of noise tokens in the sequence.
        
    Returns:
        tuple: Inputs and targets.
    """

    if not exists(rng):
        rng = np.random.default_rng()

    # generate keys and values
    copy_prefix = vocab_size - 1
    pad_token = vocab_size - 2 if not multi_query else vocab_size - 1
    non_special_vocab_size = vocab_size - 2 if not multi_query else vocab_size - 1
    non_special_vocab_size -= noise_vocab_size
    key_vocab = np.arange(non_special_vocab_size//2)
    value_vocab = np.arange(non_special_vocab_size//2, non_special_vocab_size)
    if is_training:
        # generate keys and values of variable motif sizes
        keys = {}
        for motif_size in range(1, k_motif_size+1):
            keys_size = generate_vocab_permutations(key_vocab, token_motif_size=motif_size, rng=rng)
            keys[motif_size] = keys_size
        values = {}
        for motif_size in range(1, v_motif_size+1):
            values_size = generate_vocab_permutations(value_vocab, token_motif_size=motif_size, rng=rng)
            values[motif_size] = values_size
    else:
        # we always prompt at the maximum key motif size
        keys = {k_motif_size: generate_vocab_permutations(key_vocab, token_motif_size=k_motif_size, rng=rng)}
        values = {}
        for motif_size in range(1, v_motif_size+1):
            values_size = generate_vocab_permutations(value_vocab, token_motif_size=motif_size, rng=rng)
            values[motif_size] = values_size

    # generate noise vocab, if needed:
    assert frac_noise >= 0 and frac_noise < 1, "frac_noise must be 0 =< frac_noise < 1"
    if frac_noise > 0:
        assert noise_vocab_size > 0, "noise_vocab_size must be >0 if frac_noise >0"
        noise_vocab = np.arange(non_special_vocab_size, non_special_vocab_size+noise_vocab_size)

    # generate key-value probe:
    k_probe_size = rng.choice(list(keys.keys())) if is_training else k_motif_size
    v_probe_size = rng.choice(list(values.keys()))
    k_probe = tuple(rng.choice(keys[k_probe_size]))
    v_probe = tuple(rng.choice(values[v_probe_size]))
    kv_probe_size = k_probe_size + v_probe_size
    probe_idx = rng.choice(seq_len - kv_probe_size - kv_probe_size) # make sure we add key-value pair to the non-probe part of the sequence
    probe_added = False # flag indicating whether key-value probe has been added to the sequence    

    # generate inputs/targets:
    kv_map = {s: {} for s in range(1, k_motif_size+1)}
    inputs, targets = [], []
    keys_presented = {}
    # make sure we dont generate too long sequences
    # we pad later to make sure outupts are of length input_seq_len
    while len(inputs) < seq_len - kv_probe_size - (k_motif_size + v_motif_size): 

        # determine key-value motif sizes:
        k_size = rng.choice(list(keys.keys())) if is_training else k_motif_size
        v_size = rng.choice(list(values.keys()))

        # make sure key-value probe is in the sequence:
        if len(inputs) >= probe_idx and not probe_added:
            inputs.extend(k_probe)
            inputs.extend(v_probe)
            targets.extend(tuple([target_ignore_idx]*(k_probe_size+len(v_probe))))
            kv_map[k_probe_size][k_probe] = v_probe
            keys_presented[k_probe] = v_probe
            probe_added = True
            continue

        # determine if noise or key-value pair collected:
        is_noise = rng.random()<frac_noise if frac_noise>0 else False

        # collect noise:
        if is_noise:
            noise_size = k_size + v_size
            noise = rng.choice(noise_vocab, size=noise_size, replace=True)
            inputs.extend(noise)
            targets.extend(tuple([target_ignore_idx]*noise_size))

        # collect key-value pair:
        else:
            # key:
            k = tuple(rng.choice(keys[k_size]))
            inputs.extend(k)

            # value:
            if k == k_probe:
                v = v_probe
                probe_added = True
            else:
                if k not in kv_map[k_size]:
                    v = tuple(rng.choice(values[v_size]))
                    kv_map[k_size][k] = v
                else:
                    v = kv_map[k_size][k]
            inputs.extend(v)
            
            # determine targets:
            targets.extend(tuple([target_ignore_idx]*k_size))
            if k not in keys_presented:
                targets.extend(tuple([target_ignore_idx]*len(v)))
            else:
                if multi_query:
                    targets.extend(v) # probe value if key has been presented before
                else:
                    targets.extend(tuple([target_ignore_idx]*len(v))) 

            keys_presented[k] = v   

    # add a final key-value pair to the sequence as well as a copy-prefix:
    if not multi_query:
        inputs.extend(tuple([copy_prefix]))
    inputs.extend(k_probe)
    inputs.extend(v_probe)

    if not multi_query:
        targets.extend(tuple([-100]))
    targets.extend(tuple([-100]*k_probe_size))
    targets.extend(v_probe)
                   
    inputs = np.array(inputs).astype(int)
    targets = np.array(targets).astype(int)

    # pad inputs/targets to seq_len:
    if len(inputs)<(seq_len+1): # add one to account for autoregressive shift
        n_pad = seq_len+1-len(inputs)
        inputs = np.concatenate([np.array([pad_token]*n_pad), inputs])
        targets = np.concatenate([np.array([target_ignore_idx]*n_pad), targets])
    
    if is_training:
        # autoregressive shift
        return inputs[:-1], inputs[1:] # use shifted inputs as targets for training
    else:
        return inputs[:-1], targets[1:]


# memorization:
    
def generate_kv_map(
    vocab_size: int,
    k_motif_size: int = 1,
    v_motif_size: int = 1,
    seed: int = 12345,
    *args, **kwargs
) -> dict:
    """
    Generate a fixed mapping from keys to values.
    
    Args:
        vocab_size (int): The size of the vocabulary.
        {k,v}_motif_size (int, optional): The number of adjacent tokens used to represent a key/value.
        seed (int, optional): The seed used for the random number generator.
    
    Returns:
        dict: A dictionary mapping keys to values.
    """
    kv_map_rng = np.random.default_rng(seed) # fixed rng to ensure stable kv mapping across instances
    key_vocab = np.arange(vocab_size//2)
    value_vocab = np.arange(vocab_size//2, vocab_size)
    keys = generate_vocab_permutations(key_vocab, k_motif_size, kv_map_rng)
    values = generate_vocab_permutations(value_vocab, v_motif_size, kv_map_rng)
    return {k: v for k, v in zip(keys, values)}

def generate_memorization_instance(
    vocab_size: int = 256,
    seq_len: int = 32,
    kv_map: dict = None,
    noise_vocab_size: int = 0,
    frac_noise: float = 0,
    rng: np.random.Generator = None,
    target_ignore_idx: int = -100,
    kv_map_seed: int = 12345,
    *args, **kwargs
) -> tp.Tuple[np.array, np.array]:
    """
    Generate an instance of the memorization task.

    Args:
        vocab_size (int, optional): The size of the vocabulary.
        seq_len (int, optional): The length of the generated sequence.
        kv_map (dict, optional): A dictionary mapping keys to values.
        noise_vocab_size (int, optional): The size of the noise vocabulary (will be subtracted from vocab_size).
        frac_noise (float, optional): The fraction of noise tokens in the sequence.
        rng (np.random.Generator, optional): The random number generator to use if provided.
        target_ignore_idx (int, optional): Index used in targets to indicate which entries to ignore.
        kv_map_seed (int, optional): The seed for the random number generator used to generate the kv mapping.

    Returns:
        tuple: Inputs (keys) and targets (values)  
    """
    
    if not exists(rng):
        rng = np.random.default_rng()

    # define vocab sizes:
    insert_token = vocab_size - 1
    non_special_vocab_size = vocab_size - 1
    non_special_vocab_size -= noise_vocab_size
    
    if not exists(kv_map):
        # generate fixed mapping from keys to values:
        kv_map = generate_kv_map(
            vocab_size=non_special_vocab_size, # account for insert token
            k_motif_size=1,
            v_motif_size=1,
            seed=kv_map_seed
        )

    keys = list(kv_map.keys())

    # generate noise vocab:
    assert frac_noise >= 0 and frac_noise <=1, "frac_noise must be >=0 and <=1"
    if frac_noise > 0:
        assert noise_vocab_size > 0, "noise vocab size must be >0 if frac_noise >0"
        noise_vocab = np.arange(non_special_vocab_size, non_special_vocab_size+noise_vocab_size)

    # generate inputs/targets:
    inputs, targets = [], []
    kv_motif_size = 2
    num_kv_pairs = seq_len // kv_motif_size
    not_noise_idx = rng.choice(num_kv_pairs) # make sure we have at least one key/value pair in the sequence
    for i in range(num_kv_pairs): # subtract one to account for the final key-value pair
        
        # determine if noise or key-value pair collected:
        is_noise = rng.random()<frac_noise if i!=not_noise_idx and frac_noise>0 else False

        # collect noise:
        if is_noise:
            noise = rng.choice(noise_vocab, size=kv_motif_size, replace=True)
            inputs.append(noise)
            targets.append([target_ignore_idx]*kv_motif_size)
        
        # collect key/value pair:
        else:
            k = tuple(rng.choice(keys))
            v = kv_map[k]
        
            inputs.append(k)
            inputs.append([insert_token])
            
            targets.append([target_ignore_idx])
            targets.append(v)
    
    inputs = np.concatenate(inputs).astype(int)
    targets = np.concatenate(targets).astype(int)
    return inputs, targets


# compression:

def generate_compression_instance(
    vocab_size: int = 16,
    seq_len: int = 32,
    noise_vocab_size: int = 0,
    frac_noise: float = 0,
    rng: np.random.Generator = None,
    target_ignore_idx: int = -100,
    *args, **kwargs
) -> tp.Tuple[np.array, np.array]:
    """
    Generate an instance of the compression task.
    
    Args:
        vocab_size (int, optional): The size of the vocabulary.
        seq_len (int, optional): The length of the generated sequence.
        noise_vocab_size (int, optional): The size of the noise vocabulary (will be subtracted from vocab_size).
        frac_noise (float, optional): The fraction of noise tokens in the sequence.
        rng (np.random.Generator, optional): The random number generator to use if provided.
        target_ignore_idx (int, optional): Index used in targets to indicate which entries to ignore.
        
    Returns:
        tuple: Inputs and targets.    
    """

    if not exists(rng):
        rng = np.random.default_rng()

    # generate inputs/targets:
    compression_token = vocab_size - 1
    non_special_vocab_size = vocab_size - 1
    non_special_vocab_size -= noise_vocab_size
    vocab = np.arange(non_special_vocab_size)
    inputs = rng.choice(vocab, size=(seq_len-1,), replace=True).reshape(-1)
    inputs = np.concatenate([inputs, np.array([compression_token])])
    targets = np.array(inputs)

    # add noise:
    if frac_noise > 0:
        assert noise_vocab_size > 0, "noise vocab size must be > 0 if frac_noise > 0"
        noise_vocab = np.arange(non_special_vocab_size, non_special_vocab_size+noise_vocab_size)
        for i in range(seq_len-1): # exclude compression token
            if rng.random() < frac_noise:
                inputs[i:(i+1)] = rng.choice(noise_vocab)
                targets[i:(i+1)] = target_ignore_idx

    return inputs, inputs

# Markov compression:

def generate_mf_compression_instance(
    vocab_size: int = 16,
    seq_len: int = 32,
    k_motif_size: int = 1,
    v_motif_size: int = 1,
    noise_vocab_size: int = 0,
    frac_noise: float = 0,
    rng: np.random.Generator = None,
    target_ignore_idx: int = -100,
    *args, **kwargs
) -> tp.Tuple[np.array, np.array]:
    """
    Generate an instance of the Markov compression task.
    
    Args:
        vocab_size (int, optional): The size of the vocabulary.
        seq_len (int, optional): The length of the generated sequence.
        noise_vocab_size (int, optional): The size of the noise vocabulary (will be subtracted from vocab_size).
        frac_noise (float, optional): The fraction of noise tokens in the sequence.
        rng (np.random.Generator, optional): The random number generator to use if provided.
        target_ignore_idx (int, optional): Index used in targets to indicate which entries to ignore.
        
    Returns:
        tuple: Inputs and targets.    
    """

    if not exists(rng):
        rng = np.random.default_rng()

    # generate inputs/targets:
    compression_token = vocab_size - 1
    non_special_vocab_size = vocab_size - 1
    non_special_vocab_size -= noise_vocab_size
    vocab = np.arange(non_special_vocab_size)
    #inputs = rng.choice(vocab, size=(seq_len-1,), replace=True).reshape(-1)
    matrix = rng.random((non_special_vocab_size, non_special_vocab_size))
    matrix = matrix / matrix.sum(axis=1, keepdims=True)

    inputs = []
    start_tokens = rng.choice(vocab,k_motif_size)
    inputs += [token for token in start_tokens]

    for _ in range(seq_len - 2):
        probs = matrix[current_token]
        next_token = rng.choice(vocab, p=probs)
        inputs.append(next_token)
        current_token = next_token

    inputs = np.concatenate([np.array(inputs), np.array([compression_token])])
    targets = np.array(inputs)

    # add noise:
    if frac_noise > 0:
        assert noise_vocab_size > 0, "noise vocab size must be > 0 if frac_noise > 0"
        noise_vocab = np.arange(non_special_vocab_size, non_special_vocab_size+noise_vocab_size)
        for i in range(seq_len-1): # exclude compression token
            if rng.random() < frac_noise:
                inputs[i:(i+1)] = rng.choice(noise_vocab)
                targets[i:(i+1)] = target_ignore_idx

    return inputs, targets


def generate_m_compression_instance(
    vocab_size: int = 16,
    seq_len: int = 32,
    k_motif_size: int = 1,
    v_motif_size: int = 1,
    noise_vocab_size: int = 0,
    frac_noise: float = 0,
    rng: np.random.Generator = None,
    target_ignore_idx: int = -100,
    *args, **kwargs
) -> tp.Tuple[np.array, np.array]:
    """
    Generate an instance of the Markov compression task.
    
    Args:
        vocab_size (int, optional): The size of the vocabulary.
        seq_len (int, optional): The length of the generated sequence.
        noise_vocab_size (int, optional): The size of the noise vocabulary (will be subtracted from vocab_size).
        frac_noise (float, optional): The fraction of noise tokens in the sequence.
        rng (np.random.Generator, optional): The random number generator to use if provided.
        target_ignore_idx (int, optional): Index used in targets to indicate which entries to ignore.
        
    Returns:
        tuple: Inputs and targets.    
    """

    if not exists(rng):
        rng = np.random.default_rng()

    # generate inputs/targets:
    compression_token = vocab_size - 1
    non_special_vocab_size = vocab_size - 1
    non_special_vocab_size -= noise_vocab_size
    vocab = np.arange(non_special_vocab_size)
    #inputs = rng.choice(vocab, size=(seq_len-1,), replace=True).reshape(-1)
    from scipy.special import softmax 
    temperature = 0.01
    logits = rng.random((non_special_vocab_size, non_special_vocab_size))
    matrix = np.array([
        softmax(row / temperature) for row in logits
    ])
    #matrix = matrix / matrix.sum(axis=1, keepdims=True)

    inputs = []
    current_token = rng.choice(vocab)
    inputs.append(current_token)

    for _ in range(seq_len - 2):
        probs = matrix[current_token]
        next_token = rng.choice(vocab, p=probs)
        inputs.append(next_token)
        current_token = next_token

    inputs = np.concatenate([np.array(inputs), np.array([compression_token])])
    targets = np.array(inputs)

    # add noise:
    if frac_noise > 0:
        assert noise_vocab_size > 0, "noise vocab size must be > 0 if frac_noise > 0"
        noise_vocab = np.arange(non_special_vocab_size, non_special_vocab_size+noise_vocab_size)
        for i in range(seq_len-1): # exclude compression token
            if rng.random() < frac_noise:
                inputs[i:(i+1)] = rng.choice(noise_vocab)
                targets[i:(i+1)] = target_ignore_idx

    return inputs, targets


# copying:

def generate_copying_instance(
    vocab_size: int = 16,
    seq_len: int = 256,
    num_tokens_to_copy: int = 16,
    rng: np.random.Generator = None,
    target_ignore_idx: int = -100,
    selective: bool = False,
    *args, **kwargs
) -> tp.Tuple[np.array, np.array]:
    """
    Generate an instance of the copying task.

    Args:
        vocab_size (int, optional): The size of the vocabulary.
        seq_len (int, optional): The length of the generated sequence.
        num_tokens_to_copy (int, optional): The number of tokens to copy.
        rng (np.random.Generator, optional): The random number generator to use if provided.
        target_ignore_idx (int, optional): Index used in targets to indicate which entries to ignore.
        selective (bool, optional): Whether to randomly insert blank tokens in sequence (-> selective copying task).

    Returns:
        tuple: Inputs and targets.
    """
    
    if not exists(rng):
        rng = np.random.default_rng()

    # define vocab:
    copy_token = vocab_size - 1
    blank_token = vocab_size - 2
    non_special_vocab_size = vocab_size - 2
    vocab = np.arange(non_special_vocab_size)

    # generate inputs:
    assert seq_len > (num_tokens_to_copy * 2) + 1, "seq_len must be > (num_tokens_to_copy * 2) + 1"
    num_blank_tokens = seq_len - (num_tokens_to_copy * 2) - 1
    to_copy = rng.choice(vocab, size=(num_tokens_to_copy,), replace=True).reshape(-1)
    if not selective:
        inputs = list(to_copy)
        inputs += [blank_token] * num_blank_tokens
    else:
        inputs = np.array(to_copy)
        insert_indices = np.random.randint(0, len(inputs), num_blank_tokens)
        inputs = np.insert(inputs, insert_indices, [blank_token]*num_blank_tokens).tolist()
    inputs += [copy_token] 
    inputs += [blank_token] * num_tokens_to_copy
    inputs = np.array(inputs)

    # generate targets:
    targets = [target_ignore_idx] * (num_tokens_to_copy + num_blank_tokens + 1)
    targets += list(to_copy)
    targets = np.array(targets)

    return inputs, targets


# selective copying:

def generate_selective_copying_instance(
    vocab_size: int = 16,
    seq_len: int = 256,
    num_tokens_to_copy: int = 16,
    rng: np.random.Generator = None,
    target_ignore_idx: int = -100,
    *args, **kwargs
) -> tp.Tuple[np.array, np.array]:
    """
    Generate a instance of the selective copying task.

    Args:
        vocab_size (int, optional): The size of the vocabulary.
        seq_len (int, optional): The length of the generated sequence.
        num_tokens_to_copy (int, optional): The number of tokens to copy.
        rng (np.random.Generator, optional): The random number generator to use if provided.
        target_ignore_idx (int, optional): Index used in targets to indicate which entries to ignore.

    Returns:
        tuple: Inputs and targets.
    """
    return generate_copying_instance(
        vocab_size=vocab_size,
        seq_len=seq_len,
        num_tokens_to_copy=num_tokens_to_copy,
        rng=rng,
        target_ignore_idx=target_ignore_idx,
        selective=True
    )

# # group sequence:
from abstract_algebra.finite_algebras import (
    FiniteAlgebra,
    generate_cyclic_group,
    generate_symmetric_group,
)

# Building a finite group (esp. a symmetric group and its commutator subalgebra)
# is expensive. The original instance functions rebuilt the group on *every*
# sample, which dominated data-generation time. We instead build each group once
# and cache it, together with a precomputed Cayley (multiplication) table that
# turns the per-token reduction into an O(1) integer lookup.
_GROUP_CACHE: tp.Dict[tp.Tuple[str, int], dict] = {}

# Groups larger than this are not materialized into a full |G| x |G| Cayley
# table (which costs O(|G|^2) memory); the reduction falls back to a cached
# element->index map instead. Group tasks in practice use tiny groups (e.g. S5
# has 120 elements), so this threshold is never hit in normal use.
_MAX_CAYLEY_ELEMENTS = 2048


def _build_group(kind: str, n: int) -> FiniteAlgebra:
    """Construct a finite group from a kind identifier ('S', 'Z', 'A') and degree n."""
    if kind == 'S':
        return generate_symmetric_group(n)
    elif kind == 'Z':
        return generate_cyclic_group(n)
    elif kind == 'A':
        s_n = generate_symmetric_group(n)
        a_n = s_n.commutator_subalgebra()
        a_n.name = f"A{n}"
        return a_n
    else:
        raise ValueError("Group kind must be one of 'S', 'Z', or 'A'")


def get_cached_group(kind: str, n: int) -> dict:
    """
    Return cached data for the group of the given kind and degree.

    The group is built only once per (kind, n); subsequent calls reuse it.

    Returns:
        dict with keys:
            'group' (FiniteAlgebra): the group,
            'num_elements' (int): |G|,
            'elem_to_idx' (dict): element -> index map,
            'cayley' (np.ndarray | None): |G| x |G| table with
                cayley[i, j] = index(G.op(elements[i], elements[j])),
                or None for very large groups.
    """
    key = (kind, n)
    if key not in _GROUP_CACHE:
        group = _build_group(kind, n)
        elements = group.elements
        num_elements = len(elements)
        elem_to_idx = {e: i for i, e in enumerate(elements)}

        cayley = None
        if num_elements <= _MAX_CAYLEY_ELEMENTS:
            cayley = np.empty((num_elements, num_elements), dtype=np.int64)
            for i, ei in enumerate(elements):
                for j, ej in enumerate(elements):
                    cayley[i, j] = elem_to_idx[group.op(ei, ej)]

        _GROUP_CACHE[key] = {
            'group': group,
            'num_elements': num_elements,
            'elem_to_idx': elem_to_idx,
            'cayley': cayley,
        }
    return _GROUP_CACHE[key]


def group_reduce(lhs: str | int, rhs: int, G) -> int:
    """Reduce a sequence of group elements to a single element (kept for backwards compat)."""
    if isinstance(lhs, str):
        prod = G.op(lhs, G.elements[rhs])
    else:
        prod = G.op(G.elements[lhs], G.elements[rhs])

    return G.elements.index(prod)


def prefix_products(inputs: np.ndarray, group_data: dict) -> np.ndarray:
    """
    Compute the running (prefix) product of a sequence of group-element indices.

    targets[t] = index of (elements[inputs[0]] * ... * elements[inputs[t]]),
    with the reduction seeded from accumulator index 0 (matching the original
    implementation).

    Works on a 1D sequence or a 2D batch of sequences (reduction along axis -1).
    """
    inputs = np.asarray(inputs)
    cayley = group_data['cayley']
    group = group_data['group']
    elem_to_idx = group_data['elem_to_idx']

    if inputs.ndim == 1:
        acc = 0
        targets = np.empty(inputs.shape[0], dtype=np.int64)
        if cayley is not None:
            for t, x in enumerate(inputs):
                acc = cayley[acc, x]
                targets[t] = acc
        else:
            elements = group.elements
            for t, x in enumerate(inputs):
                acc = elem_to_idx[group.op(elements[acc], elements[x])]
                targets[t] = acc
        return targets

    # batched (num_sequences, seq_len): vectorized scan along the sequence axis
    num_seq, seq_len = inputs.shape
    acc = np.zeros(num_seq, dtype=np.int64)
    targets = np.empty((num_seq, seq_len), dtype=np.int64)
    if cayley is not None:
        for t in range(seq_len):
            acc = cayley[acc, inputs[:, t]]
            targets[:, t] = acc
    else:
        elements = group.elements
        for t in range(seq_len):
            acc = np.array(
                [elem_to_idx[group.op(elements[a], elements[x])]
                 for a, x in zip(acc, inputs[:, t])],
                dtype=np.int64,
            )
            targets[:, t] = acc
    return targets


def _generate_group_instance(
    kind: str,
    vocab_size: int,
    seq_len: int,
    rng: np.random.Generator = None,
) -> tp.Tuple[np.array, np.array]:
    """Shared fast implementation for the group tasks (uses the cached group)."""
    if not exists(rng):
        rng = np.random.default_rng()

    group_data = get_cached_group(kind, vocab_size)
    num_elements = group_data['num_elements']

    inputs = rng.choice(range(num_elements), size=seq_len)
    targets = prefix_products(inputs, group_data)

    return inputs, targets


def generate_group_S_instance(
    vocab_size: int = 4,
    seq_len: int = 4,
    rng: np.random.Generator = None,
    *args, **kwargs
) -> tp.Tuple[np.array, np.array]:
    """
    Generate an instance of the symmetric-group (S_n) task.

    Args:
        vocab_size (int, optional): The degree n of the symmetric group S_n.
        seq_len (int, optional): The length of the generated sequence.
        rng (np.random.Generator, optional): The random number generator to use if provided.

    Returns:
        tuple: Inputs and targets.
    """
    return _generate_group_instance('S', vocab_size, seq_len, rng)


def generate_group_Z_instance(
    vocab_size: int = 4,
    seq_len: int = 4,
    rng: np.random.Generator = None,
    *args, **kwargs
) -> tp.Tuple[np.array, np.array]:
    """
    Generate an instance of the cyclic-group (Z_n) task.

    Args:
        vocab_size (int, optional): The number of elements in the cyclic group Z_n.
        seq_len (int, optional): The length of the generated sequence.
        rng (np.random.Generator, optional): The random number generator to use if provided.

    Returns:
        tuple: Inputs and targets.
    """
    return _generate_group_instance('Z', vocab_size, seq_len, rng)


def generate_group_A_instance(
    vocab_size: int = 4,
    seq_len: int = 4,
    rng: np.random.Generator = None,
    *args, **kwargs
) -> tp.Tuple[np.array, np.array]:
    """
    Generate an instance of the alternating-group (A_n) task.

    Args:
        vocab_size (int, optional): The degree n of the alternating group A_n.
        seq_len (int, optional): The length of the generated sequence.
        rng (np.random.Generator, optional): The random number generator to use if provided.

    Returns:
        tuple: Inputs and targets.
    """
    return _generate_group_instance('A', vocab_size, seq_len, rng)