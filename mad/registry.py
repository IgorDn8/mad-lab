from mad.data.instances import (
    generate_in_context_recall_instance,
    generate_noisy_in_context_recall_instance,
    generate_fuzzy_in_context_recall_instance,
    generate_memorization_instance,
    generate_compression_instance,    
    generate_m_compression_instance,
    generate_selective_copying_instance,
    generate_group_S_instance,
    generate_group_Z_instance,
    generate_group_A_instance,
)
from mad.model import layers
from mad import model


task_registry = {
    'in-context-recall': {
        'instance_fn': generate_in_context_recall_instance,
        'cfg': 'configs/tasks/in-context-recall.yml',
        'shorthand': 'CR'
    },
    'noisy-in-context-recall': {
        'instance_fn': generate_noisy_in_context_recall_instance,
        'cfg': 'configs/tasks/noisy-in-context-recall.yml',
        'shorthand': 'NR'
    },
    'fuzzy-in-context-recall': {
        'instance_fn': generate_fuzzy_in_context_recall_instance,
        'cfg': 'configs/tasks/fuzzy-in-context-recall.yml',
        'shorthand': 'FR'
    },
    'memorization': {
        'instance_fn': generate_memorization_instance,
        'cfg': 'configs/tasks/memorization.yml',
        'shorthand': 'M'
    },
    'compression': {
        'instance_fn': generate_compression_instance,
        'cfg': 'configs/tasks/compression.yml',
        'shorthand': 'C'
    },
    'selective-copying': {
        'instance_fn': generate_selective_copying_instance,
        'cfg': 'configs/tasks/selective-copying.yml',
        'shorthand': 'SC'
    },    
    'm-compression': {
        'instance_fn': generate_m_compression_instance,
        'cfg': 'configs/tasks/m-compression.yml',
        'shorthand': 'MC'
    },
    'group-S': {
        'instance_fn': generate_group_S_instance,
        'cfg': 'configs/tasks/group-S.yml',
        'shorthand': 'GS'
    },
    'group-Z': {
        'instance_fn': generate_group_Z_instance,
        'cfg': 'configs/tasks/group-Z.yml',
        'shorthand': 'GZ'
    },
    'group-A': {
        'instance_fn': generate_group_A_instance,
        'cfg': 'configs/tasks/group-A.yml',
        'shorthand': 'GA'
    },
}


layer_registry = {
    # channel mixers:
    'mlp': {
        'module': layers.Mlp,
        'cfg': 'configs/layers/mlp.yml',
        'shorthand': 'M'
    },
    'moe-mlp': {
        'module': layers.MoeMlp,
        'cfg': 'configs/layers/moe-mlp.yml',
        'shorthand': 'MoE'
    },
    'swiglu': {
        'module': layers.SwiGLU,
        'cfg': 'configs/layers/swiglu.yml',
        'shorthand': 'Sg'
    },
    #sequence mixers:
    'attention': {
        'module': layers.Attention,
        'cfg': 'configs/layers/attention.yml',
        'shorthand': 'A'
    },
    'sliding-attention': {
        'module': layers.Attention,
        'cfg': 'configs/layers/sliding-attention.yml',
        'shorthand': 'As'
    },
    'linear-attention': {
        'module': layers.LinearAttention,
        'cfg': 'configs/layers/linear-attention.yml',
        'shorthand': 'Al'
    },
    'gated-linear-attention': {
        'module': layers.GatedLinearAttention,
        'cfg': 'configs/layers/gated-linear-attention.yml',
        'shorthand': 'Alg'
    },
    'hyena': {
        'module': layers.HyenaOperator,
        'cfg': 'configs/layers/hyena.yml',
        'shorthand': 'H'
    },
    'hyena-experts': {
        'module': layers.HyenaExpertsOperator,
        'cfg': 'configs/layers/hyena-experts.yml',
        'shorthand': 'He'
    },
    'mamba': {
        'module':layers.Mamba,
        'cfg': 'configs/layers/mamba.yml',
        'shorthand': 'Mb'
    },
    'mh-attention': {
        'module': layers.Attention,
        'cfg': 'configs/layers/mh-attention.yml',
        'shorthand': 'mA'
    },
    'mh-sliding-attention': {
        'module': layers.Attention,
        'cfg': 'configs/layers/mh-sliding-attention.yml',
        'shorthand': 'mAs'
    },
    'mh-linear-attention': {
        'module': layers.LinearAttention,
        'cfg': 'configs/layers/mh-linear-attention.yml',
        'shorthand': 'mAl'
    },
    'mh-gated-linear-attention': {
        'module': layers.GatedLinearAttention,
        'cfg': 'configs/layers/mh-gated-linear-attention.yml',
        'shorthand': 'mAlg'
    },
    'mh-hyena': {
        'module': layers.MultiHeadHyenaOperator,
        'cfg': 'configs/layers/mh-hyena.yml',
        'shorthand': 'mH'
    },
    'rwkv5-time-mixer': {
        'module': layers.time_mixer_rwkv5_wrapped_bf16,
        'cfg': 'configs/layers/rwkv5-time-mixer.yml',
        'shorthand': 'R5t'
    },
    'rwkv5-channel-mixer': {
        'module': layers.channel_mixer_rwkv5_wrapped,
        'cfg': 'configs/layers/rwkv5-channel-mixer.yml',
        'shorthand': 'R5c'
    },
    'rwkv6-time-mixer': {
        'module': layers.time_mixer_rwkv6_wrapped_bf16,
        'cfg': 'configs/layers/rwkv6-time-mixer.yml',
        'shorthand': 'R6t'
    },
    'rwkv6-channel-mixer': {
        'module': layers.channel_mixer_rwkv6_wrapped,
        'cfg': 'configs/layers/rwkv6-channel-mixer.yml',
        'shorthand': 'R6c'
    },           
    'lstm-d128-h8': {
        'module': layers.LSTM,
        'cfg': 'configs/layers/lstm-d128-h8.yml',
        'shorthand': 'oLSTMd128h8'
    },  
    'lstm-d128-h16': {
        'module': layers.LSTM,
        'cfg': 'configs/layers/lstm-d128-h16.yml',
        'shorthand': 'oLSTMd128h16'
    }, 
    'lstm-d128-h24': {
        'module': layers.LSTM,
        'cfg': 'configs/layers/lstm-d128-h24.yml',
        'shorthand': 'oLSTMd128h24'
    },  
    'lstm-d128-h32': {
        'module': layers.LSTM,
        'cfg': 'configs/layers/lstm-d128-h32.yml',
        'shorthand': 'oLSTMd128h32'
    }, 
    'lstm-d128-h48': {
        'module': layers.LSTM,
        'cfg': 'configs/layers/lstm-d128-h48.yml',
        'shorthand': 'oLSTMd128h48'
    },  
    'lstm-d128-h64': {
        'module': layers.LSTM,
        'cfg': 'configs/layers/lstm-d128-h64.yml',
        'shorthand': 'oLSTMd128h64'
    }, 
    'lstm-d128-h96': {
        'module': layers.LSTM,
        'cfg': 'configs/layers/lstm-d128-h96.yml',
        'shorthand': 'oLSTMd128h96'
    },
    'lstm-d128-h128': {
        'module': layers.LSTM,
        'cfg': 'configs/layers/lstm-d128-h128.yml',
        'shorthand': 'oLSTMd128h128'
    },            
    'lstm-d128-h192': {
        'module': layers.LSTM,
        'cfg': 'configs/layers/lstm-d128-h192.yml',
        'shorthand': 'oLSTMd128h192'
    }, 
    'lstm-d128-h224': {
        'module': layers.LSTM,
        'cfg': 'configs/layers/lstm-d128-h224.yml',
        'shorthand': 'oLSTMd128h224'
    }, 
    'lstm-d128-h416': {
        'module': layers.LSTM,
        'cfg': 'configs/layers/lstm-d128-h416.yml',
        'shorthand': 'oLSTMd128h416'
    }, 
    # --- d=1024 (wide, shallow-state) iso-parameter tier (~1M) ---
    'lstm-d1024-h176': {
        'module': layers.LSTM,
        'cfg': 'configs/layers/lstm-d1024-h176.yml',
        'shorthand': 'oLSTMd1024h176'
    }, 
    # --- d=1024 iso-parameter tier (~10M) ---
    'lstm-d1024-h1064': {
        'module': layers.LSTM,
        'cfg': 'configs/layers/lstm-d1024-h1064.yml',
        'shorthand': 'oLSTMd1024h1064'
    }, 
    'lstm-d128-h256': {
        'module': layers.LSTM,
        'cfg': 'configs/layers/lstm-d128-h256.yml',
        'shorthand': 'oLSTMd128h256'
    },  
    'lstm-d128-h320': {
        'module': layers.LSTM,
        'cfg': 'configs/layers/lstm-d128-h320.yml',
        'shorthand': 'oLSTMd128h320'
    },   
    'lstm-d128-h384': {
        'module': layers.LSTM,
        'cfg': 'configs/layers/lstm-d128-h384.yml',
        'shorthand': 'oLSTMd128h384'
    },    
    'lstm-d128-h448': {
        'module': layers.LSTM,
        'cfg': 'configs/layers/lstm-d128-h448.yml',
        'shorthand': 'oLSTMd128h448'
    },  # begin pdssm
    'pdssm-d128-h64': {
        'module': layers.PDSSM,
        'cfg': 'configs/layers/pdssm-d128-h64.yml',
        'shorthand': 'PDSSMd128h64'
    },
    'pdssm-d128-h128': {
        'module': layers.PDSSM,
        'cfg': 'configs/layers/pdssm-d128-h128.yml',
        'shorthand': 'PDSSMd128h128'
    },
    'pdssm-d128-h256': {
        'module': layers.PDSSM,
        'cfg': 'configs/layers/pdssm-d128-h256.yml',
        'shorthand': 'PDSSMd128h256'
    },
    'pdssm-d1024-h104': {
        'module': layers.PDSSM,
        'cfg': 'configs/layers/pdssm-d1024-h104.yml',
        'shorthand': 'PDSSMd1024h104'
    },
    'pdssm-d1024-h632': {
        'module': layers.PDSSM,
        'cfg': 'configs/layers/pdssm-d1024-h632.yml',
        'shorthand': 'PDSSMd1024h632'
    },  # Mamba2 (flash-linear-attention / Triton backend)
    'mamba2-fla-d128': {
        'module': layers.Mamba2fla,
        'cfg': 'configs/layers/mamba2-fla-d128.yml',
        'shorthand': 'mamba2'
    },
    # larger Mamba2 (expand=6) tuned to the ~0.33M iso-parameter budget
    'mamba2-fla-d128-iso': {
        'module': layers.Mamba2fla,
        'cfg': 'configs/layers/mamba2-fla-d128-iso.yml',
        'shorthand': 'mamba2iso'
    },
    # Mamba2 (expand=20) tuned to the ~1.0M iso-parameter budget
    'mamba2-fla-d128-iso1m': {
        'module': layers.Mamba2fla,
        'cfg': 'configs/layers/mamba2-fla-d128-iso1m.yml',
        'shorthand': 'mamba2iso1m'
    },
    # Mamba2 (d=1024, expand=3) tuned to the ~10M iso-parameter budget
    'mamba2-fla-d1024-iso10m': {
        'module': layers.Mamba2fla,
        'cfg': 'configs/layers/mamba2-fla-d1024-iso10m.yml',
        'shorthand': 'mamba2d1024iso10m'
    },  # begin bdlru                                       
    'bdlru-sel-wd1-d128-h16': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd1-d128-h16.yml',
        'shorthand': 'BDLRUselwd1d128h16'
    },        
    'bdlru-sel-wd1-d128-h21': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd1-d128-h21.yml',
        'shorthand': 'BDLRUselwd1d128h21'
    },        
    'bdlru-sel-wd1-d128-h26': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd1-d128-h26.yml',
        'shorthand': 'BDLRUselwd1d128h26'
    },        
    'bdlru-sel-wd1-d128-h32': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd1-d128-h32.yml',
        'shorthand': 'BDLRUselwd1d128h32'
    },        
    'bdlru-sel-wd1-d128-h43': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd1-d128-h43.yml',
        'shorthand': 'BDLRUselwd1d128h43'
    },        
    'bdlru-sel-wd1-d128-h64': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd1-d128-h64.yml',
        'shorthand': 'BDLRUselwd1d128h64'
    },        
    'bdlru-sel-wd1-d128-h96': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd1-d128-h96.yml',
        'shorthand': 'BDLRUselwd1d128h96'
    },     
    'bdlru-sel-wd1-d128-h128': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd1-d128-h128.yml',
        'shorthand': 'BDLRUselwd1d128h128'
    },     
    'bdlru-sel-wd1-d128-h160': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd1-d128-h160.yml',
        'shorthand': 'BDLRUselwd1d128h160'
    },     
    'bdlru-sel-wd1-d128-h192': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd1-d128-h192.yml',
        'shorthand': 'BDLRUselwd1d128h192'
    },      
    'bdlru-sel-wd1-d128-h256': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd1-d128-h256.yml',
        'shorthand': 'BDLRUselwd1d128h256'
    },      
    'bdlru-sel-wd1-d128-h384': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd1-d128-h384.yml',
        'shorthand': 'BDLRUselwd1d128h384'
    },      
    'bdlru-sel-wd1-d128-h512': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd1-d128-h512.yml',
        'shorthand': 'BDLRUselwd1d128h512'
    },      
    'bdlru-sel-wd1-d128-h768': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd1-d128-h768.yml',
        'shorthand': 'BDLRUselwd1d128h768'
    },      
    'bdlru-sel-wd1-d128-h1024': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd1-d128-h1024.yml',
        'shorthand': 'BDLRUselwd1d128h1024'
    }, # end         
    'bdlru-sel-wd2-d128-h8': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd2-d128-h8.yml',
        'shorthand': 'BDLRUselwd2d128h8'
    },          
    'bdlru-sel-wd2-d128-h16': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd2-d128-h16.yml',
        'shorthand': 'BDLRUselwd2d128h16'
    },        
    'bdlru-sel-wd2-d128-h21': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd2-d128-h21.yml',
        'shorthand': 'BDLRUselwd2d128h21'
    },        
    'bdlru-sel-wd2-d128-h26': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd2-d128-h26.yml',
        'shorthand': 'BDLRUselwd2d128h26'
    },        
    'bdlru-sel-wd2-d128-h32': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd2-d128-h32.yml',
        'shorthand': 'BDLRUselwd2d128h32'
    },        
    'bdlru-sel-wd2-d128-h43': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd2-d128-h43.yml',
        'shorthand': 'BDLRUselwd2d128h43'
    },        
    'bdlru-sel-wd2-d128-h64': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd2-d128-h64.yml',
        'shorthand': 'BDLRUselwd2d128h64'
    },        
    'bdlru-sel-wd2-d128-h96': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd2-d128-h96.yml',
        'shorthand': 'BDLRUselwd2d128h96'
    },     
    'bdlru-sel-wd2-d128-h128': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd2-d128-h128.yml',
        'shorthand': 'BDLRUselwd2d128h128'
    },     
    'bdlru-sel-wd2-d128-h160': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd2-d128-h160.yml',
        'shorthand': 'BDLRUselwd2d128h160'
    },     
    'bdlru-sel-wd2-d128-h192': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd2-d128-h192.yml',
        'shorthand': 'BDLRUselwd2d128h192'
    },     
    'bdlru-sel-wd2-d128-h256': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd2-d128-h256.yml',
        'shorthand': 'BDLRUselwd2d128h256'
    },     
    'bdlru-sel-wd2-d128-h768': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd2-d128-h768.yml',
        'shorthand': 'BDLRUselwd2d128h768'
    },     
    'bdlru-sel-wd1-d1024-h240': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd1-d1024-h240.yml',
        'shorthand': 'BDLRUselwd1d1024h240'
    },     
    'bdlru-sel-wd2-d1024-h96': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd2-d1024-h96.yml',
        'shorthand': 'BDLRUselwd2d1024h96'
    },     
    'bdlru-sel-wd1-d1024-h2432': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd1-d1024-h2432.yml',
        'shorthand': 'BDLRUselwd1d1024h2432'
    },     
    'bdlru-sel-wd2-d1024-h976': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd2-d1024-h976.yml',
        'shorthand': 'BDLRUselwd2d1024h976'
    },     
    'bdlru-sel-wd2-d128-h320': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd2-d128-h320.yml',
        'shorthand': 'BDLRUselwd2d128h320'
    },      
    'bdlru-sel-wd2-d128-h384': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd2-d128-h384.yml',
        'shorthand': 'BDLRUselwd2d128h384'
    },      
    'bdlru-sel-wd2-d128-h512': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd2-d128-h512.yml',
        'shorthand': 'BDLRUselwd2d128h512'
    }, # end        
    'bdlru-sel-wd3-d128-h4': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd3-d128-h4.yml',
        'shorthand': 'BDLRUselwd3d128h4'
    },         
    'bdlru-sel-wd3-d128-h8': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd3-d128-h8.yml',
        'shorthand': 'BDLRUselwd3d128h8'
    },         
    'bdlru-sel-wd3-d128-h16': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd3-d128-h16.yml',
        'shorthand': 'BDLRUselwd3d128h16'
    },        
    'bdlru-sel-wd3-d128-h21': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd3-d128-h21.yml',
        'shorthand': 'BDLRUselwd3d128h21'
    },        
    'bdlru-sel-wd3-d128-h26': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd3-d128-h26.yml',
        'shorthand': 'BDLRUselwd3d128h26'
    },        
    'bdlru-sel-wd3-d128-h32': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd3-d128-h32.yml',
        'shorthand': 'BDLRUselwd3d128h32'
    },        
    'bdlru-sel-wd3-d128-h43': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd3-d128-h43.yml',
        'shorthand': 'BDLRUselwd3d128h43'
    },        
    'bdlru-sel-wd3-d128-h64': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd3-d128-h64.yml',
        'shorthand': 'BDLRUselwd3d128h64'
    },        
    'bdlru-sel-wd3-d128-h96': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd3-d128-h96.yml',
        'shorthand': 'BDLRUselwd3d128h96'
    },     
    'bdlru-sel-wd3-d128-h128': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd3-d128-h128.yml',
        'shorthand': 'BDLRUselwd3d128h128'
    },     
    'bdlru-sel-wd3-d128-h160': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd3-d128-h160.yml',
        'shorthand': 'BDLRUselwd3d128h160'
    },     
    'bdlru-sel-wd3-d128-h192': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd3-d128-h192.yml',
        'shorthand': 'BDLRUselwd3d128h192'
    },      
    'bdlru-sel-wd3-d128-h256': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd3-d128-h256.yml',
        'shorthand': 'BDLRUselwd3d128h256'
    },      
    'bdlru-sel-wd3-d128-h320': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd3-d128-h320.yml',
        'shorthand': 'BDLRUselwd3d128h320'
    },      
    'bdlru-sel-wd3-d128-h384': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd3-d128-h384.yml',
        'shorthand': 'BDLRUselwd3d128h384'
    }, # end        
    'bdlru-sel-wd4-d128-h4': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd4-d128-h4.yml',
        'shorthand': 'BDLRUselwd4d128h4'
    },         
    'bdlru-sel-wd4-d128-h8': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd4-d128-h8.yml',
        'shorthand': 'BDLRUselwd4d128h8'
    },       
    'bdlru-sel-wd4-d128-h16': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd4-d128-h16.yml',
        'shorthand': 'BDLRUselwd4d128h16'
    },        
    'bdlru-sel-wd4-d128-h21': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd4-d128-h21.yml',
        'shorthand': 'BDLRUselwd4d128h21'
    },        
    'bdlru-sel-wd4-d128-h26': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd4-d128-h26.yml',
        'shorthand': 'BDLRUselwd4d128h26'
    },        
    'bdlru-sel-wd4-d128-h32': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd4-d128-h32.yml',
        'shorthand': 'BDLRUselwd4d128h32'
    },        
    'bdlru-sel-wd4-d128-h43': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd4-d128-h43.yml',
        'shorthand': 'BDLRUselwd4d128h43'
    },        
    'bdlru-sel-wd4-d128-h64': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd4-d128-h64.yml',
        'shorthand': 'BDLRUselwd4d128h64'
    },        
    'bdlru-sel-wd4-d128-h96': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd4-d128-h96.yml',
        'shorthand': 'BDLRUselwd4d128h96'
    },     
    'bdlru-sel-wd4-d128-h128': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd4-d128-h128.yml',
        'shorthand': 'BDLRUselwd4d128h128'
    },     
    'bdlru-sel-wd4-d128-h160': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd4-d128-h160.yml',
        'shorthand': 'BDLRUselwd4d128h160'
    },     
    'bdlru-sel-wd4-d128-h192': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd4-d128-h192.yml',
        'shorthand': 'BDLRUselwd4d128h192'
    }, # end           
    'bdlru-sel-wd5-d128-h2': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd5-d128-h2.yml',
        'shorthand': 'BDLRUselwd5d128h2'
    },           
    'bdlru-sel-wd5-d128-h3': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd5-d128-h3.yml',
        'shorthand': 'BDLRUselwd5d128h3'
    },         
    'bdlru-sel-wd5-d128-h4': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd5-d128-h4.yml',
        'shorthand': 'BDLRUselwd5d128h4'
    },         
    'bdlru-sel-wd5-d128-h8': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd5-d128-h8.yml',
        'shorthand': 'BDLRUselwd5d128h8'
    },          
    'bdlru-sel-wd5-d128-h16': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd5-d128-h16.yml',
        'shorthand': 'BDLRUselwd5d128h16'
    },        
    'bdlru-sel-wd5-d128-h21': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd5-d128-h21.yml',
        'shorthand': 'BDLRUselwd5d128h21'
    },        
    'bdlru-sel-wd5-d128-h26': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd5-d128-h26.yml',
        'shorthand': 'BDLRUselwd5d128h26'
    },        
    'bdlru-sel-wd5-d128-h32': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd5-d128-h32.yml',
        'shorthand': 'BDLRUselwd5d128h32'
    },        
    'bdlru-sel-wd5-d128-h43': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd5-d128-h43.yml',
        'shorthand': 'BDLRUselwd5d128h43'
    },        
    'bdlru-sel-wd5-d128-h64': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd5-d128-h64.yml',
        'shorthand': 'BDLRUselwd5d128h64'
    },        
    'bdlru-sel-wd5-d128-h96': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd5-d128-h96.yml',
        'shorthand': 'BDLRUselwd5d128h96'
    },     
    'bdlru-sel-wd5-d128-h128': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd5-d128-h128.yml',
        'shorthand': 'BDLRUselwd5d128h128'
    },     
    'bdlru-sel-wd5-d128-h160': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd5-d128-h160.yml',
        'shorthand': 'BDLRUselwd5d128h160'
    },     
    'bdlru-sel-wd5-d128-h192': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd5-d128-h192.yml',
        'shorthand': 'BDLRUselwd5d128h192'
    },      
    'bdlru-sel-wd1-d128-h256': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd1-d128-h256.yml',
        'shorthand': 'BDLRUselwd1d128h256'
    },         
    'bdlru-sel-wd1-d128-h384': {
        'module': layers.BDLRU_sel,
        'cfg': 'configs/layers/bdlru-sel-wd1-d128-h384.yml',
        'shorthand': 'BDLRUselwd1d128h384'
    },       
    # next are tanhl1       
    # next are relul1       
    # next are hlru                                       
    'hlru-sel-wd1-d128-h16': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd1-d128-h16.yml',
        'shorthand': 'HLRUselwd1d128h16'
    },        
    'hlru-sel-wd1-d128-h21': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd1-d128-h21.yml',
        'shorthand': 'HLRUselwd1d128h21'
    },        
    'hlru-sel-wd1-d128-h26': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd1-d128-h26.yml',
        'shorthand': 'HLRUselwd1d128h26'
    },        
    'hlru-sel-wd1-d128-h32': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd1-d128-h32.yml',
        'shorthand': 'HLRUselwd1d128h32'
    },        
    'hlru-sel-wd1-d128-h43': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd1-d128-h43.yml',
        'shorthand': 'HLRUselwd1d128h43'
    },        
    'hlru-sel-wd1-d128-h64': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd1-d128-h64.yml',
        'shorthand': 'HLRUselwd1d128h64'
    },        
    'hlru-sel-wd1-d128-h96': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd1-d128-h96.yml',
        'shorthand': 'HLRUselwd1d128h96'
    },     
    'hlru-sel-wd1-d128-h128': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd1-d128-h128.yml',
        'shorthand': 'HLRUselwd1d128h128'
    },     
    'hlru-sel-wd1-d128-h160': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd1-d128-h160.yml',
        'shorthand': 'HLRUselwd1d128h160'
    },     
    'hlru-sel-wd1-d128-h192': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd1-d128-h192.yml',
        'shorthand': 'HLRUselwd1d128h192'
    }, 
    'hlru-sel-wd1-d128-h256': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd1-d128-h256.yml',
        'shorthand': 'HLRUselwd1d128h256'
    },         
    'hlru-sel-wd1-d128-h384': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd1-d128-h384.yml',
        'shorthand': 'HLRUselwd1d128h384'
    },       
    'hlru-sel-wd1-d128-h512': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd1-d128-h512.yml',
        'shorthand': 'HLRUselwd1d128h512'
    },      
    'hlru-sel-wd1-d128-h768': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd1-d128-h768.yml',
        'shorthand': 'HLRUselwd1d128h768'
    },      
    'hlru-sel-wd1-d128-h1024': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd1-d128-h1024.yml',
        'shorthand': 'HLRUselwd1d128h1024'
    },      
    'hlru-sel-wd1-d128-h1536': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd1-d128-h1536.yml',
        'shorthand': 'HLRUselwd1d128h1536'
    }, # end         
    'hlru-sel-wd2-d128-h8': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd2-d128-h8.yml',
        'shorthand': 'HLRUselwd2d128h8'
    },          
    'hlru-sel-wd2-d128-h16': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd2-d128-h16.yml',
        'shorthand': 'HLRUselwd2d128h16'
    },        
    'hlru-sel-wd2-d128-h21': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd2-d128-h21.yml',
        'shorthand': 'HLRUselwd2d128h21'
    },        
    'hlru-sel-wd2-d128-h26': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd2-d128-h26.yml',
        'shorthand': 'HLRUselwd2d128h26'
    },        
    'hlru-sel-wd2-d128-h32': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd2-d128-h32.yml',
        'shorthand': 'HLRUselwd2d128h32'
    },        
    'hlru-sel-wd2-d128-h43': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd2-d128-h43.yml',
        'shorthand': 'HLRUselwd2d128h43'
    },        
    'hlru-sel-wd2-d128-h64': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd2-d128-h64.yml',
        'shorthand': 'HLRUselwd2d128h64'
    },        
    'hlru-sel-wd2-d128-h96': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd2-d128-h96.yml',
        'shorthand': 'HLRUselwd2d128h96'
    },     
    'hlru-sel-wd2-d128-h128': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd2-d128-h128.yml',
        'shorthand': 'HLRUselwd2d128h128'
    },     
    'hlru-sel-wd2-d128-h160': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd2-d128-h160.yml',
        'shorthand': 'HLRUselwd2d128h160'
    },     
    'hlru-sel-wd2-d128-h192': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd2-d128-h192.yml',
        'shorthand': 'HLRUselwd2d128h192'
    },     
    'hlru-sel-wd2-d128-h256': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd2-d128-h256.yml',
        'shorthand': 'HLRUselwd2d128h256'
    },     
    'hlru-sel-wd2-d128-h384': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd2-d128-h384.yml',
        'shorthand': 'HLRUselwd2d128h384'
    },      
    'hlru-sel-wd2-d128-h512': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd2-d128-h512.yml',
        'shorthand': 'HLRUselwd2d128h512'
    },       
    'hlru-sel-wd2-d128-h768': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd2-d128-h768.yml',
        'shorthand': 'HLRUselwd2d128h768'
    },       
    'hlru-sel-wd2-d128-h1024': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd2-d128-h1024.yml',
        'shorthand': 'HLRUselwd2d128h1024'
    }, # end        
    'hlru-sel-wd3-d128-h4': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd3-d128-h4.yml',
        'shorthand': 'HLRUselwd3d128h4'
    },         
    'hlru-sel-wd3-d128-h8': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd3-d128-h8.yml',
        'shorthand': 'HLRUselwd3d128h8'
    },         
    'hlru-sel-wd3-d128-h16': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd3-d128-h16.yml',
        'shorthand': 'HLRUselwd3d128h16'
    },        
    'hlru-sel-wd3-d128-h21': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd3-d128-h21.yml',
        'shorthand': 'HLRUselwd3d128h21'
    },        
    'hlru-sel-wd3-d128-h26': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd3-d128-h26.yml',
        'shorthand': 'HLRUselwd3d128h26'
    },        
    'hlru-sel-wd3-d128-h32': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd3-d128-h32.yml',
        'shorthand': 'HLRUselwd3d128h32'
    },        
    'hlru-sel-wd3-d128-h43': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd3-d128-h43.yml',
        'shorthand': 'HLRUselwd3d128h43'
    },        
    'hlru-sel-wd3-d128-h64': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd3-d128-h64.yml',
        'shorthand': 'HLRUselwd3d128h64'
    },        
    'hlru-sel-wd3-d128-h96': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd3-d128-h96.yml',
        'shorthand': 'HLRUselwd3d128h96'
    },     
    'hlru-sel-wd3-d128-h128': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd3-d128-h128.yml',
        'shorthand': 'HLRUselwd3d128h128'
    },     
    'hlru-sel-wd3-d128-h160': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd3-d128-h160.yml',
        'shorthand': 'HLRUselwd3d128h160'
    },     
    'hlru-sel-wd3-d128-h192': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd3-d128-h192.yml',
        'shorthand': 'HLRUselwd3d128h192'
    },      
    'hlru-sel-wd3-d128-h256': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd3-d128-h256.yml',
        'shorthand': 'HLRUselwd3d128h256'
    },      
    'hlru-sel-wd3-d128-h320': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd3-d128-h320.yml',
        'shorthand': 'HLRUselwd3d128h320'
    },      
    'hlru-sel-wd3-d128-h384': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd3-d128-h384.yml',
        'shorthand': 'HLRUselwd3d128h384'
    },       
    'hlru-sel-wd3-d128-h512': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd3-d128-h512.yml',
        'shorthand': 'HLRUselwd3d128h512'
    },      
    'hlru-sel-wd3-d128-h640': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd3-d128-h640.yml',
        'shorthand': 'HLRUselwd3d128h640'
    },      
    'hlru-sel-wd3-d128-h768': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd3-d128-h768.yml',
        'shorthand': 'HLRUselwd3d128h768'
    },# end        
    'hlru-sel-wd4-d128-h4': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd4-d128-h4.yml',
        'shorthand': 'HLRUselwd4d128h4'
    },         
    'hlru-sel-wd4-d128-h8': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd4-d128-h8.yml',
        'shorthand': 'HLRUselwd4d128h8'
    },       
    'hlru-sel-wd4-d128-h16': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd4-d128-h16.yml',
        'shorthand': 'HLRUselwd4d128h16'
    },        
    'hlru-sel-wd4-d128-h21': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd4-d128-h21.yml',
        'shorthand': 'HLRUselwd4d128h21'
    },        
    'hlru-sel-wd4-d128-h26': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd4-d128-h26.yml',
        'shorthand': 'HLRUselwd4d128h26'
    },        
    'hlru-sel-wd4-d128-h32': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd4-d128-h32.yml',
        'shorthand': 'HLRUselwd4d128h32'
    },        
    'hlru-sel-wd4-d128-h43': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd4-d128-h43.yml',
        'shorthand': 'HLRUselwd4d128h43'
    },        
    'hlru-sel-wd4-d128-h64': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd4-d128-h64.yml',
        'shorthand': 'HLRUselwd4d128h64'
    },        
    'hlru-sel-wd4-d128-h96': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd4-d128-h96.yml',
        'shorthand': 'HLRUselwd4d128h96'
    },     
    'hlru-sel-wd4-d128-h128': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd4-d128-h128.yml',
        'shorthand': 'HLRUselwd4d128h128'
    },     
    'hlru-sel-wd4-d128-h160': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd4-d128-h160.yml',
        'shorthand': 'HLRUselwd4d128h160'
    },     
    'hlru-sel-wd4-d128-h192': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd4-d128-h192.yml',
        'shorthand': 'HLRUselwd4d128h192'
    },     
    'hlru-sel-wd4-d128-h320': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd4-d128-h320.yml',
        'shorthand': 'HLRUselwd4d128h320'
    },     
    'hlru-sel-wd4-d128-h512': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd4-d128-h512.yml',
        'shorthand': 'HLRUselwd4d128h512'
    },     
    'hlru-sel-wd4-d128-h640': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd4-d128-h640.yml',
        'shorthand': 'HLRUselwd4d128h640'
    }, # end           
    'hlru-sel-wd5-d128-h2': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd5-d128-h2.yml',
        'shorthand': 'HLRUselwd5d128h2'
    },           
    'hlru-sel-wd5-d128-h3': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd5-d128-h3.yml',
        'shorthand': 'HLRUselwd5d128h3'
    },         
    'hlru-sel-wd5-d128-h4': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd5-d128-h4.yml',
        'shorthand': 'HLRUselwd5d128h4'
    },         
    'hlru-sel-wd5-d128-h8': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd5-d128-h8.yml',
        'shorthand': 'HLRUselwd5d128h8'
    },          
    'hlru-sel-wd5-d128-h16': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd5-d128-h16.yml',
        'shorthand': 'HLRUselwd5d128h16'
    },        
    'hlru-sel-wd5-d128-h21': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd5-d128-h21.yml',
        'shorthand': 'HLRUselwd5d128h21'
    },        
    'hlru-sel-wd5-d128-h26': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd5-d128-h26.yml',
        'shorthand': 'HLRUselwd5d128h26'
    },        
    'hlru-sel-wd5-d128-h32': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd5-d128-h32.yml',
        'shorthand': 'HLRUselwd5d128h32'
    },        
    'hlru-sel-wd5-d128-h43': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd5-d128-h43.yml',
        'shorthand': 'HLRUselwd5d128h43'
    },        
    'hlru-sel-wd5-d128-h64': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd5-d128-h64.yml',
        'shorthand': 'HLRUselwd5d128h64'
    },        
    'hlru-sel-wd5-d128-h96': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd5-d128-h96.yml',
        'shorthand': 'HLRUselwd5d128h96'
    },     
    'hlru-sel-wd5-d128-h128': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd5-d128-h128.yml',
        'shorthand': 'HLRUselwd5d128h128'
    },     
    'hlru-sel-wd5-d128-h160': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd5-d128-h160.yml',
        'shorthand': 'HLRUselwd5d128h160'
    },     
    'hlru-sel-wd5-d128-h192': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd5-d128-h192.yml',
        'shorthand': 'HLRUselwd5d128h192'
    },     
    'hlru-sel-wd5-d128-h256': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd5-d128-h256.yml',
        'shorthand': 'HLRUselwd5d128h256'
    },     
    'hlru-sel-wd5-d128-h320': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd5-d128-h320.yml',
        'shorthand': 'HLRUselwd5d128h320'
    },     
    'hlru-sel-wd5-d128-h512': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd5-d128-h512.yml',
        'shorthand': 'HLRUselwd5d128h512'
    }, # end             
    # next are old 
    'mamba-v2-d128-s2-e1-c4': {
        'module':layers.MambaV2,
        'cfg': 'configs/layers/mamba-v2-d128-s2-e1-c4.yml',
        'shorthand': 'Mbv2d128s2e1c4'
    },
    'mamba-v2-d128-s2-e2-c4': {
        'module':layers.MambaV2,
        'cfg': 'configs/layers/mamba-v2-d128-s2-e2-c4.yml',
        'shorthand': 'Mbv2d128s2e2c4'
    },
    'mamba-v2-d128-s2-e4-c4': {
        'module':layers.MambaV2,
        'cfg': 'configs/layers/mamba-v2-d128-s2-e4-c4.yml',
        'shorthand': 'Mbv2d128s2e4c4'
    },
    'mamba-v2-d128-s2-e8-c4': {
        'module':layers.MambaV2,
        'cfg': 'configs/layers/mamba-v2-d128-s2-e8-c4.yml',
        'shorthand': 'Mbv2d128s2e8c4'
    },
    'mamba-v2-d128-s2-e12-c4': {
        'module':layers.MambaV2,
        'cfg': 'configs/layers/mamba-v2-d128-s2-e12-c4.yml',
        'shorthand': 'Mbv2d128s2e12c4'
    }, 
    'mamba-v2-d128-s2-e16-c4': {
        'module':layers.MambaV2,
        'cfg': 'configs/layers/mamba-v2-d128-s2-e16-c4.yml',
        'shorthand': 'Mbv2d128s2e16c4'
    }, 
    'mamba-v2-d128-s2-e32-c4': {
        'module':layers.MambaV2,
        'cfg': 'configs/layers/mamba-v2-d128-s2-e32-c4.yml',
        'shorthand': 'Mbv2d128s2e32c4'
    },#r 
    'mamba-v2-d128-s4-e1-c4': {
        'module':layers.MambaV2,
        'cfg': 'configs/layers/mamba-v2-d128-s4-e1-c4.yml',
        'shorthand': 'Mbv2d128s4e1c4'
    },
    'mamba-v2-d128-s4-e2-c4': {
        'module':layers.MambaV2,
        'cfg': 'configs/layers/mamba-v2-d128-s4-e2-c4.yml',
        'shorthand': 'Mbv2d128s4e2c4'
    },
    'mamba-v2-d128-s4-e4-c4': {
        'module':layers.MambaV2,
        'cfg': 'configs/layers/mamba-v2-d128-s4-e4-c4.yml',
        'shorthand': 'Mbv2d128s4e4c4'
    },
    'mamba-v2-d128-s4-e8-c4': {
        'module':layers.MambaV2,
        'cfg': 'configs/layers/mamba-v2-d128-s4-e8-c4.yml',
        'shorthand': 'Mbv2d128s4e8c4'
    },
    'mamba-v2-d128-s4-e12-c4': {
        'module':layers.MambaV2,
        'cfg': 'configs/layers/mamba-v2-d128-s4-e12-c4.yml',
        'shorthand': 'Mbv2d128s4e12c4'
    }, 
    'mamba-v2-d128-s4-e16-c4': {
        'module':layers.MambaV2,
        'cfg': 'configs/layers/mamba-v2-d128-s4-e16-c4.yml',
        'shorthand': 'Mbv2d128s4e16c4'
    }, 
    'mamba-v2-d128-s4-e32-c4': {
        'module':layers.MambaV2,
        'cfg': 'configs/layers/mamba-v2-d128-s4-e32-c4.yml',
        'shorthand': 'Mbv2d128s4e32c4'
    },#r
    'mamba-v2-d128-s8-e1-c4': {
        'module':layers.MambaV2,
        'cfg': 'configs/layers/mamba-v2-d128-s8-e1-c4.yml',
        'shorthand': 'Mbv2d128s8e1c4'
    },
    'mamba-v2-d128-s8-e2-c4': {
        'module':layers.MambaV2,
        'cfg': 'configs/layers/mamba-v2-d128-s8-e2-c4.yml',
        'shorthand': 'Mbv2d128s8e2c4'
    },
    'mamba-v2-d128-s8-e4-c4': {
        'module':layers.MambaV2,
        'cfg': 'configs/layers/mamba-v2-d128-s8-e4-c4.yml',
        'shorthand': 'Mbv2d128s8e4c4'
    },
    'mamba-v2-d128-s8-e8-c4': {
        'module':layers.MambaV2,
        'cfg': 'configs/layers/mamba-v2-d128-s8-e8-c4.yml',
        'shorthand': 'Mbv2d128s8e8c4'
    },
    'mamba-v2-d128-s8-e12-c4': {
        'module':layers.MambaV2,
        'cfg': 'configs/layers/mamba-v2-d128-s8-e12-c4.yml',
        'shorthand': 'Mbv2d128s8e12c4'
    }, 
    'mamba-v2-d128-s8-e16-c4': {
        'module':layers.MambaV2,
        'cfg': 'configs/layers/mamba-v2-d128-s8-e16-c4.yml',
        'shorthand': 'Mbv2d128s8e16c4'
    }, 
    'mamba-v2-d128-s8-e32-c4': {
        'module':layers.MambaV2,
        'cfg': 'configs/layers/mamba-v2-d128-s8-e32-c4.yml',
        'shorthand': 'Mbv2d128s8e32c4'
    }, 
    'mamba-v2-d128-s8-e64-c4': {
        'module':layers.MambaV2,
        'cfg': 'configs/layers/mamba-v2-d128-s8-e64-c4.yml',
        'shorthand': 'Mbv2d128s8e64c4'
    }, #r #r   #r 
    'mamba-v2-d128-s2-e1-c2': {
        'module':layers.MambaV2,
        'cfg': 'configs/layers/mamba-v2-d128-s2-e1-c2.yml',
        'shorthand': 'Mbv2d128s2e1c2'
    },
    'mamba-v2-d128-s2-e2-c2': {
        'module':layers.MambaV2,
        'cfg': 'configs/layers/mamba-v2-d128-s2-e2-c2.yml',
        'shorthand': 'Mbv2d128s2e2c2'
    },
    'mamba-v2-d128-s2-e4-c2': {
        'module':layers.MambaV2,
        'cfg': 'configs/layers/mamba-v2-d128-s2-e4-c2.yml',
        'shorthand': 'Mbv2d128s2e4c2'
    },
    'mamba-v2-d128-s2-e8-c2': {
        'module':layers.MambaV2,
        'cfg': 'configs/layers/mamba-v2-d128-s2-e8-c2.yml',
        'shorthand': 'Mbv2d128s2e8c2'
    },
    'mamba-v2-d128-s2-e12-c2': {
        'module':layers.MambaV2,
        'cfg': 'configs/layers/mamba-v2-d128-s2-e12-c2.yml',
        'shorthand': 'Mbv2d128s2e12c2'
    }, 
    'mamba-v2-d128-s2-e16-c2': {
        'module':layers.MambaV2,
        'cfg': 'configs/layers/mamba-v2-d128-s2-e16-c2.yml',
        'shorthand': 'Mbv2d128s2e16c2'
    }, 
    'mamba-v2-d128-s2-e32-c2': {
        'module':layers.MambaV2,
        'cfg': 'configs/layers/mamba-v2-d128-s2-e32-c2.yml',
        'shorthand': 'Mbv2d128s2e32c2'
    },#r 
    'mamba-v2-d128-s4-e1-c2': {
        'module':layers.MambaV2,
        'cfg': 'configs/layers/mamba-v2-d128-s4-e1-c2.yml',
        'shorthand': 'Mbv2d128s4e1c2'
    },
    'mamba-v2-d128-s4-e2-c2': {
        'module':layers.MambaV2,
        'cfg': 'configs/layers/mamba-v2-d128-s4-e2-c2.yml',
        'shorthand': 'Mbv2d128s4e2c2'
    },
    'mamba-v2-d128-s4-e4-c2': {
        'module':layers.MambaV2,
        'cfg': 'configs/layers/mamba-v2-d128-s4-e4-c2.yml',
        'shorthand': 'Mbv2d128s4e4c2'
    },
    'mamba-v2-d128-s4-e8-c2': {
        'module':layers.MambaV2,
        'cfg': 'configs/layers/mamba-v2-d128-s4-e8-c2.yml',
        'shorthand': 'Mbv2d128s4e8c2'
    },
    'mamba-v2-d128-s4-e12-c2': {
        'module':layers.MambaV2,
        'cfg': 'configs/layers/mamba-v2-d128-s4-e12-c2.yml',
        'shorthand': 'Mbv2d128s4e12c2'
    }, 
    'mamba-v2-d128-s4-e16-c2': {
        'module':layers.MambaV2,
        'cfg': 'configs/layers/mamba-v2-d128-s4-e16-c2.yml',
        'shorthand': 'Mbv2d128s4e16c2'
    }, 
    'mamba-v2-d128-s4-e32-c2': {
        'module':layers.MambaV2,
        'cfg': 'configs/layers/mamba-v2-d128-s4-e32-c2.yml',
        'shorthand': 'Mbv2d128s4e32c2'
    },#r
    'mamba-v2-d128-s8-e1-c2': {
        'module':layers.MambaV2,
        'cfg': 'configs/layers/mamba-v2-d128-s8-e1-c2.yml',
        'shorthand': 'Mbv2d128s8e1c2'
    },
    'mamba-v2-d128-s8-e2-c2': {
        'module':layers.MambaV2,
        'cfg': 'configs/layers/mamba-v2-d128-s8-e2-c2.yml',
        'shorthand': 'Mbv2d128s8e2c2'
    },
    'mamba-v2-d128-s8-e4-c2': {
        'module':layers.MambaV2,
        'cfg': 'configs/layers/mamba-v2-d128-s8-e4-c2.yml',
        'shorthand': 'Mbv2d128s8e4c2'
    },
    'mamba-v2-d128-s8-e8-c2': {
        'module':layers.MambaV2,
        'cfg': 'configs/layers/mamba-v2-d128-s8-e8-c2.yml',
        'shorthand': 'Mbv2d128s8e8c2'
    },
    'mamba-v2-d128-s8-e12-c2': {
        'module':layers.MambaV2,
        'cfg': 'configs/layers/mamba-v2-d128-s8-e12-c2.yml',
        'shorthand': 'Mbv2d128s8e12c2'
    }, 
    'mamba-v2-d128-s8-e16-c2': {
        'module':layers.MambaV2,
        'cfg': 'configs/layers/mamba-v2-d128-s8-e16-c2.yml',
        'shorthand': 'Mbv2d128s8e16c2'
    }, 
    'mamba-v2-d128-s8-e32-c2': {
        'module':layers.MambaV2,
        'cfg': 'configs/layers/mamba-v2-d128-s8-e32-c2.yml',
        'shorthand': 'Mbv2d128s8e32c2'
    }, 
    'mamba-v2-d128-s8-e64-c2': {
        'module':layers.MambaV2,
        'cfg': 'configs/layers/mamba-v2-d128-s8-e64-c2.yml',
        'shorthand': 'Mbv2d128s8e64c2'
    },#r   deltaproduct                  
    'dproduct-orig': {
        'module': layers.dproduct,
        'cfg': 'configs/layers/dproduct-orig.yml',
        'shorthand': 'dproducto'
    },                
    # Clean Householder baselines: only `rank` (= num_householder) varies (2/4/8),
    # head_dim=64, num_heads=8 held fixed. Used as speed baselines vs BD-LRU blocks.
    'dproduct-hh2': {
        'module': layers.dproduct,
        'cfg': 'configs/layers/dproduct-hh2-hd64-nh8.yml',
        'shorthand': 'dproducthh2'
    },
    'dproduct-hh4': {
        'module': layers.dproduct,
        'cfg': 'configs/layers/dproduct-hh4-hd64-nh8.yml',
        'shorthand': 'dproducthh4'
    },
    'dproduct-hh6': {
        'module': layers.dproduct,
        'cfg': 'configs/layers/dproduct-hh6-hd64-nh8.yml',
        'shorthand': 'dproducthh6'
    },
    'dproduct-d1024-hh2': {
        'module': layers.dproduct,
        'cfg': 'configs/layers/dproduct-d1024-hh2-hd16-nh8.yml',
        'shorthand': 'dproductd1024hh2'
    },
    'dproduct-d1024-hh2-10m': {
        'module': layers.dproduct,
        'cfg': 'configs/layers/dproduct-d1024-hh2-hd168-nh8.yml',
        'shorthand': 'dproductd1024hh210m'
    },
    'dproduct-hh8': {
        'module': layers.dproduct,
        'cfg': 'configs/layers/dproduct-hh8-hd64-nh8.yml',
        'shorthand': 'dproducthh8'
    },
    'dproduct-gn-hd16-nh8': {
        'module': layers.dproduct,
        'cfg': 'configs/layers/dproduct-gn-hd16-nh8.yml',
        'shorthand': 'dproductgnhd16nh8'
    },                        
    'dproduct-gn-hd32-nh8': {
        'module': layers.dproduct,
        'cfg': 'configs/layers/dproduct-gn-hd32-nh8.yml',
        'shorthand': 'dproductgnhd32nh8'
    },                         
    'dproduct-gn-hd64-nh8': {
        'module': layers.dproduct,
        'cfg': 'configs/layers/dproduct-gn-hd64-nh8.yml',
        'shorthand': 'dproductgnhd64nh8'
    },                       
    'dproduct-gn-hd128-nh8': {
        'module': layers.dproduct,
        'cfg': 'configs/layers/dproduct-gn-hd128-nh8.yml',
        'shorthand': 'dproductgnhd128nh8'
    },                       
    'dproduct-gn-hd256-nh8': {
        'module': layers.dproduct,
        'cfg': 'configs/layers/dproduct-gn-hd256-nh8.yml',
        'shorthand': 'dproductgnhd256nh8'
    },                      
    'dproduct-gn-hd16-nh4': {
        'module': layers.dproduct,
        'cfg': 'configs/layers/dproduct-gn-hd16-nh4.yml',
        'shorthand': 'dproductgnhd16nh4'
    },                        
    'dproduct-gn-hd32-nh4': {
        'module': layers.dproduct,
        'cfg': 'configs/layers/dproduct-gn-hd32-nh4.yml',
        'shorthand': 'dproductgnhd32nh4'
    },                         
    'dproduct-gn-hd64-nh4': {
        'module': layers.dproduct,
        'cfg': 'configs/layers/dproduct-gn-hd64-nh4.yml',
        'shorthand': 'dproductgnhd64nh4'
    },                       
    'dproduct-gn-hd128-nh4': {
        'module': layers.dproduct,
        'cfg': 'configs/layers/dproduct-gn-hd128-nh4.yml',
        'shorthand': 'dproductgnhd128nh4'
    },                        
    'dproduct-gn-hd256-nh4': {
        'module': layers.dproduct,
        'cfg': 'configs/layers/dproduct-gn-hd256-nh4.yml',
        'shorthand': 'dproductgnhd256nh4'
    },                         
    'dproduct-gn-hd32-nh2': {
        'module': layers.dproduct,
        'cfg': 'configs/layers/dproduct-gn-hd32-nh2.yml',
        'shorthand': 'dproductgnhd32nh2'
    },                         
    'dproduct-gn-hd64-nh2': {
        'module': layers.dproduct,
        'cfg': 'configs/layers/dproduct-gn-hd64-nh2.yml',
        'shorthand': 'dproductgnhd64nh2'
    },                       
    'dproduct-gn-hd128-nh2': {
        'module': layers.dproduct,
        'cfg': 'configs/layers/dproduct-gn-hd128-nh2.yml',
        'shorthand': 'dproductgnhd128nh2'
    },                        
    'dproduct-gn-hd256-nh2': {
        'module': layers.dproduct,
        'cfg': 'configs/layers/dproduct-gn-hd256-nh2.yml',
        'shorthand': 'dproductgnhd256nh2'
    },                        
    'dproduct-gn-hd512-nh2': {
        'module': layers.dproduct,
        'cfg': 'configs/layers/dproduct-gn-hd512-nh2.yml',
        'shorthand': 'dproductgnhd512nh2'
    },   #dprodcut 4                     
    'dproduct-gn4-hd140-nh4': {
        'module': layers.dproduct,
        'cfg': 'configs/layers/dproduct-gn4-hd140-nh4.yml',
        'shorthand': 'dproductgn4hd140nh4'
    },                        
    'dproduct-gn4-hd8-nh4': {
        'module': layers.dproduct,
        'cfg': 'configs/layers/dproduct-gn4-hd8-nh4.yml',
        'shorthand': 'dproductgn4hd8nh4'
    },                          
    'dproduct-gn4-hd4-nh8': {
        'module': layers.dproduct,
        'cfg': 'configs/layers/dproduct-gn4-hd4-nh8.yml',
        'shorthand': 'dproductgn4hd4nh8'
    },                          
    'dproduct-gn4-hd16-nh2': {
        'module': layers.dproduct,
        'cfg': 'configs/layers/dproduct-gn4-hd16-nh2.yml',
        'shorthand': 'dproductgn4hd16nh2'
    },                        
    'dproduct-gn4-hd150-nh4': {
        'module': layers.dproduct,
        'cfg': 'configs/layers/dproduct-gn4-hd150-nh4.yml',
        'shorthand': 'dproductgn4hd150nh4'
    },                       
    'dproduct-gn4-hd8-nh8': {
        'module': layers.dproduct,
        'cfg': 'configs/layers/dproduct-gn4-hd8-nh8.yml',
        'shorthand': 'dproductgn4hd8nh8'
    },                    
    'dproduct-gn4-hd16-nh8': {
        'module': layers.dproduct,
        'cfg': 'configs/layers/dproduct-gn4-hd16-nh8.yml',
        'shorthand': 'dproductgn4hd16nh8'
    },                        
    'dproduct-gn4-hd32-nh8': {
        'module': layers.dproduct,
        'cfg': 'configs/layers/dproduct-gn4-hd32-nh8.yml',
        'shorthand': 'dproductgn4hd32nh8'
    },                         
    'dproduct-gn4-hd64-nh8': {
        'module': layers.dproduct,
        'cfg': 'configs/layers/dproduct-gn4-hd64-nh8.yml',
        'shorthand': 'dproductgn4hd64nh8'
    },                       
    'dproduct-gn4-hd128-nh8': {
        'module': layers.dproduct,
        'cfg': 'configs/layers/dproduct-gn4-hd128-nh8.yml',
        'shorthand': 'dproductgn4hd128nh8'
    },                       
    'dproduct-gn4-hd16-nh4': {
        'module': layers.dproduct,
        'cfg': 'configs/layers/dproduct-gn4-hd16-nh4.yml',
        'shorthand': 'dproductgn4hd16nh4'
    },                        
    'dproduct-gn4-hd32-nh4': {
        'module': layers.dproduct,
        'cfg': 'configs/layers/dproduct-gn4-hd32-nh4.yml',
        'shorthand': 'dproductgn4hd32nh4'
    },                         
    'dproduct-gn4-hd64-nh4': {
        'module': layers.dproduct,
        'cfg': 'configs/layers/dproduct-gn4-hd64-nh4.yml',
        'shorthand': 'dproductgn4hd64nh4'
    },                       
    'dproduct-gn4-hd128-nh4': {
        'module': layers.dproduct,
        'cfg': 'configs/layers/dproduct-gn4-hd128-nh4.yml',
        'shorthand': 'dproductgn4hd128nh4'
    },                        
    'dproduct-gn4-hd256-nh4': {
        'module': layers.dproduct,
        'cfg': 'configs/layers/dproduct-gn4-hd256-nh4.yml',
        'shorthand': 'dproductgn4hd256nh4'
    },                         
    'dproduct-gn4-hd32-nh2': {
        'module': layers.dproduct,
        'cfg': 'configs/layers/dproduct-gn4-hd32-nh2.yml',
        'shorthand': 'dproductgn4hd32nh2'
    },                         
    'dproduct-gn4-hd64-nh2': {
        'module': layers.dproduct,
        'cfg': 'configs/layers/dproduct-gn4-hd64-nh2.yml',
        'shorthand': 'dproductgn4hd64nh2'
    },                       
    'dproduct-gn4-hd128-nh2': {
        'module': layers.dproduct,
        'cfg': 'configs/layers/dproduct-gn4-hd128-nh2.yml',
        'shorthand': 'dproductgn4hd128nh2'
    },                        
    'dproduct-gn4-hd256-nh2': {
        'module': layers.dproduct,
        'cfg': 'configs/layers/dproduct-gn4-hd256-nh2.yml',
        'shorthand': 'dproductgn4hd256nh2'
    },  # deltanet     
    'dnet-orig': {
        'module': layers.dnet,
        'cfg': 'configs/layers/dnet-orig.yml',
        'shorthand': 'dneto'
    },                    
    'dnet-gn-hd192-nh8': {
        'module': layers.dnet,
        'cfg': 'configs/layers/dnet-gn-hd192-nh8.yml',
        'shorthand': 'dnetgnhd192nh8'
    },
    'dnet-d1024-hd24-nh8': {
        'module': layers.dnet,
        'cfg': 'configs/layers/dnet-d1024-hd24-nh8.yml',
        'shorthand': 'dnetd1024hd24nh8'
    },
    'dnet-d1024-hd240-nh8': {
        'module': layers.dnet,
        'cfg': 'configs/layers/dnet-d1024-hd240-nh8.yml',
        'shorthand': 'dnetd1024hd240nh8'
    },
    'dnet-gn-hd160-nh8': {
        'module': layers.dnet,
        'cfg': 'configs/layers/dnet-gn-hd160-nh8.yml',
        'shorthand': 'dnetgnhd160nh8'
    },                        
    'dnet-gn-hd8-nh8': {
        'module': layers.dnet,
        'cfg': 'configs/layers/dnet-gn-hd8-nh8.yml',
        'shorthand': 'dnetgnhd8nh8'
    },                    
    'dnet-gn-hd16-nh8': {
        'module': layers.dnet,
        'cfg': 'configs/layers/dnet-gn-hd16-nh8.yml',
        'shorthand': 'dnetgnhd16nh8'
    },                        
    'dnet-gn-hd32-nh8': {
        'module': layers.dnet,
        'cfg': 'configs/layers/dnet-gn-hd32-nh8.yml',
        'shorthand': 'dnetgnhd32nh8'
    },                         
    'dnet-gn-hd64-nh8': {
        'module': layers.dnet,
        'cfg': 'configs/layers/dnet-gn-hd64-nh8.yml',
        'shorthand': 'dnetgnhd64nh8'
    },                       
    'dnet-gn-hd128-nh8': {
        'module': layers.dnet,
        'cfg': 'configs/layers/dnet-gn-hd128-nh8.yml',
        'shorthand': 'dnetgnhd128nh8'
    },                        
    'dnet-gn-hd256-nh8': {
        'module': layers.dnet,
        'cfg': 'configs/layers/dnet-gn-hd256-nh8.yml',
        'shorthand': 'dnetgnhd256nh8'
    },                       
    'dnet-gn-hd16-nh4': {
        'module': layers.dnet,
        'cfg': 'configs/layers/dnet-gn-hd16-nh4.yml',
        'shorthand': 'dnetgnhd16nh4'
    },                        
    'dnet-gn-hd32-nh4': {
        'module': layers.dnet,
        'cfg': 'configs/layers/dnet-gn-hd32-nh4.yml',
        'shorthand': 'dnetgnhd32nh4'
    },                         
    'dnet-gn-hd64-nh4': {
        'module': layers.dnet,
        'cfg': 'configs/layers/dnet-gn-hd64-nh4.yml',
        'shorthand': 'dnetgnhd64nh4'
    },                       
    'dnet-gn-hd128-nh4': {
        'module': layers.dnet,
        'cfg': 'configs/layers/dnet-gn-hd128-nh4.yml',
        'shorthand': 'dnetgnhd128nh4'
    },                        
    'dnet-gn-hd256-nh4': {
        'module': layers.dnet,
        'cfg': 'configs/layers/dnet-gn-hd256-nh4.yml',
        'shorthand': 'dnetgnhd256nh4'
    },                         
    'dnet-gn-hd32-nh2': {
        'module': layers.dnet,
        'cfg': 'configs/layers/dnet-gn-hd32-nh2.yml',
        'shorthand': 'dnetgnhd32nh2'
    },                         
    'dnet-gn-hd64-nh2': {
        'module': layers.dnet,
        'cfg': 'configs/layers/dnet-gn-hd64-nh2.yml',
        'shorthand': 'dnetgnhd64nh2'
    },                       
    'dnet-gn-hd128-nh2': {
        'module': layers.dnet,
        'cfg': 'configs/layers/dnet-gn-hd128-nh2.yml',
        'shorthand': 'dnetgnhd128nh2'
    },                        
    'dnet-gn-hd256-nh2': {
        'module': layers.dnet,
        'cfg': 'configs/layers/dnet-gn-hd256-nh2.yml',
        'shorthand': 'dnetgnhd256nh2'
    },                        
    'dnet-gn-hd512-nh2': {
        'module': layers.dnet,
        'cfg': 'configs/layers/dnet-gn-hd512-nh2.yml',
        'shorthand': 'dnetgnhd512nh2'
    },           
    'dproduct-gn2-hd64-nh8': {
        'module': layers.dproduct,
        'cfg': 'configs/layers/dproduct-gn2-hd64-nh8.yml',
        'shorthand': 'dproductgn2hd32nh8'
    },                          
    'dproduct-gn3-hd64-nh8': {
        'module': layers.dproduct,
        'cfg': 'configs/layers/dproduct-gn3-hd64-nh8.yml',
        'shorthand': 'dproductgn3hd64nh8'
    },                          
    'dproduct-gn5-hd64-nh8': {
        'module': layers.dproduct,
        'cfg': 'configs/layers/dproduct-gn5-hd64-nh8.yml',
        'shorthand': 'dproductgn5hd64nh8'
    },                            
    'dproduct-gn8-hd64-nh8': {
        'module': layers.dproduct,
        'cfg': 'configs/layers/dproduct-gn8-hd64-nh8.yml',
        'shorthand': 'dproductgn8hd64nh8'
    },   
    # for benchmarking
    # for benchmarking
    # for benchmarking
    # for benchmarking
    # for benchmarking
    'hlru-sel-wd5-d128-h320': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd5-d128-h320.yml',
        'shorthand': 'HLRUselwd5d128h320'
    },              
    # for benchmarking
    # for benchmarking
    # for benchmarking
    'hlru-sel-wd4-d128-h256': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd4-d128-h256.yml',
        'shorthand': 'HLRUselwd4d128h256'
    },
    'hlru-sel-wd4-d128-h768': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd4-d128-h768.yml',
        'shorthand': 'HLRUselwd4d128h768'
    },
    'hlru-sel-wd4-d1024-h96': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd4-d1024-h96.yml',
        'shorthand': 'HLRUselwd4d1024h96'
    },
    'hlru-sel-wd4-d1024-h976': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd4-d1024-h976.yml',
        'shorthand': 'HLRUselwd4d1024h976'
    },     
    'hlru-sel-wd3-d128-h384': {
        'module': layers.HLRU_sel,
        'cfg': 'configs/layers/hlru-sel-wd3-d128-h384.yml',
        'shorthand': 'HLRUselwd3d128h384'
    },            
    # for benchmarking2  
    'dnet-gn-d768-hd256-nh10-e4': {
        'module':  layers.dnet,
        'cfg': 'configs/layers/dnet-gn-d768-hd256-nh10-e4.yml',
        'shorthand': 'dnetgnd768hd256nh10e4'
    },                
    'dproduct-gn4-d768-hd256-nh5-e4': {
        'module': layers.dproduct,
        'cfg': 'configs/layers/dproduct-gn4-d768-hd256-nh5-e4.yml',
        'shorthand': 'dproductgn4d768hd256nh5e4'
    }, # fr hlru add
}


# ---------------------------------------------------------------------------
# Iso-hidden-state benchmark tiers.
#
# These match the *recurrent state size* (total number of scalars carried
# across time), NOT the parameter count. State size per family:
#   LSTM            : hidden_dim (cell state)
#   BD-LRU / H-LRU  : hidden_dim * window_dim   (block-diagonal state = N * m)
#   PDSSM           : hidden_dim                (complex diagonal modes)
#   Mamba2          : d_inner * state_size = (expand * dim) * state_size
#   DeltaNet/Product: num_heads * head_dim * d_v = num_heads * head_dim**2  (expand_v=1)
#                     (#householders / rank do NOT change the state size)
#
# Tiers: d=128 at state in {512,1024,2048,4096}; d=1024 at state 4096 only.
# BD-LRU / H-LRU are emitted at block sizes m in {1, 2, 4, 8, 16} (N = state / m)
# so block size can be compared at matched state. A single generator
# (scripts/gen_iso_state_configs.py) writes the YAML files from this same spec.
# ---------------------------------------------------------------------------
_ISO_STATE_DIM_STATES = {128: [512, 1024, 2048, 4096], 1024: [4096]}
_ISO_STATE_BLOCKS = (1, 2, 4, 8, 16)


def iso_state_layer_specs():
    """Yield (name, module, cfg_path, shorthand, cfg_dict) for every iso-state config."""
    specs = []

    def add(name, module, cfg):
        cfg_path = f'configs/layers/{name}.yml'
        specs.append((name, module, cfg_path, name.replace('-', ''), cfg))

    for dim, states in _ISO_STATE_DIM_STATES.items():
        for s in states:
            add(f'lstm-d{dim}-s{s}', layers.LSTM,
                {'dim': dim, 'hidden_dim': s})
            for m in _ISO_STATE_BLOCKS:
                add(f'bdlru-sel-wd{m}-d{dim}-s{s}', layers.BDLRU_sel,
                    {'dim': dim, 'hidden_dim': s // m, 'window_dim': m, 'implementation': 'orig'})
                add(f'hlru-sel-wd{m}-d{dim}-s{s}', layers.HLRU_sel,
                    {'dim': dim, 'hidden_dim': s // m, 'window_dim': m, 'implementation': 'orig'})
            add(f'pdssm-d{dim}-s{s}', layers.PDSSM,
                {'dim': dim, 'hidden_dim': s, 'dictionary_size': 8,
                 'hidden_D_multiple': 2, 'dropout_rate': 0.01, 'implementation': 'sequential'})
            # Mamba2: state = (expand*dim) * state_size, fix expand=2, head_dim=64
            add(f'mamba2-fla-d{dim}-s{s}', layers.Mamba2fla,
                {'dim': dim, 'head_dim': 64, 'state_size': s // (2 * dim), 'expand': 2,
                 'n_groups': 1, 'conv_kernel': 4, 'chunk_size': 256, 'backend': 'triton'})
            # Delta*: state = num_heads * head_dim**2, fix head_dim=16 -> num_heads = s/256
            add(f'dnet-d{dim}-s{s}', layers.dnet,
                {'dim': dim, 'head_dim': 16, 'num_heads': s // 256,
                 'expand_v': 1, 'gated': True, 'negative': True})
            # DeltaProduct at 2/4/8 householders (rank). Rank does NOT change the
            # state size, so all three sit at the same state -- the intended
            # #householders-vs-block-size comparison against BD-LRU / H-LRU.
            for r in (2, 4, 8):
                add(f'dproduct-hh{r}-d{dim}-s{s}', layers.dproduct,
                    {'dim': dim, 'head_dim': 16, 'num_heads': s // 256, 'expand_v': 1,
                     'gated': True, 'negative': True, 'rank': r})
    return specs


for _name, _module, _cfg, _short, _ in iso_state_layer_specs():
    layer_registry.setdefault(_name, {'module': _module, 'cfg': _cfg, 'shorthand': _short})


# Iso-PARAMETER family sweeps (all families; BD-LRU/H-LRU at blocks {1,2,4,8,16},
# DeltaProduct at householders {2,4,8}). The size knob of each family is solved to
# hit the tier's parameter budget, so the values live in a generated index rather
# than a closed-form. Regenerate with: uv run python -m scripts.gen_iso_param_sweeps
_ISO_PARAM_MODULES = {
    'lstm': layers.LSTM, 'bdlru': layers.BDLRU_sel, 'hlru': layers.HLRU_sel,
    'pdssm': layers.PDSSM, 'mamba2': layers.Mamba2fla, 'dnet': layers.dnet,
    'dproduct': layers.dproduct,
}
def _register_iso_param_layers() -> None:
    import json as _json
    import os as _os
    import warnings as _warnings

    # Inline get_base_path() to avoid importing mad.paths here (paths imports this module).
    _base = _os.getenv('TUNE_ORIG_WORKING_DIR', '')
    _iso_param_index = _os.path.join(_base, 'configs/iso_param_sweeps.json')
    if not _os.path.exists(_iso_param_index):
        return
    try:
        with open(_iso_param_index) as _f:
            for _entry in _json.load(_f):
                layer_registry.setdefault(_entry['name'], {
                    'module': _ISO_PARAM_MODULES.get(_entry['family']),
                    'cfg': _entry['cfg_path'],
                    'shorthand': _entry['name'].replace('-', ''),
                })
    except Exception as exc:  # noqa: BLE001 - registry must import even if the index is corrupt
        _warnings.warn(
            f'Could not load iso-parameter layer index ({_iso_param_index}): {exc}. '
            'Run `uv run python -m scripts.gen_iso_param_sweeps` to regenerate it.',
            stacklevel=2,
        )


_register_iso_param_layers()


def validate_layer_names(names: list[str]) -> None:
    """Raise a clear error if any layer name is unknown or unavailable."""
    import re as _re

    unknown = [name for name in names if name not in layer_registry]
    if unknown:
        hint = ''
        if any(_re.search(r'iso\d+m|iso033m', name) for name in unknown):
            hint = (
                ' Iso-tier layers are generated configs; ensure '
                'configs/iso_param_sweeps.json exists (run: '
                'uv run python -m scripts.gen_iso_param_sweeps).'
            )
        raise ValueError(f'Unknown layer name(s): {unknown}.{hint}')

    unavailable = [
        name for name in names
        if layer_registry[name].get('module') is None
    ]
    if unavailable:
        raise ValueError(
            f'Layer(s) unavailable in this environment (install required extras): '
            f'{unavailable}'
        )


model_registry = {
    'language-model': model.LanguageModel,
    'autoencoder': model.AutoEncoder
}
