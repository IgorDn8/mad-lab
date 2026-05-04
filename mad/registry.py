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


model_registry = {
    'language-model': model.LanguageModel,
    'autoencoder': model.AutoEncoder
}
