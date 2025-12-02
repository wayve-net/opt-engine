# model_config.py
from transformers import AutoConfig
import torch.nn as nn

class NetworkModelConfig:
    """Configuration for AMD A8 PRO-7150B"""
    
    def __init__(self):
        self.vocab_size = 6144 # Smaller vocab
        self.hidden_size = 384 # Reduced from 512
        self.num_hidden_layers = 6 # Reduced from 8
        self.num_attention_heads = 8 # " "
        self.intermediate_size = 1152  # 3x hidden_size
        self.max_position_embeddings = 512 # Smaller context
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1
        
    def create_config(self):
        return AutoConfig.from_dict({
            "model_type": "gpt2", # Better for older hardware
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "intermediate_size": self.intermediate_size,
            "max_position_embeddings": self.max_position_embeddings,
            "hidden_dropout_prob": self.hidden_dropout_prob,
            "attention_probs_dropout_prob": self.attention_probs_dropout_prob,
        })

# Estimated parameters: ~65M (conservative for AMD A8)
