# count_params.py
# Use to confirm the config file is up to speed and accurate parameter count
from model_config import NetworkModelConfig
from transformers import AutoModelForCausalLM

def count_parameters(model):
    """
    Counts the total number of trainable parameters in a PyTorch model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Create the configuration from your custom file
config = NetworkModelConfig().create_config()

# Instantiate a Causal Language Model from the configuration
model = AutoModelForCausalLM.from_config(config)

# Count the parameters
total_params = count_parameters(model)

# Print the result in a readable format
print(f"Total number of parameters: {total_params:,}")
print(f"Total number of parameters (in millions): {total_params / 1_000_000:.2f}M")