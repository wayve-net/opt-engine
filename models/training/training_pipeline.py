# train_network_model.py
import torch
from model_config import NetworkModelConfig
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import os

class NetworkModelTrainer:
    def __init__(self, model_name="network-ops-slm"):
        self.model_name = model_name
        self.device = "cpu"
        
        # AMD CPU optimizations
        torch.set_num_threads(4)  # adjust to your CPU cores
        
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer"""
        
        # Load custom tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("./network_tokenizer")
        
        # Initialize small model
        config = NetworkModelConfig().create_config()
        self.model = AutoModelForCausalLM.from_config(config)
        
        # Print model size
        param_count = sum(p.numel() for p in self.model.parameters())
        print(f"Model parameters: {param_count:,}")
        print(f"Estimated size: {param_count * 4 / (1024**2):.1f}MB (FP32)")
        
    def prepare_dataset(self, text_files):
        """Prepare training dataset"""
        
        # Load and concatenate all text files
        texts = []
        for file_path in text_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                texts.append(f.read())
        
        # Tokenize
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,  # Smaller context for memory efficiency
            return_tensors="pt"
        )
        
        # Create dataset
        dataset = Dataset.from_dict({
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"]
        })
        
        return dataset
    
    def train(self, dataset):
        """Train the model with CPU-optimized settings"""
        
        # Training arguments optimized for CPU
        training_args = TrainingArguments(
            output_dir="./network_model_checkpoints",
            overwrite_output_dir=True,
            num_train_epochs=2,
            per_device_train_batch_size=2,  # Small batch for CPU
            gradient_accumulation_steps=32,  # Effective batch size: 32
            learning_rate=5e-4,
            warmup_steps=500,
            logging_steps=100,
            save_steps=1000,
            eval_steps=1000,
            save_total_limit=3,
            dataloader_num_workers=2,  # Adjust based on CPU cores
            fp16=False,  # CPU doesn't support FP16
            gradient_checkpointing=True,  # Save memory
            max_grad_norm=1.0, # Gradient clipping
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            report_to="wandb"  # optional: experiment tracking
        )
        
        # Data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, not masked LM
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
        )
        
        # Start training
        trainer.train()
        
        # Save final model
        trainer.save_model("./final_network_model")
        self.tokenizer.save_pretrained("./final_network_model")

if __name__ == "__main__":
    trainer = NetworkModelTrainer()
    trainer.setup_model_and_tokenizer()

    # NOTE: You must have a 'data' folder with your text files for this to work.
    # Replace this with the path to your actual training data.
    text_files = ["./data/file1.txt", "./data/file2.txt"] # Example list of files

    dataset = trainer.prepare_dataset(text_files)
    trainer.train(dataset)