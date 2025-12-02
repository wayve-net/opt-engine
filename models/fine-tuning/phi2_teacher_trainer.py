# Teacher trainer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
import json
import os
import random
import gc

from amd_optimization import setup_amd_optimizations

class Phi2TeacherTrainer:
    def __init__(self, model_name="microsoft/phi-2"):
        self.device = "cpu"  # AMD CPU optimized
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map=self.device
        )
        # Resize embeddings if needed
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.gradient_checkpointing_enable() # To save memory
    
    def prepare_dataset(self, data_path):
        """Load and tokenize dataset with your exact schema format"""
        with open(data_path, 'r') as f:
            data = [json.loads(line) for line in f]
        
        def tokenize_fn(batch):
            """Tokenize with proper prompt formatting"""
            texts = []
            for example in batch:
                # Format: "Decompose [input]\n{json_output}<eos>"
                prompt = f"Decompose {example['input']}\n"
                target = json.dumps(example['output'], separators=(',', ':'))
                full_text = prompt + target + self.tokenizer.eos_token
                texts.append(full_text)
            
            # Tokenize with consistent padding
            tokenized = self.tokenizer(
                texts, 
                truncation=True, 
                padding=True, 
                max_length=512,
                return_tensors="pt"
            )
            
            # Labels for language modeling (same as input_ids)
            tokenized["labels"] = tokenized["input_ids"].clone()
            return tokenized
        
        # Convert to HuggingFace dataset
        dataset = Dataset.from_list(data)
        tokenized_dataset = dataset.map(
            lambda batch: tokenize_fn([batch]), 
            remove_columns=dataset.column_names,
            desc="Tokenizing dataset"
        )
        return tokenized_dataset
    
    def train(self, train_dataset, output_dir="./phi2-teacher", epochs=3):
        """Optimized training for AMD CPU with fp16"""
        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,  # Effective batch size of 8
            learning_rate=1e-5,  # Conservative for stability
            weight_decay=0.01,
            fp16=True,  # fp16 for AMD CPU compatibility
            bf16=False,
            logging_steps=5,
            save_steps=50,
            save_total_limit=2,
            evaluation_strategy="no",
            warmup_steps=20,
            dataloader_num_workers=0,  # CPU optimization
            remove_unused_columns=False,
            report_to=None,  # Disable wandb/tensorboard
            load_best_model_at_end=False,
        )
        
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer,
        )
        
        print("Starting fine-tuning on your network operations dataset...")
        print(f"Training examples: {len(train_dataset)}")
        print("This will take several hours on CPU - be patient!")
        
        trainer.train()
        gc.collect()
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"Model saved to {output_dir}")
        return trainer

# Usage with automatic dataset preparation
if __name__ == "__main__":
    setup_amd_optimizations()
    # Ensure training data exists
    if not os.path.exists("training_data.jsonl"):
        print("Generating training data from your schema...")
        from data_generator import generate_training_data
        generate_training_data()
    
    trainer = Phi2TeacherTrainer()
    dataset = trainer.prepare_dataset("training_data.jsonl")
    trainer.train(dataset)