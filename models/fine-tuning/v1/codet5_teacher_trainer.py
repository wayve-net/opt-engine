# Teacher trainer for CodeT5-Small
import torch
from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
import json
import os
import random
import gc

from amd_optimization import setup_amd_optimizations

class CodeT5TeacherTrainer:
    def __init__(self, model_name="Salesforce/codet5-small"):
        self.device = "cpu"  # AMD CPU optimized
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16, 
            device_map=self.device
        )
        # Enable gradient checkpointing for memory efficiency
        self.model.gradient_checkpointing_enable()
        
        # Data collator for seq2seq models
        self.data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            return_tensors="pt"
        )
    
    def prepare_dataset(self, data_path):
        """Load and tokenize dataset for seq2seq format"""
        with open(data_path, 'r') as f:
            data = [json.loads(line) for line in f]
        
        def tokenize_fn(examples):
            """Tokenize for encoder-decoder architecture"""
            inputs = []
            targets = []
            
            for example in examples:
                # Input: "Decompose [input]"
                input_text = f"Decompose {example['input']}"
                inputs.append(input_text)
                
                # Target: JSON output
                target_text = json.dumps(example['output'], separators=(',', ':'))
                targets.append(target_text)
            
            # Tokenize inputs (encoder)
            model_inputs = self.tokenizer(
                inputs,
                max_length=256,  # Shorter for CodeT5-Small
                truncation=True,
                padding=False  # Will be handled by data collator
            )
            
            # Tokenize targets (decoder)
            labels = self.tokenizer(
                targets,
                max_length=256,
                truncation=True,
                padding=False
            )
            
            # Set labels for the model
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        
        # Convert to HuggingFace dataset
        dataset = Dataset.from_list(data)
        tokenized_dataset = dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing dataset"
        )
        return tokenized_dataset
    
    def train(self, train_dataset, output_dir="./codet5-teacher", epochs=3):
        """Optimized training for CodeT5-Small on AMD CPU"""
        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=2,  # CodeT5-Small can handle slightly larger batches
            gradient_accumulation_steps=4,  # Effective batch size of 8
            learning_rate=3e-5,  # Higher learning rate suitable for CodeT5
            weight_decay=0.01,
            fp16=True,  # fp16 for AMD CPU compatibility
            bf16=False,
            logging_steps=5,
            save_steps=50,
            save_total_limit=2,
            evaluation_strategy="no",
            warmup_steps=100,  # More warmup steps for seq2seq
            dataloader_num_workers=0,  # CPU optimization
            remove_unused_columns=False,
            report_to=None,  # Disable wandb/tensorboard
            load_best_model_at_end=False,
            predict_with_generate=True,  # Important for seq2seq models
            generation_max_length=256,
            generation_num_beams=1,  # Greedy decoding for speed
        )
        
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
        )
        
        print("Starting fine-tuning CodeT5-Small on your network operations dataset...")
        print(f"Training examples: {len(train_dataset)}")
        print("CodeT5-Small should train faster than Phi-2 on CPU!")
        
        trainer.train()
        gc.collect()
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"Model saved to {output_dir}")
        return trainer
    
    def generate_decomposition(self, input_text, max_length=256):
        """Generate decomposition for a given input"""
        prompt = f"Decompose {input_text}"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=2,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        try:
            # Try to parse as JSON
            return json.loads(decoded_output)
        except json.JSONDecodeError:
            # Return raw text if not valid JSON
            return {"raw_output": decoded_output}

# Usage with automatic dataset preparation
if __name__ == "__main__":
    setup_amd_optimizations()
    
    # Ensure training data exists
    if not os.path.exists("training_data.jsonl"):
        print("Generating training data from your schema...")
        from data_generator import generate_training_data
        generate_training_data()
    
    trainer = CodeT5TeacherTrainer()
    dataset = trainer.prepare_dataset("training_data.jsonl")
    trainer.train(dataset)
    
    # Test the trained model
    print("\nTesting trained model:")
    test_input = "Configure VLAN 100 on interface GigabitEthernet0/1"
    result = trainer.generate_decomposition(test_input)
    print(f"Input: {test_input}")
    print(f"Output: {json.dumps(result, indent=2)}")