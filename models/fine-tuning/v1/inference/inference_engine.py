# Updated Inference Engine
import torch
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer
import json
import re
import os
import logging
import time
import asyncio
from typing import Dict, List, Any, Optional, Type, Union
from dataclasses import dataclass
from pathlib import Path
from pydantic import BaseModel, ValidationError
from prometheus_client import Histogram, start_http_server
from enum import Enum

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Prometheus metrics
INFERENCE_LATENCY = Histogram(
    'model_inference_seconds',
    'Time spent in model inference',
    ['model_name', 'task_type']
)

# Custom Exceptions
class ModelLoadError(Exception):
    """Exception raised for errors during model loading."""
    pass

class GenerationError(Exception):
    """Exception raised for errors during text generation."""
    pass

# Pydantic schema for network operations
class NetworkStep(BaseModel):
    step: int
    action: str
    expected_time: str
    resource: str

class NetworkSchema(BaseModel):
    operation: str
    steps: List[NetworkStep]

# Centralized data type definition for generic use
class DataType(Enum):
    OHLCV = "ohlcv"
    ORDERBOOK = "orderbook"
    TRADE = "trade"
    NEWS = "news"
    SENTIMENT = "sentiment"
    ONCHAIN = "onchain"
    SOCIAL = "social"
    NETWORK_DECOMPOSITION = "network_decomposition"
    FILE_OPS = "file_ops"
    NATURAL_LANGUAGE = "natural_language"

@dataclass
class ModelConfig:
    """Configuration for each model instance"""
    name: str
    model_path: str
    task_type: str
    schema_validator: Optional[Type[BaseModel]] = None
    generation_config: Optional[Dict] = None
    device: Optional[str] = None

class MultiModelInferenceEngine:
    def __init__(self, config_file: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize multi-model inference engine.
        Args:
            config_file: Optional path to a custom models configuration JSON file.
            device: Optional device override ('cpu', 'cuda', 'mps').
        """
        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        self.model_configs: Dict[str, ModelConfig] = {}
        self.main_device = self._get_best_device(device)
        self.dispatch_map: Dict[str, str] = {} # Now loaded from config
        
        logger.info(f"Main inference device set to: {self.main_device}")
        
        self.load_config(config_file)
    
    def _get_best_device(self, device: Optional[str] = None) -> torch.device:
        """
        Automatically detects and returns the best available device (MPS, CUDA, or CPU).
        Allows for a user-specified override.
        """
        if device:
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    
    def load_config(self, config_file: Optional[str]):
        """
        Load model configurations and dispatch map from JSON file.
        Prioritizes user-defined config, then local, then creates default.
        """
        user_config_path = Path.home() / '.inference_engine' / 'models_config.json'
        
        if config_file:
            path_to_load = Path(config_file)
        elif user_config_path.exists():
            path_to_load = user_config_path
        else:
            path_to_load = Path("models_config.json")
            if not path_to_load.exists():
                self.create_default_config(path_to_load)
                return

        logger.info(f"Loading configuration from: {path_to_load}")
        try:
            with open(path_to_load, 'r') as f:
                configs = json.load(f)
            
            for config_data in configs.get("models", []):
                if config_data.get("task_type") == "decomposition":
                    config_data["schema_validator"] = NetworkSchema
                
                config = ModelConfig(**config_data)
                self.model_configs[config.name] = config
                logger.info(f"Loaded config for model: {config.name}")
            
            # Load the dispatch map
            self.dispatch_map = configs.get("dispatch_map", {})
            logger.info("Loaded model dispatch map.")

        except Exception as e:
            logger.error(f"Failed to load configuration file {path_to_load}: {e}")
            raise ModelLoadError(f"Configuration load failed: {e}")
    
    def create_default_config(self, config_file: Path):
        """Create a default configuration file."""
        default_config = {
            "models": [
                {
                    "name": "network_operations",
                    "model_path": "./codet5-network-teacher",
                    "task_type": "decomposition",
                    "generation_config": {
                        "max_length": 256,
                        "num_beams": 2,
                        "temperature": 0.7,
                        "do_sample": True
                    }
                },
                {
                    "name": "chart_pattern_model",
                    "model_path": "./finetuned-chart-model",
                    "task_type": "structured_data_ohlcv",
                    "generation_config": {}
                },
                {
                    "name": "sentiment_model",
                    "model_path": "./finetuned-sentiment-nlp-model",
                    "task_type": "structured_data_text",
                    "generation_config": {}
                },
                {
                    "name": "onchain_model",
                    "model_path": "./finetuned-onchain-model",
                    "task_type": "structured_data_onchain",
                    "generation_config": {}
                }
            ],
            "dispatch_map": {
                DataType.OHLCV.value: "chart_pattern_model",
                DataType.NEWS.value: "sentiment_model",
                DataType.SOCIAL.value: "sentiment_model",
                DataType.ONCHAIN.value: "onchain_model",
                DataType.NATURAL_LANGUAGE.value: "text_summarization"
            }
        }
        
        try:
            config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            logger.info(f"Created default config file: {config_file}")
            self.load_config(config_file)
        except Exception as e:
            logger.error(f"Failed to create default config file at {config_file}: {e}")
            raise ModelLoadError(f"Default config creation failed: {e}")
    
    def load_model(self, model_name: str):
        """Load a specific model and tokenizer onto its configured or main device."""
        if model_name in self.models:
            return
        
        if model_name not in self.model_configs:
            raise ModelLoadError(f"Model '{model_name}' not found in configuration")
        
        config = self.model_configs[model_name]
        
        if not os.path.exists(config.model_path):
            logger.warning(f"Model path {config.model_path} doesn't exist. Using base Salesforce/codet5-small.")
            model_path = "Salesforce/codet5-small"
        else:
            model_path = config.model_path
        
        model_device = self._get_best_device(config.device)
        logger.info(f"Loading model: {model_name} from {model_path} to device {model_device}")
        
        try:
            # Tokenizers are typically not needed for structured data models
            self.tokenizers[model_name] = AutoTokenizer.from_pretrained(model_path)
            
            if "structured_data" in config.task_type:
                logger.warning("Placeholder: Loading a dummy model for structured data.")
                self.models[model_name] = type('DummyStructuredModel', (object,), {'predict': lambda x: [f"Prediction for {d['data_type']}" for d in x]})()
            else:
                model_class = AutoModelForSeq2SeqLM
                self.models[model_name] = model_class.from_pretrained(
                    model_path,
                    device_map=model_device,
                    torch_dtype=torch.float16
                )
                self.models[model_name].eval()
            
            logger.info(f"Successfully loaded model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise ModelLoadError(f"Failed to load model '{model_name}': {e}")
    
    def unload_model(self, model_name: str):
        """Unload a model to free memory."""
        if model_name in self.models:
            del self.models[model_name]
            if model_name in self.tokenizers:
                del self.tokenizers[model_name]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"Unloaded model: {model_name}")
    
    def generate(self, model_name: str, input_data: Any, **kwargs) -> Dict[str, Any]:
        """Generate output using specified model."""
        if model_name not in self.models:
            self.load_model(model_name)
        
        config = self.model_configs[model_name]
        model = self.models[model_name]
        
        inputs = input_data if isinstance(input_data, list) else [input_data]
        
        if config.task_type.startswith("structured_data"):
            predictions = model.predict(inputs)
            return {"model_name": model_name, "input": inputs, "output": predictions, "task_type": config.task_type}
        
        else:
            formatted_input = self.format_input(input_data, config.task_type)
            tokenizer = self.tokenizers[model_name]
            inputs_tokenized = tokenizer(
                formatted_input, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            ).to(model.device)
            gen_config = config.generation_config or {}
            gen_config.update(kwargs)
            with INFERENCE_LATENCY.labels(model_name=model_name, task_type=config.task_type).time():
                with torch.inference_mode():
                    outputs = model.generate(
                        **inputs_tokenized,
                        **gen_config,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            result = self.post_process_output(generated_text, config)
            return {"model_name": model_name, "input": input_data, "output": result, "task_type": config.task_type}
    
    def batch_generate(self, model_name: str, inputs: List[Any], **kwargs) -> List[Dict]:
        """Generate outputs for multiple inputs using efficient batching."""
        if model_name not in self.models:
            self.load_model(model_name)
        
        config = self.model_configs[model_name]
        model = self.models[model_name]
        
        if config.task_type.startswith("structured_data"):
            predictions = model.predict(inputs)
            return [{"model_name": model_name, "input": i, "output": p, "task_type": config.task_type} for i, p in zip(inputs, predictions)]
        
        else:
            formatted_inputs = [self.format_input(text, config.task_type) for text in inputs]
            tokenizer = self.tokenizers[model_name]
            inputs_tokenized = tokenizer(
                formatted_inputs,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(model.device)
            
            gen_config = config.generation_config or {}
            gen_config.update(kwargs)
            
            with INFERENCE_LATENCY.labels(model_name=model_name, task_type=config.task_type).time():
                with torch.inference_mode():
                    outputs = model.generate(
                        **inputs_tokenized,
                        **gen_config,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
            
            generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            results = []
            for i, text in enumerate(generated_texts):
                results.append({
                    "model_name": model_name,
                    "input": inputs[i],
                    "formatted_input": formatted_inputs[i],
                    "output": self.post_process_output(text, config),
                    "task_type": config.task_type
                })
            
            return results
        
    def get_data_points_by_type(self, data_list: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Groups data points in a list of dictionaries by their 'data_type' key."""
        grouped_data = {}
        for item in data_list:
            data_type = item.get('data_type')
            if data_type:
                if data_type not in grouped_data:
                    grouped_data[data_type] = []
                grouped_data[data_type].append(item)
        return grouped_data

    async def process_batch(self, data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Processes a generic list of data dictionaries, dispatching data to relevant models
        based on the data structure's 'data_type' key.
        """
        all_results = {}
        grouped_data = self.get_data_points_by_type(data_list)

        tasks = []
        for data_type, data_points in grouped_data.items():
            model_name = self.dispatch_map.get(data_type)
            if model_name:
                logger.info(f"Dispatching {len(data_points)} {data_type} data points to {model_name}")
                # Create a task for each model's batch generation
                tasks.append(asyncio.create_task(
                    asyncio.to_thread(self.batch_generate, model_name, data_points)
                ))
            else:
                logger.warning(f"No model configured for data type: {data_type}")
        
        # Await all the inference tasks
        if tasks:
            inference_results = await asyncio.gather(*tasks)
            # Combine results from all models
            for result_list in inference_results:
                if result_list:
                    model_name = result_list[0]['model_name']
                    all_results[model_name] = result_list
        
        return all_results
    
    def list_models(self) -> List[str]:
        return list(self.model_configs.keys())
    
    def get_model_info(self, model_name: str) -> Dict:
        if model_name not in self.model_configs:
            return {"error": f"Model '{model_name}' not found"}
        
        config = self.model_configs[model_name]
        return {
            "name": config.name,
            "model_path": config.model_path,
            "task_type": config.task_type,
            "loaded": model_name in self.models,
            "generation_config": config.generation_config
        }
    
    def post_process_output(self, generated_text: str, config: ModelConfig):
        """Post-process and validate model output based on task type."""
        if config.task_type == "decomposition":
            # Attempt to parse as JSON
            try:
                # Remove markdown code block fences and json specifier
                clean_text = re.sub(r'```json\n|```', '', generated_text, flags=re.MULTILINE).strip()
                result = json.loads(clean_text)
                
                # Validate against the Pydantic schema
                if config.schema_validator:
                    try:
                        config.schema_validator.model_validate(result)
                        return result
                    except ValidationError as e:
                        logger.error(f"Validation error for {config.name}: {e.errors()}")
                        raise GenerationError(f"Output validation failed: {e.errors()}")
                return result
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode JSON from model output: {e}")
                raise GenerationError(f"Invalid JSON output: {e}")
        else:
            return generated_text

    def format_input(self, input_text: str, task_type: str) -> str:
        """Format the input text with a prompt template based on task type."""
        # This is a placeholder for your specific prompt engineering logic
        if task_type == "decomposition":
            return f"Decompose the following operation into a sequence of network steps, providing the expected time and resource for each step, and output the result as a JSON array:\nOperation: {input_text}\nJSON:"
        elif task_type == "summarization":
            return f"Summarize the following text:\nText: {input_text}\nSummary:"
        else:
            return input_text

if __name__ == "__main__":
    start_http_server(8000)
    logger.info("Prometheus metrics server started on port 8000.")
    
    engine = MultiModelInferenceEngine(config_file="models_config.json")
    
    logger.info("Available models:")
    for model in engine.list_models():
        info = engine.get_model_info(model)
        logger.info(f"  - {model}: {info['task_type']} ({'loaded' if info['loaded'] else 'not loaded'})")

    logger.info("\n=== Generic Batch Processing with Data Types ===")
    
    generic_data_batch = [
        {"data_type": "ohlcv", "symbol": "BTCUSDT", "close": 50000.5, "volume": 1234.5},
        {"data_type": "social", "content": "Bitcoin is going to the moon! #BTC"},
        {"data_type": "onchain", "metric": "active_addresses", "value": 1.2e6}
    ]
    
    async def run_batch_example():
        try:
            results = await engine.process_batch(generic_data_batch)
            logger.info("\nProcessed generic batch. Results:")
            for model_name, preds in results.items():
                logger.info(f"  - Results from {model_name}: {preds}")
        except Exception as e:
            logger.error(f"Error processing generic batch: {e}")

    asyncio.run(run_batch_example())