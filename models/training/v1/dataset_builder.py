# dataset_builder.py
"""
Dataset builder that processes raw collected data into a structured
training dataset for fine-tuning.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetBuilder:
    """
    Processes raw data collected by DataCollector and builds a structured
    training dataset.
    """
    
    def __init__(self, raw_data_path: Path):
        self.raw_data_path = raw_data_path
        self.output_dir = Path("data")
        self.output_dir.mkdir(exist_ok=True)
    
    def _classify_subtask(self, action: str) -> str:
        """Classifies a sub-task as CPU, GPU, or I/O intensive."""
        action_lower = action.lower()
        io_keywords = ["send", "receive", "read", "write", "download", "upload", "fetch", "connect", "listen", "lookup", "request", "response"]
        if any(keyword in action_lower for keyword in io_keywords):
            return "I/O"
        gpu_keywords = ["render", "visualize", "cryptographic hash", "complex calculation"]
        if any(keyword in action_lower for keyword in gpu_keywords):
            return "GPU"
        cpu_keywords = ["parse", "process", "calculate", "analyze", "decompress", "verify", "validate", "prepare", "generate", "create", "construct"]
        if any(keyword in action_lower for keyword in cpu_keywords):
            return "CPU"
        return "CPU"
        
    def _generate_code_decomposition(self, code: str, category: str) -> Dict:
        """Generate structured decomposition for code"""
        decomposition = {"operation": category.replace('_', ' ').title(), "steps": []}
        if category == "http_requests":
            actions = ["Prepare request headers", "Establish TCP connection", "Send HTTP request", "Receive HTTP response", "Parse response body"]
        elif category == "socket_programming":
            actions = ["Create socket", "Bind address", "Listen for connection", "Accept connection", "Send/receive data"]
        elif category == "api_clients":
            actions = ["Define API endpoint and parameters", "Construct request payload", "Send request over network", "Process JSON response"]
        else:
            actions = ["Analyze code execution flow"]
        for i, action in enumerate(actions, 1):
            decomposition["steps"].append({
                "step": i, "action": action,
                "expected_time": f"{random.randint(50, 500)}ns",
                "resource": self._classify_subtask(action)
            })
        return decomposition
    
    def _create_documentation_examples(self, doc_data: Dict) -> List[Dict]:
        """Create training examples from documentation"""
        examples = []
        content = doc_data.get("content", {})
        if "code_blocks" in content:
            for code_block in content["code_blocks"][:3]:
                if len(code_block.strip()) > 50:
                    doc_decomposition = {
                        "operation": f"Documentation Example: {doc_data.get('title')}",
                        "steps": [{"step": 1, "action": "Understand the documented concept", "expected_time": "100ns", "resource": "CPU"}]
                    }
                    examples.append({"instruction": "Explain and decompose this network programming example", "input": f"Documentation source: {doc_data.get('url')}\nCode:\n{code_block}", "output": doc_decomposition})
        return examples
    
    def _create_tutorial_examples(self, tutorial_data: Dict) -> List[Dict]:
        """Create training examples from tutorial content"""
        examples = []
        content = tutorial_data.get("content", {})
        if "code_examples" in content:
            for code_example in content["code_examples"][:2]:
                if len(code_example.strip()) > 50:
                    tutorial_decomposition = {
                        "operation": f"Tutorial Breakdown: {tutorial_data.get('url')}",
                        "steps": [{"step": 1, "action": "Break down the problem", "expected_time": "50ns", "resource": "CPU"}]
                    }
                    examples.append({"instruction": "Break down this tutorial example into learning steps", "input": f"Tutorial: {tutorial_data.get('url')}\nExample:\n{code_example}", "output": tutorial_decomposition})
        return examples
    
    def generate_synthetic_operations(self) -> List[Dict]:
        """Generate synthetic decompositions for core network ops"""
        operations = {
            "TCP 3-way handshake": ["Send SYN packet", "Receive SYN-ACK packet", "Send final ACK packet"],
            "DNS Lookup": ["Send DNS query to resolver", "Recursive resolver lookup", "Receive DNS response"],
            "TLS Handshake": ["Send ClientHello", "Receive ServerHello and certificate", "Verify server certificate"],
        }
        dataset = []
        for op, actions in operations.items():
            steps = []
            for i, action in enumerate(actions, 1):
                steps.append({"step": i, "action": action, "expected_time": f"{random.randint(50, 500)}ns", "resource": self._classify_subtask(action)})
            dataset.append({"operation": op, "steps": steps})
        return dataset
        
    def _categorize_code(self, code: str) -> str:
        """Enhanced code categorization"""
        code_lower = code.lower()
        categories = {"socket_programming": ["socket", "bind"], "http_requests": ["requests", "urllib"], "async_networking": ["async", "await"], "api_clients": ["api", "rest"], "websockets": ["websocket"], "network_security": ["ssl", "tls"]}
        for category, keywords in categories.items():
            if any(keyword in code_lower for keyword in keywords):
                return category
        return "general_networking"
    
    def _create_code_training_example(self, file_data: Dict) -> Optional[Dict]:
        """Create training example from code file"""
        code = file_data.get("content", "")
        if len(code.strip()) < 100:
            return None
        category = self._categorize_code(code)
        return {"instruction": f"Decompose and analyze this {category} code", "input": f"Repository: {file_data.get('repo')}\nFile: {file_data.get('filename')}\nCode:\n{code[:1000]}...", "output": self._generate_code_decomposition(code, category), "metadata": {"source": "github_scrapling"}}
    
    def build_training_dataset(self):
        """Builds the final training dataset from raw data and saves it."""
        try:
            with open(self.raw_data_path, 'r', encoding='utf-8') as f:
                collected_data = json.load(f)
        except FileNotFoundError:
            logger.error(f"❌ Raw data file not found at {self.raw_data_path}. Please run `data_collector.py` first.")
            return

        training_examples = []
        for file_data in collected_data.get("github_files", []):
            example = self._create_code_training_example(file_data)
            if example:
                training_examples.append(example)
        for doc_data in collected_data.get("documentation", []):
            examples = self._create_documentation_examples(doc_data)
            training_examples.extend(examples)
        for tutorial_data in collected_data.get("tutorials", []):
            examples = self._create_tutorial_examples(tutorial_data)
            training_examples.extend(examples)
        
        synthetic_data = self.generate_synthetic_operations()
        for synthetic_op in synthetic_data:
            training_examples.append({"instruction": f"Decompose network operation: {synthetic_op.get('operation')}", "input": "Synthetic operation details", "output": synthetic_op})
            
        final_dataset_path = self.output_dir / "scrapling_training_dataset.json"
        with open(final_dataset_path, 'w', encoding='utf-8') as f:
            json.dump(training_examples, f, indent=2)
        
        logger.info(f"✅ Training dataset built and saved to {final_dataset_path}")
        return training_examples

if __name__ == "__main__":
    builder = DatasetBuilder(raw_data_path=Path("data") / "collected_raw_data.json")
    builder.build_training_dataset()