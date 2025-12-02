# dataset_builder_optimized.py
"""
Highly optimized dataset builder with performance improvements, 
better maintainability, and extensibility.
"""

import json
import ijson  # For streaming JSON parsing
from pathlib import Path
from typing import List, Dict, Optional, Any, Set
import time
import logging
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
from dataclasses import asdict
import random

# Import our modular components
from data_types import (
    TrainingExample, DatasetStatistics, ProcessingConfig,
    GitHubFileData, DocumentationData, TutorialData, TracesData
)
from data_config import DATASET_BALANCE_CONFIG, FILE_PROCESSING_CONFIG
from generators import (
    BaseGenerator, 
    GitHubGenerator, 
    DocumentationGenerator, 
    TutorialGenerator, 
    SyntheticGenerator
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizedDatasetBuilder:
    """
    Highly optimized dataset builder with modular generators,
    parallel processing, deduplication, and intelligent balancing.
    """
    
    def __init__(self, raw_data_dir: Path = None, config: ProcessingConfig = None):
        self.raw_data_dir = raw_data_dir or Path("data")
        self.output_dir = Path("data")
        self.output_dir.mkdir(exist_ok=True)
        
        # Load configuration
        self.config = config or ProcessingConfig(
            enable_deduplication=FILE_PROCESSING_CONFIG["deduplication_enabled"],
            enable_parallel_processing=FILE_PROCESSING_CONFIG["parallel_processing"],
            max_workers=FILE_PROCESSING_CONFIG["max_workers"],
            max_code_length=FILE_PROCESSING_CONFIG["max_code_length"],
            min_code_length=FILE_PROCESSING_CONFIG["min_code_length"],
            balance_categories=True,
            enhance_with_traces=False
        )
        
        # File paths
        self.github_file = self.raw_data_dir / "github_raw.json"
        self.docs_file = self.raw_data_dir / "docs_raw.json"
        self.tutorials_file = self.raw_data_dir / "tutorials_raw.json"
        self.traces_file = self.raw_data_dir / "traces_raw.json"
        
        # Initialize generators
        self.generators = {
            'github': GitHubGenerator(self.config),
            'documentation': DocumentationGenerator(self.config),
            'tutorial': TutorialGenerator(self.config),
            'synthetic': SyntheticGenerator(self.config)
        }
        
        # Statistics tracking
        self.processing_start_time = 0
        self.seen_examples: Set[str] = set()
        self.category_counts = defaultdict(int)

    def _load_json_streaming(self, file_path: Path) -> List[Dict]:
        """Load JSON with streaming for large files."""
        try:
            if not file_path.exists():
                logger.warning(f"⚠️  File not found: {file_path}")
                return []
            
            file_size = file_path.stat().st_size
            
            # Use streaming for large files (>10MB)
            if file_size > 10 * 1024 * 1024:
                logger.info(f"📥 Streaming large file: {file_path.name} ({file_size // 1024 // 1024}MB)")
                return self._stream_json_array(file_path)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    logger.info(f"📁 Loaded {len(data)} entries from {file_path.name}")
                    return data
                    
        except Exception as e:
            logger.error(f"❌ Error loading {file_path}: {e}")
            return []

    def _stream_json_array(self, file_path: Path) -> List[Dict]:
        """Stream JSON array for memory efficiency."""
        data = []
        try:
            with open(file_path, 'rb') as f:
                parser = ijson.items(f, 'item')
                for item in parser:
                    data.append(item)
                    if len(data) % 1000 == 0:
                        logger.info(f"📥 Streamed {len(data)} items...")
        except Exception as e:
            logger.error(f"❌ Streaming error for {file_path}: {e}")
            
        return data

    def _deduplicate_examples(self, examples: List[TrainingExample]) -> List[TrainingExample]:
        """Remove duplicate examples based on content hash."""
        if not self.config.enable_deduplication:
            return examples
        
        logger.info(f"🔍 Deduplicating {len(examples)} examples...")
        
        unique_examples = []
        seen_hashes = set()
        
        for example in examples:
            if example.source_hash not in seen_hashes:
                unique_examples.append(example)
                seen_hashes.add(example.source_hash)
        
        removed_count = len(examples) - len(unique_examples)
        logger.info(f"✂️  Removed {removed_count} duplicate examples")
        
        return unique_examples

    def _balance_dataset_categories(self, examples: List[TrainingExample]) -> List[TrainingExample]:
        """Balance dataset by category representation."""
        if not self.config.balance_categories:
            return examples
        
        logger.info(f"⚖️  Balancing dataset categories...")
        
        # Group by category
        category_groups = defaultdict(list)
        for example in examples:
            category = example.metadata.get('category', 'unknown')
            category_groups[category].append(example)
        
        # Apply balancing rules
        balanced_examples = []
        min_examples = DATASET_BALANCE_CONFIG["min_examples_per_category"]
        max_examples = DATASET_BALANCE_CONFIG["max_examples_per_category"]
        
        for category, category_examples in category_groups.items():
            example_count = len(category_examples)
            
            if example_count < min_examples:
                # Upsample underrepresented categories
                upsampled = self._upsample_category(category_examples, min_examples)
                balanced_examples.extend(upsampled)
                logger.info(f"📈 Upsampled {category}: {example_count} → {len(upsampled)}")
            elif example_count > max_examples:
                # Downsample overrepresented categories
                downsampled = random.sample(category_examples, max_examples)
                balanced_examples.extend(downsampled)
                logger.info(f"📉 Downsampled {category}: {example_count} → {len(downsampled)}")
            else:
                balanced_examples.extend(category_examples)
        
        logger.info(f"⚖️  Balanced dataset: {len(examples)} → {len(balanced_examples)} examples")
        return balanced_examples

    def _upsample_category(self, examples: List[TrainingExample], target_count: int) -> List[TrainingExample]:
        """Upsample a category by creating variations of existing examples."""
        if len(examples) >= target_count:
            return examples
        
        upsampled = examples.copy()
        needed = target_count - len(examples)
        
        # Create variations by slightly modifying existing examples
        for _ in range(needed):
            base_example = random.choice(examples)
            variation = self._create_example_variation(base_example)
            upsampled.append(variation)
        
        return upsampled

    def _create_example_variation(self, base_example: TrainingExample) -> TrainingExample:
        """Create a variation of an existing example."""
        # Simple variation: modify the instruction slightly
        variations = [
            "Analyze and break down",
            "Decompose and explain", 
            "Examine and structure",
            "Study and decompose"
        ]
        
        new_instruction = base_example.instruction
        for old, new in zip(["Analyze and decompose", "Break down"], 
                           random.choices(variations, k=2)):
            new_instruction = new_instruction.replace(old, new, 1)
        
        # Create new example with modified instruction
        variation = TrainingExample(
            instruction=new_instruction,
            input=base_example.input,
            output=base_example.output,
            metadata={**base_example.metadata, "variation": True}
        )
        
        return variation

    def _enhance_with_execution_traces(self, examples: List[TrainingExample], 
                                     traces_data: List[TracesData]) -> List[TrainingExample]:
        """Enhance examples with real execution trace data."""
        if not self.config.enhance_with_traces or not traces_data:
            return examples
        
        logger.info(f"🔬 Enhancing examples with execution traces...")
        
        # Create mapping of traces by executable/category
        trace_map = {}
        for trace in traces_data:
            key = trace.get("category", "general")
            if key not in trace_map:
                trace_map[key] = []
            trace_map[key].append(trace)
        
        enhanced_examples = []
        for example in examples:
            category = example.metadata.get('category', 'general')
            
            # Find matching traces
            if category in trace_map:
                trace_data = random.choice(trace_map[category])
                enhanced_example = self._apply_trace_data(example, trace_data)
                enhanced_examples.append(enhanced_example)
            else:
                enhanced_examples.append(example)
        
        return enhanced_examples

    def _apply_trace_data(self, example: TrainingExample, 
                         trace_data: TracesData) -> TrainingExample:
        """Apply real trace data to enhance timing estimates."""
        # Extract real timing data
        timing_data = trace_data.get("timing_data", {})
        syscalls = trace_data.get("syscalls", [])
        
        # Update decomposition steps with real timings
        enhanced_steps = []
        for step in example.output.steps:
            # Try to match step action with syscall data
            real_timing = self._match_step_to_trace(step.action, syscalls, timing_data)
            if real_timing:
                step.expected_time = f"{real_timing}μs"
            enhanced_steps.append(step)
        
        # Create enhanced example
        enhanced_output = example.output
        enhanced_output.steps = enhanced_steps
        enhanced_output.metadata["enhanced_with_traces"] = True
        
        return TrainingExample(
            instruction=example.instruction,
            input=example.input,
            output=enhanced_output,
            metadata={**example.metadata, "trace_enhanced": True}
        )

    def _match_step_to_trace(self, step_action: str, syscalls: List[Dict], 
                           timing_data: Dict[str, float]) -> Optional[int]:
        """Match step action to real syscall timing."""
        action_lower = step_action.lower()
        
        # Simple keyword matching
        syscall_mappings = {
            "connect": ["connect", "socket"],
            "send": ["send", "write", "sendto"],
            "receive": ["recv", "read", "recvfrom"],
            "dns": ["getaddrinfo", "gethostbyname"],
            "ssl": ["ssl_read", "ssl_write"]
        }
        
        for keyword, syscall_names in syscall_mappings.items():
            if keyword in action_lower:
                for syscall_name in syscall_names:
                    if syscall_name in timing_data:
                        # Convert to microseconds
                        return int(timing_data[syscall_name] * 1000000)
        
        return None

    def _generate_comprehensive_statistics(self, examples: List[TrainingExample], 
                                         processing_time: float) -> DatasetStatistics:
        """Generate comprehensive dataset statistics."""
        logger.info(f"📊 Generating dataset statistics...")
        
        stats = DatasetStatistics()
        stats.total_examples = len(examples)
        stats.processing_time_ms = processing_time * 1000
        
        # Collect various metrics
        code_lengths = []
        all_functions = []
        all_classes = []
        duplicate_hashes = set()
        
        for example in examples:
            # Source distribution
            source = example.metadata.get('source', 'unknown')
            stats.source_distribution[source] = stats.source_distribution.get(source, 0) + 1
            
            # Category distribution
            category = example.metadata.get('category', 'unknown')
            stats.category_distribution[category] = stats.category_distribution.get(category, 0) + 1
            
            # Complexity distribution
            complexity = example.output.complexity
            stats.complexity_distribution[complexity] = stats.complexity_distribution.get(complexity, 0) + 1
            
            # Resource usage from steps
            for step in example.output.steps:
                resource = step.resource
                stats.resource_usage[resource] = stats.resource_usage.get(resource, 0) + 1
            
            # Code analysis if available
            code_analysis = example.metadata.get('code_analysis')
            if code_analysis:
                if hasattr(code_analysis, 'lines_of_code'):
                    code_lengths.append(code_analysis.lines_of_code)
                if hasattr(code_analysis, 'functions'):
                    all_functions.extend(code_analysis.functions)
                if hasattr(code_analysis, 'classes'):
                    all_classes.extend(code_analysis.classes)
            
            # Track duplicates
            if example.source_hash in duplicate_hashes:
                stats.duplicate_count += 1
            else:
                duplicate_hashes.add(example.source_hash)
        
        # Calculate statistics
        if code_lengths:
            stats.avg_code_length = sum(code_lengths) / len(code_lengths)
            stats.median_code_length = sorted(code_lengths)[len(code_lengths) // 2]
        
        # Most common functions and classes
        function_counter = Counter(all_functions)
        class_counter = Counter(all_classes)
        
        stats.most_common_functions = function_counter.most_common(10)
        stats.most_common_classes = class_counter.most_common(10)
        
        return stats

    def _save_dataset_artifacts(self, examples: List[TrainingExample], 
                               stats: DatasetStatistics) -> None:
        """Save dataset and related artifacts."""
        timestamp = int(time.time())
        
        # Main dataset file
        dataset_path = self.output_dir / f"network_programming_dataset_{timestamp}.json"
        
        # Statistics file
        stats_path = self.output_dir / f"dataset_statistics_{timestamp}.json"
        
        # Category breakdown file
        category_path = self.output_dir / f"category_breakdown_{timestamp}.json"
        
        try:
            # Save main dataset
            serializable_examples = []
            for example in examples:
                example_dict = {
                    "instruction": example.instruction,
                    "input": example.input,
                    "output": {
                        "operation": example.output.operation,
                        "category": example.output.category,
                        "steps": [asdict(step) for step in example.output.steps],
                        "complexity": example.output.complexity,
                        "estimated_total_time": example.output.estimated_total_time,
                        "parallelizable_steps": example.output.parallelizable_steps,
                        "critical_path_length": example.output.critical_path_length,
                        "metadata": example.output.metadata,
                    },
                    "metadata": {
                        **example.metadata,
                        "code_analysis": (
                            asdict(example.metadata["code_analysis"])
                            if "code_analysis" in example.metadata 
                            else None
                        ),
                    },
                    "source_hash": example.source_hash
                }
                serializable_examples.append(example_dict)
            
            with open(dataset_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_examples, f, indent=2, ensure_ascii=False)
            
            # Save statistics
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(stats), f, indent=2, ensure_ascii=False)
            
            # Save category breakdown
            category_breakdown = self._create_category_breakdown(examples)
            with open(category_path, 'w', encoding='utf-8') as f:
                json.dump(category_breakdown, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Dataset saved to {dataset_path}")
            logger.info(f"📊 Statistics saved to {stats_path}")
            logger.info(f"📂 Category breakdown saved to {category_path}")
            
        except Exception as e:
            logger.error(f"❌ Error saving dataset artifacts: {e}")
            raise

    def _create_category_breakdown(self, examples: List[TrainingExample]) -> Dict[str, Any]:
        """Create detailed category breakdown for analysis."""
        breakdown = {
            "categories": {},
            "complexity_by_category": defaultdict(lambda: defaultdict(int)),
            "resource_usage_by_category": defaultdict(lambda: defaultdict(int)),
            "sources_by_category": defaultdict(lambda: defaultdict(int))
        }
        
        for example in examples:
            category = example.metadata.get('category', 'unknown')
            source = example.metadata.get('source', 'unknown')
            complexity = example.output.complexity
            
            # Initialize category if not exists
            if category not in breakdown["categories"]:
                breakdown["categories"][category] = {
                    "count": 0,
                    "avg_steps": 0,
                    "avg_time_us": 0,
                    "examples": []
                }
            
            # Update category stats
            cat_stats = breakdown["categories"][category]
            cat_stats["count"] += 1
            
            # Calculate average steps and time
            step_count = len(example.output.steps)
            total_time_str = example.output.estimated_total_time
            total_time_us = int(total_time_str.replace('μs', '')) if 'μs' in total_time_str else 0
            
            cat_stats["avg_steps"] = ((cat_stats["avg_steps"] * (cat_stats["count"] - 1)) + step_count) / cat_stats["count"]
            cat_stats["avg_time_us"] = ((cat_stats["avg_time_us"] * (cat_stats["count"] - 1)) + total_time_us) / cat_stats["count"]
            
            # Add example summary
            if len(cat_stats["examples"]) < 5:  # Keep only first 5 as examples
                cat_stats["examples"].append({
                    "instruction": example.instruction[:100] + "...",
                    "complexity": complexity,
                    "steps": step_count,
                    "source": source
                })
            
            # Update cross-tabulations
            breakdown["complexity_by_category"][category][complexity] += 1
            breakdown["sources_by_category"][category][source] += 1
            
            # Resource usage by category
            for step in example.output.steps:
                breakdown["resource_usage_by_category"][category][step.resource] += 1
        
        # Convert defaultdicts to regular dicts for JSON serialization
        breakdown["complexity_by_category"] = {k: dict(v) for k, v in breakdown["complexity_by_category"].items()}
        breakdown["resource_usage_by_category"] = {k: dict(v) for k, v in breakdown["resource_usage_by_category"].items()}
        breakdown["sources_by_category"] = {k: dict(v) for k, v in breakdown["sources_by_category"].items()}
        
        return breakdown

    def build_training_dataset(self) -> Optional[List[TrainingExample]]:
        """Build the comprehensive training dataset with all optimizations."""
        self.processing_start_time = time.time()
        logger.info("🏗️  Starting optimized dataset building process...")
        
        # Load raw data from separate files
        logger.info("📥 Loading raw data files...")
        github_data = self._load_json_streaming(self.github_file)
        docs_data = self._load_json_streaming(self.docs_file)
        tutorials_data = self._load_json_streaming(self.tutorials_file)
        traces_data = self._load_json_streaming(self.traces_file)
        
        if not any([github_data, docs_data, tutorials_data]):
            logger.error("❌ No raw data found. Please run data_collector.py first.")
            return None
        
        # Generate examples using modular generators
        all_examples = []
        
        if self.config.enable_parallel_processing:
            logger.info("⚡ Using parallel processing for example generation...")
            with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = {}
                
                if github_data:
                    futures['github'] = executor.submit(
                        self.generators['github'].generate_examples, github_data
                    )
                
                if docs_data:
                    futures['documentation'] = executor.submit(
                        self.generators['documentation'].generate_examples, docs_data
                    )
                
                if tutorials_data:
                    futures['tutorial'] = executor.submit(
                        self.generators['tutorial'].generate_examples, tutorials_data
                    )
                
                # Synthetic examples
                futures['synthetic'] = executor.submit(
                    self.generators['synthetic'].generate_examples
                )
                
                # Collect results
                for generator_name, future in futures.items():
                    try:
                        examples = future.result()
                        all_examples.extend(examples)
                        logger.info(f"✅ {generator_name}: {len(examples)} examples")
                    except Exception as e:
                        logger.error(f"❌ Error in {generator_name} generator: {e}")
        else:
            # Sequential processing
            logger.info("🔄 Using sequential processing...")
            
            if github_data:
                examples = self.generators['github'].generate_examples(github_data)
                all_examples.extend(examples)
            
            if docs_data:
                examples = self.generators['documentation'].generate_examples(docs_data)
                all_examples.extend(examples)
            
            if tutorials_data:
                examples = self.generators['tutorial'].generate_examples(tutorials_data)
                all_examples.extend(examples)
            
            # Add synthetic examples
            synthetic_examples = self.generators['synthetic'].generate_examples()
            all_examples.extend(synthetic_examples)
        
        if not all_examples:
            logger.error("❌ No training examples generated.")
            return None
        
        logger.info(f"📝 Generated {len(all_examples)} raw examples")
        
        # Apply post-processing optimizations
        logger.info("🔧 Applying post-processing optimizations...")
        
        # Deduplication
        all_examples = self._deduplicate_examples(all_examples)
        
        # Category balancing
        all_examples = self._balance_dataset_categories(all_examples)
        
        # Enhance with execution traces if available
        if traces_data:
            all_examples = self._enhance_with_execution_traces(all_examples, traces_data)
        
        # Final shuffle for training
        random.shuffle(all_examples)
        
        # Generate comprehensive statistics
        processing_time = time.time() - self.processing_start_time
        stats = self._generate_comprehensive_statistics(all_examples, processing_time)
        
        # Save dataset and artifacts
        self._save_dataset_artifacts(all_examples, stats)
        
        # Print summary
        self._print_build_summary(stats, processing_time)
        
        return all_examples

    def _print_build_summary(self, stats: DatasetStatistics, processing_time: float) -> None:
        """Print comprehensive build summary."""
        print("\n" + "="*60)
        print("🎉 DATASET BUILD COMPLETE")
        print("="*60)
        print(f"📊 Total Examples: {stats.total_examples}")
        print(f"⏱️  Processing Time: {processing_time:.2f}s")
        print(f"🔄 Duplicates Removed: {stats.duplicate_count}")
        print(f"📈 Average Code Length: {stats.avg_code_length:.1f} lines")
        
        print("\n📂 Source Distribution:")
        for source, count in stats.source_distribution.items():
            percentage = (count / stats.total_examples) * 100
            print(f"  • {source}: {count} ({percentage:.1f}%)")
        
        print("\n🏷️  Category Distribution:")
        for category, count in sorted(stats.category_distribution.items(), 
                                     key=lambda x: x[1], reverse=True)[:10]:
            percentage = (count / stats.total_examples) * 100
            print(f"  • {category}: {count} ({percentage:.1f}%)")
        
        print("\n⚙️  Resource Usage:")
        for resource, count in sorted(stats.resource_usage.items(), 
                                     key=lambda x: x[1], reverse=True):
            percentage = (count / sum(stats.resource_usage.values())) * 100
            print(f"  • {resource}: {count} ({percentage:.1f}%)")
        
        if stats.most_common_functions:
            print(f"\n🔧 Top Functions:")
            for func, count in stats.most_common_functions[:5]:
                print(f"  • {func}: {count}")
        
        print("\n" + "="*60)

if __name__ == "__main__":
    # Configure for optimal performance
    config = ProcessingConfig(
        enable_deduplication=True,
        enable_parallel_processing=True,
        max_workers=4,
        max_code_length=3000,
        min_code_length=100,
        balance_categories=True,
        enhance_with_traces=True
    )
    
    builder = OptimizedDatasetBuilder(config=config)
    dataset = builder.build_training_dataset()
    
    if dataset:
        print(f"\n🎊 SUCCESS: Built dataset with {len(dataset)} examples!")
        print("🚀 Ready for model training!")
    else:
        print("\n❌ FAILED: Could not build dataset. Check logs for details.")