"""
Enhanced Decomposition Strategies with Adaptive Parallelism and Graph Optimization
Includes memoization, resource-aware chunking, and optimized dependency graphs
"""

import math
import hashlib
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from task_analysis_config import TaskCategory, TaskPriority, MEMOIZATION_CONFIG, PARALLELISM_CONFIG

# Global memoization cache
_memoization_cache = {}

@dataclass
class SubTask:
    id: str
    parent_task_id: str
    name: str
    category: TaskCategory
    weight: float
    priority: TaskPriority = TaskPriority.MEDIUM
    properties: Dict = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    estimated_duration: float = 0.0
    resource_requirements: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.properties:
            self.properties = {}
        if not self.dependencies:
            self.dependencies = []
        if not self.resource_requirements:
            self.resource_requirements = {}

class BaseDecompositionStrategy:
    """Enhanced base class with memoization and adaptive features"""
    
    @staticmethod
    def generate_subtask_id(parent_id: str, subtask_name: str, index: int = None) -> str:
        suffix = f"_{index}" if index is not None else ""
        return f"{parent_id}_{subtask_name.lower().replace(' ', '_')}{suffix}"
    
    @staticmethod
    def _generate_cache_key(parent_id: str, params: Dict) -> str:
        """Generate cache key for memoization"""
        param_str = str(sorted(params.items()))
        return hashlib.md5(f"{parent_id}_{param_str}".encode()).hexdigest()
    
    @staticmethod
    def _check_memoization_cache(cache_key: str) -> Optional[Tuple[List[SubTask], List[Tuple[str, str]]]]:
        """Check if decomposition is cached"""
        if not MEMOIZATION_CONFIG['enabled'] or cache_key not in _memoization_cache:
            return None
        
        cached_entry = _memoization_cache[cache_key]
        if time.time() - cached_entry['timestamp'] > MEMOIZATION_CONFIG['ttl_seconds']:
            del _memoization_cache[cache_key]
            return None
        
        return cached_entry['result']
    
    @staticmethod
    def _cache_result(cache_key: str, result: Tuple[List[SubTask], List[Tuple[str, str]]]):
        """Cache decomposition result"""
        if not MEMOIZATION_CONFIG['enabled']:
            return
        
        if len(_memoization_cache) >= MEMOIZATION_CONFIG['max_cache_size']:
            # Remove oldest entry
            oldest_key = min(_memoization_cache.keys(), 
                           key=lambda k: _memoization_cache[k]['timestamp'])
            del _memoization_cache[oldest_key]
        
        _memoization_cache[cache_key] = {
            'result': result,
            'timestamp': time.time()
        }
    
    @staticmethod
    def _calculate_adaptive_chunks(total_items: int, item_complexity: float = 1.0, 
                                 available_resources: int = 8) -> int:
        """Calculate optimal number of chunks based on resources and complexity"""
        min_chunk_size = PARALLELISM_CONFIG['min_chunk_size']
        max_chunks = min(PARALLELISM_CONFIG['max_chunks'], available_resources)
        
        # Adjust for complexity
        effective_items = total_items * item_complexity
        optimal_chunks = min(max_chunks, max(1, effective_items // min_chunk_size))
        
        return int(optimal_chunks * PARALLELISM_CONFIG['load_balancing_factor'])

class VideoDecompositionStrategy(BaseDecompositionStrategy):
    @staticmethod
    def decompose(parent_id: str, params: Dict) -> Tuple[List[SubTask], List[Tuple[str, str]]]:
        cache_key = VideoDecompositionStrategy._generate_cache_key(parent_id, params)
        cached_result = VideoDecompositionStrategy._check_memoization_cache(cache_key)
        if cached_result:
            return cached_result
        
        frame_count = params.get('frame_count', 100)
        keyframe_interval = params.get('keyframe_interval', 10)
        resolution = params.get('resolution', '1080p')
        available_gpus = params.get('available_gpus', 4)
        
        resolution_multiplier = {'720p': 1.0, '1080p': 1.5, '4k': 3.0, '8k': 8.0}.get(resolution, 1.0)
        
        # Adaptive chunking for parallel processing
        frame_chunks = VideoDecompositionStrategy._calculate_adaptive_chunks(
            frame_count, resolution_multiplier, available_gpus
        )
        frames_per_chunk = max(1, frame_count // frame_chunks)
        
        subtasks = []
        edges = []
        
        # Create frame processing chunks
        for chunk_id in range(frame_chunks):
            start_frame = chunk_id * frames_per_chunk
            end_frame = min((chunk_id + 1) * frames_per_chunk, frame_count)
            
            chunk_weight = (end_frame - start_frame) * resolution_multiplier * 0.8
            
            chunk_task = SubTask(
                id=VideoDecompositionStrategy.generate_subtask_id(parent_id, "frame_chunk", chunk_id),
                parent_task_id=parent_id,
                name=f"Process Frames {start_frame}-{end_frame}",
                category=TaskCategory.GPU_TASKS,
                weight=chunk_weight,
                priority=TaskPriority.HIGH,
                properties={
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'resolution': resolution,
                    'chunk_id': chunk_id
                },
                resource_requirements={'gpu_memory': resolution_multiplier * 2}
            )
            subtasks.append(chunk_task)
        
        # Audio processing (parallel with video)
        if params.get('has_audio', True):
            audio_task = SubTask(
                id=VideoDecompositionStrategy.generate_subtask_id(parent_id, "audio"),
                parent_task_id=parent_id,
                name="Process Audio",
                category=TaskCategory.CPU_TASKS,
                weight=1.5,
                priority=TaskPriority.MEDIUM,
                resource_requirements={'cpu_cores': 2}
            )
            subtasks.append(audio_task)
        
        # Optimized merge task
        merge_task = SubTask(
            id=VideoDecompositionStrategy.generate_subtask_id(parent_id, "merge"),
            parent_task_id=parent_id,
            name="Merge Video Stream",
            category=TaskCategory.IO_TASKS,
            weight=2.0 * resolution_multiplier,
            priority=TaskPriority.HIGH,
            dependencies=[task.id for task in subtasks],
            resource_requirements={'memory_gb': resolution_multiplier}
        )
        subtasks.append(merge_task)
        
        # Create dependency edges - all chunks -> merge
        for chunk_task in subtasks[:-1]:  # All except merge task
            edges.append((chunk_task.id, merge_task.id))
        
        result = (subtasks, edges)
        VideoDecompositionStrategy._cache_result(cache_key, result)
        return result

class MLDecompositionStrategy(BaseDecompositionStrategy):
    @staticmethod
    def decompose(parent_id: str, params: Dict) -> Tuple[List[SubTask], List[Tuple[str, str]]]:
        cache_key = MLDecompositionStrategy._generate_cache_key(parent_id, params)
        cached_result = MLDecompositionStrategy._check_memoization_cache(cache_key)
        if cached_result:
            return cached_result
        
        batch_size = params.get('batch_size', 32)
        data_size = params.get('data_size', 1000)
        model_type = params.get('model_type', 'standard')
        available_gpus = params.get('available_gpus', 2)
        
        model_multiplier = {'lightweight': 0.5, 'standard': 1.0, 'heavy': 2.0, 'transformer': 3.0}.get(model_type, 1.0)
        num_batches = math.ceil(data_size / batch_size)
        
        # Adaptive batch grouping
        batch_groups = MLDecompositionStrategy._calculate_adaptive_chunks(
            num_batches, model_multiplier, available_gpus
        )
        batches_per_group = max(1, num_batches // batch_groups)
        
        subtasks = [
            SubTask(
                id=MLDecompositionStrategy.generate_subtask_id(parent_id, "data_prep"),
                parent_task_id=parent_id,
                name="Prepare Data Pipeline",
                category=TaskCategory.IO_TASKS,
                weight=2.0 * model_multiplier,
                priority=TaskPriority.HIGH,
                resource_requirements={'memory_gb': 4}
            )
        ]
        edges = []
        
        # Create batch processing groups
        group_tasks = []
        for group_id in range(batch_groups):
            start_batch = group_id * batches_per_group
            end_batch = min((group_id + 1) * batches_per_group, num_batches)
            
            group_task = SubTask(
                id=MLDecompositionStrategy.generate_subtask_id(parent_id, "inference_group", group_id),
                parent_task_id=parent_id,
                name=f"Inference Group {group_id}",
                category=TaskCategory.GPU_TASKS,
                weight=3.0 * model_multiplier * (end_batch - start_batch),
                priority=TaskPriority.MEDIUM,
                properties={
                    'start_batch': start_batch,
                    'end_batch': end_batch,
                    'model_type': model_type
                },
                dependencies=[subtasks[0].id],
                resource_requirements={'gpu_memory': model_multiplier * 4}
            )
            subtasks.append(group_task)
            group_tasks.append(group_task.id)
            edges.append((subtasks[0].id, group_task.id))
        
        # Results aggregation
        aggregate_task = SubTask(
            id=MLDecompositionStrategy.generate_subtask_id(parent_id, "aggregate"),
            parent_task_id=parent_id,
            name="Aggregate Results",
            category=TaskCategory.CPU_TASKS,
            weight=1.0 * model_multiplier,
            priority=TaskPriority.MEDIUM,
            dependencies=group_tasks,
            resource_requirements={'cpu_cores': 4}
        )
        subtasks.append(aggregate_task)
        
        for group_id in group_tasks:
            edges.append((group_id, aggregate_task.id))
        
        result = (subtasks, edges)
        MLDecompositionStrategy._cache_result(cache_key, result)
        return result

class FinancialDecompositionStrategy(BaseDecompositionStrategy):
    @staticmethod
    def decompose(parent_id: str, params: Dict) -> Tuple[List[SubTask], List[Tuple[str, str]]]:
        portfolio_size = params.get('portfolio_size', 100)
        calculation_type = params.get('calculation_type', 'risk_assessment')
        available_cpus = params.get('available_cpus', 8)
        
        complexity_multiplier = {
            'simple': 0.8, 'risk_assessment': 1.0, 
            'monte_carlo': 2.5, 'var_calculation': 3.0
        }.get(calculation_type, 1.0)
        
        # Resource-constrained chunking
        optimal_chunks = FinancialDecompositionStrategy._calculate_adaptive_chunks(
            portfolio_size, complexity_multiplier, available_cpus
        )
        
        subtasks = [
            SubTask(
                id=FinancialDecompositionStrategy.generate_subtask_id(parent_id, "market_data"),
                parent_task_id=parent_id,
                name="Fetch Market Data",
                category=TaskCategory.IO_TASKS,
                weight=2.0,
                priority=TaskPriority.HIGH,
                resource_requirements={'network_mb': 100}
            )
        ]
        edges = []
        
        # Parallel calculation chunks with load balancing
        calc_tasks = []
        chunk_size = max(1, portfolio_size // optimal_chunks)
        
        for i in range(optimal_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, portfolio_size)
            actual_size = end_idx - start_idx
            
            calc_task = SubTask(
                id=FinancialDecompositionStrategy.generate_subtask_id(parent_id, "calc", i),
                parent_task_id=parent_id,
                name=f"Calculate Portfolio Segment {i+1}",
                category=TaskCategory.CPU_TASKS,
                weight=3.0 * complexity_multiplier * (actual_size / portfolio_size),
                priority=TaskPriority.MEDIUM,
                properties={
                    'segment_id': i,
                    'start_index': start_idx,
                    'end_index': end_idx,
                    'calculation_type': calculation_type
                },
                dependencies=[subtasks[0].id],
                resource_requirements={'cpu_cores': 1, 'memory_gb': 2}
            )
            subtasks.append(calc_task)
            calc_tasks.append(calc_task.id)
            edges.append((subtasks[0].id, calc_task.id))
        
        # Risk aggregation with critical path optimization
        aggregate_task = SubTask(
            id=FinancialDecompositionStrategy.generate_subtask_id(parent_id, "risk_aggregate"),
            parent_task_id=parent_id,
            name="Aggregate Risk Metrics",
            category=TaskCategory.CPU_TASKS,
            weight=1.5 * complexity_multiplier,
            priority=TaskPriority.HIGH,
            dependencies=calc_tasks,
            resource_requirements={'cpu_cores': 2, 'memory_gb': 4}
        )
        subtasks.append(aggregate_task)
        
        for calc_id in calc_tasks:
            edges.append((calc_id, aggregate_task.id))
        
        return subtasks, edges

# Enhanced strategy registry with metadata
DECOMPOSITION_STRATEGIES = {
    'video': VideoDecompositionStrategy,
    'ml': MLDecompositionStrategy, 
    'financial': FinancialDecompositionStrategy,
    'image': VideoDecompositionStrategy,  # Reuse video strategy for image batches
    'database': FinancialDecompositionStrategy  # Reuse for query chunking
}