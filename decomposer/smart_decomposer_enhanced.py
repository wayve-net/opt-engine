"""
Smart Semantic Decomposer - Queue-Agnostic with Central Repository Integration
Designed for advanced queue systems and multi-decomposer orchestration
"""

import json
import time
import threading
import uuid
from typing import Dict, List, Tuple, Optional, Any, Callable, Protocol
from dataclasses import dataclass, field
import networkx as nx
from collections import defaultdict, deque
from enum import Enum
from difflib import SequenceMatcher
from abc import ABC, abstractmethod

# Queue Interface Protocol for advanced queue systems
class QueueInterface(Protocol):
    def enqueue(self, item: Any, priority: int = 0, routing_key: str = None) -> bool: ...
    def get_queue_depth(self) -> int: ...
    def get_queue_health(self) -> Dict[str, Any]: ...
    def set_flow_control(self, enabled: bool, rate_limit: Optional[int] = None): ...

# Central Repository Interface
class TaskRepositoryInterface(Protocol):
    def register_decomposer(self, decomposer_id: str, capabilities: Dict) -> bool: ...
    def request_task_batch(self, decomposer_id: str, batch_size: int = 10) -> List['RepositoryTask']: ...
    def report_completion(self, task_id: str, result: Dict) -> bool: ...
    def report_error(self, task_id: str, error: str) -> bool: ...
    def update_metrics(self, decomposer_id: str, metrics: Dict) -> bool: ...

@dataclass
class RepositoryTask:
    """Task format from Central Task Repository"""
    task_id: str
    description: str
    parameters: Dict
    priority: int
    deadline: Optional[float] = None
    retry_count: int = 0
    assigned_to: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)

@dataclass
class SubTaskAllocation:
    """Enhanced subtask with queue allocation strategy"""
    subtask: 'SubTask'
    target_queue: str
    allocation_weight: float
    flow_control_priority: int
    batching_eligible: bool = True
    dependencies: List[str] = field(default_factory=list)
    estimated_completion_time: float = 0.0

@dataclass
class DecomposerMetrics:
    """Comprehensive metrics for the decomposer"""
    tasks_processed: int = 0
    subtasks_generated: int = 0
    avg_decomposition_time: float = 0.0
    cache_hit_rate: float = 0.0
    queue_allocation_efficiency: float = 0.0
    error_rate: float = 0.0
    resource_utilization: Dict = field(default_factory=dict)
    last_updated: float = field(default_factory=time.time)

class QueueAllocationManager:
    """Manages smooth allocation to advanced queue systems"""
    
    def __init__(self, queue_configs: Dict[str, Dict]):
        self.queues: Dict[str, QueueInterface] = {}
        self.queue_configs = queue_configs
        self.allocation_strategies = {
            'round_robin': self._round_robin_allocate,
            'load_balanced': self._load_balanced_allocate,
            'priority_aware': self._priority_aware_allocate,
            'batch_optimized': self._batch_optimized_allocate
        }
        self.current_strategy = 'load_balanced'
        self.allocation_history = deque(maxlen=1000)
        self.flow_control_active = {}
        
    def register_queue(self, queue_name: str, queue_instance: QueueInterface):
        """Register an advanced queue instance"""
        self.queues[queue_name] = queue_instance
        self.flow_control_active[queue_name] = False
        
    def allocate_subtasks(self, allocations: List[SubTaskAllocation]) -> Dict[str, List[SubTaskAllocation]]:
        """Intelligently allocate subtasks to queues with smooth distribution"""
        
        # Check queue health and apply flow control
        self._monitor_queue_health()
        
        # Group by target queue
        queue_groups = defaultdict(list)
        for allocation in allocations:
            queue_groups[allocation.target_queue].append(allocation)
        
        # Apply allocation strategy
        strategy = self.allocation_strategies[self.current_strategy]
        optimized_allocation = strategy(queue_groups)
        
        # Execute allocation with batching and flow control
        successful_allocations = {}
        for queue_name, subtask_list in optimized_allocation.items():
            successful_allocations[queue_name] = self._execute_queue_allocation(
                queue_name, subtask_list
            )
            
        return successful_allocations
    
    def _monitor_queue_health(self):
        """Monitor queue health and apply flow control"""
        for queue_name, queue in self.queues.items():
            health = queue.get_queue_health()
            depth = queue.get_queue_depth()
            config = self.queue_configs.get(queue_name, {})
            
            # Apply flow control based on queue depth
            max_depth = config.get('max_depth', 1000)
            flow_control_threshold = config.get('flow_control_threshold', 0.8)
            
            if depth > max_depth * flow_control_threshold:
                if not self.flow_control_active[queue_name]:
                    queue.set_flow_control(True, rate_limit=config.get('flow_control_rate', 10))
                    self.flow_control_active[queue_name] = True
            elif depth < max_depth * 0.5:  # Release flow control
                if self.flow_control_active[queue_name]:
                    queue.set_flow_control(False)
                    self.flow_control_active[queue_name] = False
    
    def _load_balanced_allocate(self, queue_groups: Dict) -> Dict[str, List[SubTaskAllocation]]:
        """Load-balanced allocation considering queue depths"""
        balanced_groups = {}
        
        for queue_name, allocations in queue_groups.items():
            if queue_name not in self.queues:
                continue
                
            queue_depth = self.queues[queue_name].get_queue_depth()
            config = self.queue_configs.get(queue_name, {})
            max_batch_size = config.get('max_batch_size', 50)
            
            # Adjust batch size based on queue depth
            if queue_depth > config.get('max_depth', 1000) * 0.7:
                max_batch_size = max(1, max_batch_size // 2)
            
            # Sort by priority and dependency order
            sorted_allocations = sorted(
                allocations,
                key=lambda x: (x.flow_control_priority, x.allocation_weight),
                reverse=True
            )
            
            balanced_groups[queue_name] = sorted_allocations[:max_batch_size]
            
        return balanced_groups
    
    def _execute_queue_allocation(self, queue_name: str, allocations: List[SubTaskAllocation]) -> List[SubTaskAllocation]:
        """Execute allocation with batching and error handling"""
        if queue_name not in self.queues:
            return []
        
        queue = self.queues[queue_name]
        successful = []
        
        # Batch similar tasks together
        batched_allocations = self._create_batches(allocations)
        
        for batch in batched_allocations:
            for allocation in batch:
                try:
                    success = queue.enqueue(
                        item=allocation.subtask,
                        priority=allocation.flow_control_priority,
                        routing_key=allocation.target_queue
                    )
                    
                    if success:
                        successful.append(allocation)
                        self.allocation_history.append({
                            'queue': queue_name,
                            'task_id': allocation.subtask.id,
                            'timestamp': time.time(),
                            'success': True
                        })
                    else:
                        # Queue full or rejected
                        break
                        
                except Exception as e:
                    self.allocation_history.append({
                        'queue': queue_name,
                        'task_id': allocation.subtask.id,
                        'timestamp': time.time(),
                        'success': False,
                        'error': str(e)
                    })
        
        return successful
    
    def _create_batches(self, allocations: List[SubTaskAllocation]) -> List[List[SubTaskAllocation]]:
        """Create batches of similar tasks for efficient processing"""
        batches = []
        current_batch = []
        
        # Sort by batching eligibility and similarity
        sorted_allocations = sorted(
            allocations,
            key=lambda x: (x.batching_eligible, x.subtask.category.value, x.subtask.weight)
        )
        
        for allocation in sorted_allocations:
            if not allocation.batching_eligible or len(current_batch) >= 10:
                if current_batch:
                    batches.append(current_batch)
                current_batch = [allocation]
            else:
                current_batch.append(allocation)
        
        if current_batch:
            batches.append(current_batch)
            
        return batches

class SmartSemanticDecomposer:
    """Enhanced decomposer for Central Repository and advanced queue integration"""
    
    def __init__(self, decomposer_id: str = None, repository: TaskRepositoryInterface = None):
        self.decomposer_id = decomposer_id or f"decomposer_{uuid.uuid4().hex[:8]}"
        self.repository = repository
        self.queue_manager = None
        self.metrics = DecomposerMetrics()
        
        # Task processing components
        self.analyzer = AdvancedTaskAnalyzer()
        self.processing_thread = None
        self.running = False
        
        # Configuration
        self.batch_size = 5  # Tasks per batch from repository
        self.processing_interval = 0.1  # Seconds between repository requests
        self.max_concurrent_tasks = 10
        
        # Capabilities for repository registration
        self.capabilities = {
            'supported_task_types': ['video', 'ml', 'financial', 'image', 'database'],
            'max_throughput': 100,  # tasks per minute
            'specializations': ['adaptive_parallelism', 'graph_optimization', 'memoization'],
            'resource_requirements': {'min_memory_gb': 4, 'preferred_cpu_cores': 4}
        }
        
    def initialize(self, queue_manager: QueueAllocationManager):
        """Initialize with queue allocation manager"""
        self.queue_manager = queue_manager
        
        # Register with repository if available
        if self.repository:
            success = self.repository.register_decomposer(self.decomposer_id, self.capabilities)
            if not success:
                raise RuntimeError(f"Failed to register decomposer {self.decomposer_id}")
    
    def start_processing(self):
        """Start the main processing loop"""
        if not self.repository or not self.queue_manager:
            raise RuntimeError("Repository and queue manager must be initialized")
            
        self.running = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        
        print(f"Decomposer {self.decomposer_id} started processing")
    
    def stop_processing(self):
        """Stop processing and cleanup"""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        
        # Final metrics report
        if self.repository:
            self.repository.update_metrics(self.decomposer_id, self.metrics.__dict__)
    
    def _processing_loop(self):
        """Main processing loop - requests tasks from repository"""
        while self.running:
            try:
                # Request batch of tasks from repository
                tasks = self.repository.request_task_batch(
                    self.decomposer_id, 
                    self.batch_size
                )
                
                if tasks:
                    self._process_task_batch(tasks)
                else:
                    # No tasks available, brief pause
                    time.sleep(self.processing_interval * 2)
                
                # Update metrics periodically
                self._update_metrics()
                
            except Exception as e:
                print(f"Error in processing loop: {e}")
                time.sleep(self.processing_interval * 5)  # Longer pause on error
    
    def _process_task_batch(self, tasks: List[RepositoryTask]):
        """Process a batch of tasks from the repository"""
        start_time = time.time()
        
        for task in tasks:
            try:
                # Decompose task
                dag = self._decompose_repository_task(task)
                
                # Create allocations for subtasks
                allocations = self._create_subtask_allocations(dag, task)
                
                # Allocate to queues
                successful_allocations = self.queue_manager.allocate_subtasks(allocations)
                
                # Report success to repository
                self.repository.report_completion(task.task_id, {
                    'subtasks_created': sum(len(allocs) for allocs in successful_allocations.values()),
                    'queues_used': list(successful_allocations.keys()),
                    'decomposition_strategy': dag.decomposition_strategy
                })
                
                self.metrics.tasks_processed += 1
                self.metrics.subtasks_generated += len(dag.nodes)
                
            except Exception as e:
                # Report error to repository
                self.repository.report_error(task.task_id, str(e))
                self.metrics.error_rate += 1
        
        # Update processing time metrics
        processing_time = time.time() - start_time
        self.metrics.avg_decomposition_time = (
            (self.metrics.avg_decomposition_time * (self.metrics.tasks_processed - len(tasks)) + 
             processing_time) / self.metrics.tasks_processed
        )
    
    def _decompose_repository_task(self, task: RepositoryTask) -> 'TaskDAG':
        """Decompose a repository task using enhanced analysis"""
        # Enhanced task analysis
        task_type, confidence = self.analyzer.detect_task_type_advanced(
            task.description, 
            task.parameters
        )
        
        # Apply decomposition strategy
        from decomposition_strategies import DECOMPOSITION_STRATEGIES
        strategy = DECOMPOSITION_STRATEGIES.get(task_type)
        
        if not strategy:
            # Use generic strategy
            return self._create_generic_dag(task)
        
        # Execute decomposition
        subtasks, edges = strategy.decompose(task.task_id, task.parameters)
        
        # Calculate performance metrics
        critical_path = self._calculate_critical_path(subtasks, edges)
        speedup = self._estimate_speedup(subtasks)
        
        from smart_decomposer_modular import TaskDAG
        return TaskDAG(
            nodes=subtasks,
            edges=edges,
            critical_path_weight=critical_path,
            estimated_speedup=speedup,
            confidence_score=confidence,
            decomposition_strategy=task_type
        )
    
    def _create_subtask_allocations(self, dag: 'TaskDAG', original_task: RepositoryTask) -> List[SubTaskAllocation]:
        """Create allocation objects for smooth queue distribution"""
        from task_analysis_config import TaskCategory
        
        queue_mapping = {
            TaskCategory.GPU_TASKS: 'gpu_queue',
            TaskCategory.CPU_TASKS: 'cpu_queue',
            TaskCategory.IO_TASKS: 'io_queue'
        }
        
        allocations = []
        for subtask in dag.nodes:
            target_queue = queue_mapping.get(subtask.category, 'cpu_queue')
            
            allocation = SubTaskAllocation(
                subtask=subtask,
                target_queue=target_queue,
                allocation_weight=subtask.weight,
                flow_control_priority=subtask.priority.value if hasattr(subtask.priority, 'value') else 2,
                batching_eligible=subtask.category != TaskCategory.IO_TASKS,
                dependencies=[edge[0] for edge in dag.edges if edge[1] == subtask.id],
                estimated_completion_time=subtask.weight * 0.1  # Simple estimation
            )
            
            allocations.append(allocation)
        
        return allocations
    
    def _update_metrics(self):
        """Update and report metrics to repository"""
        self.metrics.last_updated = time.time()
        
        if self.repository and self.metrics.tasks_processed > 0:
            # Calculate derived metrics
            self.metrics.queue_allocation_efficiency = (
                len(self.queue_manager.allocation_history) / 
                max(1, self.metrics.subtasks_generated)
            )
            
            # Report to repository every 30 seconds
            if int(time.time()) % 30 == 0:
                self.repository.update_metrics(self.decomposer_id, self.metrics.__dict__)
    
    def _calculate_critical_path(self, subtasks: List, edges: List) -> float:
        """Simple critical path calculation"""
        return sum(getattr(task, 'weight', 1.0) for task in subtasks)
    
    def _estimate_speedup(self, subtasks: List) -> float:
        """Estimate potential speedup from parallelization"""
        if not subtasks:
            return 1.0
        
        from task_analysis_config import CATEGORY_SPEEDUP
        
        # Simple category-based estimation
        total_weight = sum(getattr(task, 'weight', 1.0) for task in subtasks)
        parallel_weight = 0
        
        for task in subtasks:
            if hasattr(task, 'category'):
                speedup = CATEGORY_SPEEDUP.get(task.category, 1.0)
                parallel_weight += getattr(task, 'weight', 1.0) / speedup
        
        return total_weight / max(parallel_weight, 1.0) if parallel_weight > 0 else 1.0
    
    def _create_generic_dag(self, task: RepositoryTask) -> 'TaskDAG':
        """Create a generic DAG for unknown task types"""
        from decomposition_strategies import SubTask
        from task_analysis_config import TaskCategory, TaskPriority
        from smart_decomposer_modular import TaskDAG
        
        subtasks = [
            SubTask(
                id=f"{task.task_id}_generic",
                parent_task_id=task.task_id,
                name="Generic Task Processing",
                category=TaskCategory.CPU_TASKS,
                weight=2.0,
                priority=TaskPriority.MEDIUM
            )
        ]
        
        return TaskDAG(
            nodes=subtasks,
            edges=[],
            critical_path_weight=2.0,
            estimated_speedup=1.0,
            confidence_score=0.3,
            decomposition_strategy="generic"
        )

class AdvancedTaskAnalyzer:
    """Enhanced analyzer with fuzzy matching and repository integration"""
    
    def __init__(self):
        self.cache = {}
        self.similarity_threshold = 0.8
    
    def detect_task_type_advanced(self, description: str, params: Dict) -> Tuple[str, float]:
        """Enhanced task detection with fuzzy matching"""
        cache_key = f"{description[:100]}_{hash(str(sorted(params.items())))}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        from task_analysis_config import TASK_PATTERNS, TASK_HIERARCHIES
        
        best_score = 0
        detected_type = "generic"
        
        for task_type, config in TASK_PATTERNS.items():
            # Fuzzy keyword matching
            keyword_score = self._fuzzy_match_keywords(description, config['keywords'])
            
            # Parameter indicators
            param_score = sum(0.1 for indicator in config.get('complexity_indicators', [])
                            if indicator in params)
            
            # Hierarchical bonus
            hierarchy_bonus = 0.2 if any(parent in description.lower() 
                                       for parent in TASK_HIERARCHIES.get(task_type, []))  else 0
            
            total_score = keyword_score + param_score + hierarchy_bonus
            
            if total_score > best_score:
                best_score = total_score
                detected_type = task_type
        
        confidence = min(0.95, best_score)
        result = (detected_type, confidence)
        self.cache[cache_key] = result
        return result
    
    def _fuzzy_match_keywords(self, description: str, keywords: List[str]) -> float:
        """Fuzzy keyword matching using similarity ratio"""
        if not keywords:
            return 0.0
        
        desc_words = description.lower().split()
        total_score = 0.0
        
        for keyword in keywords:
            best_match = max(
                SequenceMatcher(None, keyword, word).ratio()
                for word in desc_words
            )
            if best_match >= self.similarity_threshold:
                total_score += best_match
        
        return total_score / len(keywords)