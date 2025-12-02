"""
Complete usage example of the enhanced decomposer system with
Central Task Repository and Advanced Queue integration
"""

import time
import threading
from typing import Dict, List, Any
from dataclasses import dataclass
from smart_decomposer_modular import (
    SmartSemanticDecomposer, 
    QueueAllocationManager,
    RepositoryTask,
    TaskRepositoryInterface,
    QueueInterface
)

# Mock implementations for demonstration

class MockAdvancedQueue:
    """Mock advanced queue implementation"""
    
    def __init__(self, name: str, max_size: int = 1000):
        self.name = name
        self.items = []
        self.max_size = max_size
        self.flow_control_enabled = False
        self.rate_limit = None
        self.last_enqueue = 0
        
    def enqueue(self, item: Any, priority: int = 0, routing_key: str = None) -> bool:
        # Flow control check
        if self.flow_control_enabled and self.rate_limit:
            current_time = time.time()
            if current_time - self.last_enqueue < (1.0 / self.rate_limit):
                return False  # Rate limited
        
        if len(self.items) >= self.max_size:
            return False  # Queue full
            
        self.items.append({
            'item': item,
            'priority': priority,
            'routing_key': routing_key,
            'timestamp': time.time()
        })
        self.last_enqueue = time.time()
        return True
    
    def get_queue_depth(self) -> int:
        return len(self.items)
    
    def get_queue_health(self) -> Dict[str, Any]:
        return {
            'depth': len(self.items),
            'capacity_used': len(self.items) / self.max_size,
            'flow_control_active': self.flow_control_enabled,
            'avg_processing_time': 0.1,  # Mock value
            'error_rate': 0.01
        }
    
    def set_flow_control(self, enabled: bool, rate_limit: int = None):
        self.flow_control_enabled = enabled
        self.rate_limit = rate_limit
        print(f"Queue {self.name}: Flow control {'enabled' if enabled else 'disabled'}")
    
    def dequeue(self) -> Any:
        """For demonstration - normally handled by queue consumers"""
        if self.items:
            return self.items.pop(0)
        return None

class MockTaskRepository:
    """Mock Central Task Repository"""
    
    def __init__(self):
        self.decomposers = {}
        self.task_queue = []
        self.completed_tasks = {}
        self.error_tasks = {}
        self.metrics = {}
        self.task_counter = 0
        
        # Simulate incoming tasks
        self._populate_sample_tasks()
    
    def _populate_sample_tasks(self):
        """Add sample tasks to repository"""
        sample_tasks = [
            {
                'description': 'Encode 4K video with H.265 codec for streaming platform',
                'parameters': {
                    'frame_count': 7200,  # 5 minutes at 24fps
                    'resolution': '4k',
                    'codec': 'h265',
                    'bitrate': '15mbps',
                    'has_audio': True,
                    'available_gpus': 2
                },
                'priority': 3
            },
            {
                'description': 'Train deep learning model for image classification',
                'parameters': {
                    'data_size': 50000,
                    'batch_size': 64,
                    'model_type': 'transformer',
                    'epochs': 100,
                    'available_gpus': 4
                },
                'priority': 2
            },
            {
                'description': 'Calculate portfolio risk using Monte Carlo simulation',
                'parameters': {
                    'portfolio_size': 2000,
                    'simulation_count': 100000,
                    'calculation_type': 'monte_carlo',
                    'time_horizon': 252,  # 1 year trading days
                    'available_cpus': 8
                },
                'priority': 2
            },
            {
                'description': 'Process batch of medical images with noise reduction',
                'parameters': {
                    'image_count': 500,
                    'operations': ['denoise', 'enhance', 'segment'],
                    'resolution': 2048,
                    'batch_processing': True
                },
                'priority': 1
            }
        ]
        
        for i, task_data in enumerate(sample_tasks):
            task = RepositoryTask(
                task_id=f"task_{self.task_counter:06d}",
                description=task_data['description'],
                parameters=task_data['parameters'],
                priority=task_data['priority'],
                deadline=time.time() + 3600  # 1 hour deadline
            )
            self.task_queue.append(task)
            self.task_counter += 1
    
    def register_decomposer(self, decomposer_id: str, capabilities: Dict) -> bool:
        self.decomposers[decomposer_id] = {
            'capabilities': capabilities,
            'registered_at': time.time(),
            'last_heartbeat': time.time(),
            'active': True
        }
        print(f"Registered decomposer: {decomposer_id}")
        return True
    
    def request_task_batch(self, decomposer_id: str, batch_size: int = 10) -> List[RepositoryTask]:
        # Update heartbeat
        if decomposer_id in self.decomposers:
            self.decomposers[decomposer_id]['last_heartbeat'] = time.time()
        
        # Return available tasks
        available_tasks = []
        for _ in range(min(batch_size, len(self.task_queue))):
            if self.task_queue:
                task = self.task_queue.pop(0)
                task.assigned_to = decomposer_id
                available_tasks.append(task)
        
        if available_tasks:
            print(f"Assigned {len(available_tasks)} tasks to {decomposer_id}")
        
        return available_tasks
    
    def report_completion(self, task_id: str, result: Dict) -> bool:
        self.completed_tasks[task_id] = {
            'result': result,
            'completed_at': time.time()
        }
        print(f"Task {task_id} completed: {result.get('subtasks_created', 0)} subtasks created")
        return True
    
    def report_error(self, task_id: str, error: str) -> bool:
        self.error_tasks[task_id] = {
            'error': error,
            'failed_at': time.time()
        }
        print(f"Task {task_id} failed: {error}")
        return True
    
    def update_metrics(self, decomposer_id: str, metrics: Dict) -> bool:
        self.metrics[decomposer_id] = {
            'metrics': metrics,
            'updated_at': time.time()
        }
        return True
    
    def add_more_tasks(self):
        """Simulate continuous task arrival"""
        additional_tasks = [
            {
                'description': 'Analyze financial time series data with LSTM model',
                'parameters': {
                    'data_points': 100000,
                    'sequence_length': 60,
                    'model_type': 'lstm',
                    'features': 20
                },
                'priority': 2
            },
            {
                'description': 'Render 3D animation sequence with complex lighting',
                'parameters': {
                    'frame_count': 1440,  # 1 minute at 24fps
                    'resolution': '1080p',
                    'complexity': 'high',
                    'lighting_model': 'ray_tracing'
                },
                'priority': 3
            }
        ]
        
        for task_data in additional_tasks:
            task = RepositoryTask(
                task_id=f"task_{self.task_counter:06d}",
                description=task_data['description'],
                parameters=task_data['parameters'],
                priority=task_data['priority'],
                deadline=time.time() + 7200  # 2 hour deadline
            )
            self.task_queue.append(task)
            self.task_counter += 1

class QueueConsumerSimulator:
    """Simulates consumers processing tasks from queues"""
    
    def __init__(self, queue: MockAdvancedQueue, processing_rate: float = 2.0):
        self.queue = queue
        self.processing_rate = processing_rate  # items per second
        self.running = False
        self.consumer_thread = None
        self.processed_count = 0
    
    def start(self):
        self.running = True
        self.consumer_thread = threading.Thread(target=self._consume_loop, daemon=True)
        self.consumer_thread.start()
        print(f"Started consumer for {self.queue.name}")
    
    def stop(self):
        self.running = False
        if self.consumer_thread:
            self.consumer_thread.join(timeout=2)
    
    def _consume_loop(self):
        while self.running:
            item = self.queue.dequeue()
            if item:
                # Simulate processing time
                time.sleep(1.0 / self.processing_rate)
                self.processed_count += 1
                
                if self.processed_count % 10 == 0:
                    print(f"{self.queue.name} processed {self.processed_count} items")
            else:
                time.sleep(0.1)  # Brief pause when no items

def demonstrate_enhanced_decomposer():
    """Complete demonstration of the enhanced decomposer system"""
    
    print("=" * 60)
    print("Enhanced Decomposer System Demonstration")
    print("=" * 60)
    
    # 1. Initialize Mock Repository
    print("\n1. Initializing Central Task Repository...")
    repository = MockTaskRepository()
    
    # 2. Create Advanced Queue System
    print("\n2. Setting up Advanced Queue System...")
    from task_analysis_config import ADVANCED_QUEUE_CONFIG
    
    queues = {
        'gpu_queue': MockAdvancedQueue('GPU_Queue', 500),
        'cpu_queue': MockAdvancedQueue('CPU_Queue', 1000), 
        'io_queue': MockAdvancedQueue('IO_Queue', 300)
    }
    
    # Initialize Queue Allocation Manager
    queue_manager = QueueAllocationManager(ADVANCED_QUEUE_CONFIG)
    for name, queue in queues.items():
        queue_manager.register_queue(name, queue)
    
    # 3. Create and Initialize Decomposers
    print("\n3. Spawning Smart Decomposers...")
    decomposers = []
    
    for i in range(3):  # Spawn 3 decomposers
        decomposer = SmartSemanticDecomposer(
            decomposer_id=f"decomposer_{i+1}",
            repository=repository
        )
        decomposer.initialize(queue_manager)
        decomposer.start_processing()
        decomposers.append(decomposer)
    
    # 4. Start Queue Consumers
    print("\n4. Starting Queue Consumers...")
    consumers = []
    processing_rates = {'gpu_queue': 1.5, 'cpu_queue': 3.0, 'io_queue': 5.0}
    
    for queue_name, queue in queues.items():
        consumer = QueueConsumerSimulator(queue, processing_rates[queue_name])
        consumer.start()
        consumers.append(consumer)
    
    # 5. Monitor System Performance
    print("\n5. System Running - Monitoring Performance...")
    print("-" * 50)
    
    start_time = time.time()
    monitoring_duration = 30  # seconds
    
    while time.time() - start_time < monitoring_duration:
        time.sleep(5)
        
        # Print system status
        print(f"\nSystem Status (t={int(time.time() - start_time)}s):")
        print(f"Tasks in repository: {len(repository.task_queue)}")
        print(f"Completed tasks: {len(repository.completed_tasks)}")
        print(f"Failed tasks: {len(repository.error_tasks)}")
        
        # Queue depths
        for name, queue in queues.items():
            depth = queue.get_queue_depth()
            health = queue.get_queue_health()
            print(f"{name}: {depth} items ({health['capacity_used']:.1%} full)")
        
        # Add more tasks midway through
        if 10 <= time.time() - start_time < 15:
            repository.add_more_tasks()
            print("Added more tasks to repository...")
    
    # 6. System Shutdown and Metrics
    print("\n6. Shutting Down System...")
    print("-" * 50)
    
    # Stop decomposers
    for decomposer in decomposers:
        decomposer.stop_processing()
    
    # Stop consumers  
    for consumer in consumers:
        consumer.stop()
    
    # Final metrics
    print(f"\nFinal System Metrics:")
    print(f"Total tasks processed: {len(repository.completed_tasks)}")
    print(f"Total tasks failed: {len(repository.error_tasks)}")
    
    total_processed = sum(consumer.processed_count for consumer in consumers)
    print(f"Total subtasks processed by consumers: {total_processed}")
    
    # Queue final states
    print(f"\nFinal Queue States:")
    for name, queue in queues.items():
        print(f"{name}: {queue.get_queue_depth()} items remaining")
    
    # Decomposer metrics
    print(f"\nDecomposer Performance:")
    for decomposer in decomposers:
        metrics = decomposer.metrics
        print(f"{decomposer.decomposer_id}:")
        print(f"  Tasks processed: {metrics.tasks_processed}")
        print(f"  Subtasks generated: {metrics.subtasks_generated}")
        print(f"  Avg decomposition time: {metrics.avg_decomposition_time:.3f}s")
        print(f"  Error rate: {metrics.error_rate}")

def demonstrate_queue_flow_control():
    """Demonstrate advanced queue flow control features"""
    
    print("\n" + "=" * 60)
    print("Queue Flow Control Demonstration")
    print("=" * 60)
    
    from task_analysis_config import ADVANCED_QUEUE_CONFIG
    
    # Create a small queue to trigger flow control
    small_queue = MockAdvancedQueue('FlowControl_Demo', max_size=50)
    queue_manager = QueueAllocationManager({'demo_queue': ADVANCED_QUEUE_CONFIG['gpu_queue']})
    queue_manager.register_queue('demo_queue', small_queue)
    
    # Create many allocations to fill the queue
    from smart_decomposer_modular import SubTaskAllocation
    from decomposition_strategies import SubTask
    from task_analysis_config import TaskCategory, TaskPriority
    
    allocations = []
    for i in range(100):  # More than queue capacity
        subtask = SubTask(
            id=f"demo_task_{i}",
            parent_task_id="demo_parent",
            name=f"Demo Subtask {i}",
            category=TaskCategory.GPU_TASKS,
            weight=1.0,
            priority=TaskPriority.MEDIUM
        )
        
        allocation = SubTaskAllocation(
            subtask=subtask,
            target_queue='demo_queue',
            allocation_weight=1.0,
            flow_control_priority=2,
            batching_eligible=True
        )
        allocations.append(allocation)
    
    print(f"Attempting to allocate {len(allocations)} tasks to queue (capacity: 50)")
    
    # Allocate in batches and observe flow control
    batch_size = 20
    for i in range(0, len(allocations), batch_size):
        batch = allocations[i:i + batch_size]
        result = queue_manager.allocate_subtasks(batch)
        
        successful = sum(len(tasks) for tasks in result.values())
        queue_depth = small_queue.get_queue_depth()
        
        print(f"Batch {i//batch_size + 1}: {successful}/{len(batch)} allocated, "
              f"queue depth: {queue_depth}, flow control: {small_queue.flow_control_enabled}")
        
        time.sleep(1)  # Brief pause between batches

if __name__ == "__main__":
    # Run the main demonstration
    demonstrate_enhanced_decomposer()
    
    # Demonstrate flow control
    demonstrate_queue_flow_control()
    
    print("\n" + "=" * 60)
    print("Demonstration Complete")
    print("=" * 60)