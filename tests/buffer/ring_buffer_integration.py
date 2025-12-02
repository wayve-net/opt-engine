"""
Integration example showing how to use the lock-free ring buffer
with your task decomposer system
"""

import threading
import time
from typing import Dict, Any, List
from dataclasses import dataclass
from lockfree_ring_buffer import LockFreeRingBuffer, TaskBatchBuffer, BufferResult
from smart_decomposer_enhanced import SubTaskAllocation, QueueInterface

@dataclass
class QueuedTask:
    """Task wrapper for queue operations"""
    task_id: str
    subtask_allocation: SubTaskAllocation
    priority: int
    timestamp: float
    retry_count: int = 0

class RingBufferQueue(QueueInterface):
    """
    Queue implementation using lock-free ring buffer
    Integrates with your QueueAllocationManager
    """
    
    def __init__(self, 
                 name: str,
                 capacity: int = 2048,
                 batch_size: int = 20,
                 flow_control_threshold: float = 0.8):
        
        self.name = name
        self.buffer = TaskBatchBuffer(
            capacity=capacity,
            batch_size=batch_size,
            name=f"{name}_buffer"
        )
        
        # Flow control settings
        self.flow_control_enabled = False
        self.flow_control_rate = 10  # items per second
        self.flow_control_threshold = flow_control_threshold
        self._last_flow_control_time = 0.0
        
        # Priority queues using separate buffers
        self.priority_buffers: Dict[int, LockFreeRingBuffer] = {
            3: LockFreeRingBuffer(capacity//4, name=f"{name}_high_priority"),    # HIGH
            2: LockFreeRingBuffer(capacity//2, name=f"{name}_medium_priority"),  # MEDIUM  
            1: LockFreeRingBuffer(capacity//4, name=f"{name}_low_priority")      # LOW
        }
        
        # Metrics and monitoring
        self._total_enqueued = 0
        self._total_dequeued = 0
        self._rejected_count = 0
        
        # Start batch processing
        self.buffer.start_batching()
        
        # Health monitoring thread
        self._health_monitor = threading.Thread(target=self._monitor_health, daemon=True)
        self._health_monitor.start()
    
    def enqueue(self, item: Any, priority: int = 2, routing_key: str = None) -> bool:
        """
        Enqueue item with priority and routing
        
        Args:
            item: SubTaskAllocation or task data
            priority: 1=LOW, 2=MEDIUM, 3=HIGH
            routing_key: Optional routing information
        
        Returns:
            True if successfully enqueued
        """
        # Check flow control
        if self.flow_control_enabled and self._should_apply_flow_control():
            return False
        
        # Create queued task wrapper
        if isinstance(item, SubTaskAllocation):
            queued_task = QueuedTask(
                task_id=item.subtask.id,
                subtask_allocation=item,
                priority=priority,
                timestamp=time.time()
            )
        else:
            # Handle raw task data
            queued_task = QueuedTask(
                task_id=str(hash(str(item))),
                subtask_allocation=item,
                priority=priority,
                timestamp=time.time()
            )
        
        # Route to appropriate priority buffer
        target_buffer = self.priority_buffers.get(priority, self.priority_buffers[2])
        
        # Attempt enqueue with timeout
        result = target_buffer.enqueue(queued_task, timeout=0.1)
        
        if result == BufferResult.SUCCESS:
            self._total_enqueued += 1
            return True
        elif result == BufferResult.BACKPRESSURE:
            # Try batch queue as fallback
            return self.buffer.enqueue_for_batch(queued_task)
        else:
            self._rejected_count += 1
            return False
    
    def dequeue(self, timeout: float = None) -> tuple[bool, Any]:
        """
        Dequeue highest priority item
        
        Args:
            timeout: Optional timeout in seconds
        
        Returns:
            (success, item) tuple
        """
        # Check priority buffers in order (HIGH -> MEDIUM -> LOW)
        for priority in [3, 2, 1]:
            buffer = self.priority_buffers[priority]
            
            result, item = buffer.dequeue(timeout=0.001)  # Quick check
            if result == BufferResult.SUCCESS:
                self._total_dequeued += 1
                return True, item.subtask_allocation
        
        # Fallback to main buffer with full timeout
        result, item = self.buffer.dequeue(timeout=timeout)
        if result == BufferResult.SUCCESS:
            self._total_dequeued += 1
            return True, item.subtask_allocation
        
        return False, None
    
    def dequeue_batch(self, max_items: int = 10) -> List[Any]:
        """Dequeue multiple items efficiently"""
        items = []
        
        # Collect from priority buffers first
        for priority in [3, 2, 1]:
            if len(items) >= max_items:
                break
                
            buffer = self.priority_buffers[priority]
            remaining = max_items - len(items)
            
            result, batch = buffer.dequeue_batch(remaining, timeout=0.001)
            if result == BufferResult.SUCCESS:
                items.extend([item.subtask_allocation for item in batch])
        
        # Fill remaining from main buffer
        if len(items) < max_items:
            remaining = max_items - len(items)
            result, batch = self.buffer.dequeue_batch(remaining, timeout=0.001)
            if result == BufferResult.SUCCESS:
                items.extend([item.subtask_allocation for item in batch])
        
        self._total_dequeued += len(items)
        return items
    
    def get_queue_depth(self) -> int:
        """Get total items across all buffers"""
        total = self.buffer.size()
        for buffer in self.priority_buffers.values():
            total += buffer.size()
        return total
    
    def get_queue_health(self) -> Dict[str, Any]:
        """Get comprehensive queue health metrics"""
        total_capacity = self.buffer.capacity + sum(
            buf.capacity for buf in self.priority_buffers.values()
        )
        total_size = self.get_queue_depth()
        
        health = {
            'name': self.name,
            'total_size': total_size,
            'total_capacity': total_capacity,
            'utilization': total_size / total_capacity,
            'flow_control_active': self.flow_control_enabled,
            'throughput': {
                'enqueued': self._total_enqueued,
                'dequeued': self._total_dequeued,
                'rejected': self._rejected_count
            },
            'priority_breakdown': {}
        }
        
        # Add priority buffer details
        for priority, buffer in self.priority_buffers.items():
            metrics = buffer.get_metrics()
            health['priority_breakdown'][f'priority_{priority}'] = {
                'size': buffer.size(),
                'utilization': buffer.utilization(),
                'throughput': metrics.throughput_items_per_sec if metrics else 0
            }
        
        return health
    
    def set_flow_control(self, enabled: bool, rate_limit: int = None):
        """Configure flow control"""
        self.flow_control_enabled = enabled
        if rate_limit:
            self.flow_control_rate = rate_limit
        
        # Register backpressure callbacks
        if enabled:
            for buffer in [self.buffer] + list(self.priority_buffers.values()):
                buffer.register_backpressure_callback(self._handle_backpressure)
    
    def _should_apply_flow_control(self) -> bool:
        """Check if flow control should be applied"""
        if not self.flow_control_enabled:
            return False
        
        current_time = time.time()
        time_since_last = current_time - self._last_flow_control_time
        
        if time_since_last < (1.0 / self.flow_control_rate):
            return True  # Rate limit active
        
        # Check if utilization exceeds threshold
        total_utilization = self.get_queue_depth() / sum(
            buf.capacity for buf in [self.buffer] + list(self.priority_buffers.values())
        )
        
        if total_utilization > self.flow_control_threshold:
            self._last_flow_control_time = current_time
            return True
        
        return False
    
    def _handle_backpressure(self, utilization: float):
        """Handle backpressure events"""
        print(f"Backpressure detected on {self.name}: {utilization:.2%} utilization")
        
        # Could trigger additional actions:
        # - Notify load balancer
        # - Adjust batch sizes
        # - Scale up resources
    
    def _monitor_health(self):
        """Background health monitoring"""
        while True:
            try:
                health = self.get_queue_health()
                
                # Log health metrics periodically
                if health['utilization'] > 0.9:
                    print(f"WARNING: {self.name} at {health['utilization']:.1%} capacity")
                elif health['utilization'] > 0.7:
                    print(f"INFO: {self.name} at {health['utilization']:.1%} capacity")
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                print(f"Health monitor error for {self.name}: {e}")
                time.sleep(60)
    
    def shutdown(self, drain: bool = True) -> int:
        """Graceful shutdown"""
        remaining = 0
        
        # Shutdown all buffers
        remaining += self.buffer.shutdown(drain=drain)
        for buffer in self.priority_buffers.values():
            remaining += buffer.shutdown(drain=drain)
        
        print(f"Queue {self.name} shutdown complete. {remaining} items remaining.")
        return remaining


# Example integration with your existing system
class EnhancedQueueAllocationManager:
    """
    Enhanced version of your QueueAllocationManager using ring buffers
    """
    
    def __init__(self, queue_configs: Dict[str, Dict]):
        self.queue_configs = queue_configs
        self.queues: Dict[str, RingBufferQueue] = {}
        
        # Initialize ring buffer queues
        for queue_name, config in queue_configs.items():
            self.queues[queue_name] = RingBufferQueue(
                name=queue_name,
                capacity=config.get('max_depth', 1024),
                batch_size=config.get('max_batch_size', 20),
                flow_control_threshold=config.get('flow_control_threshold', 0.8)
            )
    
    def allocate_subtasks_batch(self, 
                              allocations: List[SubTaskAllocation],
                              batch_size: int = 50) -> Dict[str, int]:
        """
        Optimized batch allocation using ring buffers
        """
        allocation_results = {}
        
        # Group allocations by queue
        queue_groups = {}
        for allocation in allocations:
            queue_name = allocation.target_queue
            if queue_name not in queue_groups:
                queue_groups[queue_name] = []
            queue_groups[queue_name].append(allocation)
        
        # Process each queue's allocations in batches
        for queue_name, queue_allocations in queue_groups.items():
            if queue_name not in self.queues:
                continue
            
            queue = self.queues[queue_name]
            successful_count = 0
            
            # Process in batches
            for i in range(0, len(queue_allocations), batch_size):
                batch = queue_allocations[i:i + batch_size]
                
                for allocation in batch:
                    success = queue.enqueue(
                        allocation,
                        priority=allocation.flow_control_priority
                    )
                    if success:
                        successful_count += 1
                    else:
                        break  # Stop on first failure to avoid overwhelming queue
            
            allocation_results[queue_name] = successful_count
        
        return allocation_results
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health"""
        system_health = {
            'timestamp': time.time(),
            'total_queues': len(self.queues),
            'queues': {}
        }
        
        total_utilization = 0
        total_throughput = 0
        
        for queue_name, queue in self.queues.items():
            health = queue.get_queue_health()
            system_health['queues'][queue_name] = health
            total_utilization += health['utilization']
            
            # Calculate throughput
            enqueued = health['throughput']['enqueued']
            dequeued = health['throughput']['dequeued']
            total_throughput += (enqueued + dequeued)
        
        system_health['overall'] = {
            'average_utilization': total_utilization / len(self.queues) if self.queues else 0,
            'total_throughput': total_throughput,
            'flow_control_active_count': sum(
                1 for q in self.queues.values() if q.flow_control_enabled
            )
        }
        
        return system_health


# Example usage and benchmarking
def benchmark_ring_buffer():
    """Benchmark the ring buffer performance"""
    import random
    from concurrent.futures import ThreadPoolExecutor
    
    buffer = LockFreeRingBuffer[int](capacity=1024, name="benchmark")
    
    # Producer function
    def producer(items_to_produce: int, producer_id: int):
        produced = 0
        for i in range(items_to_produce):
            item = producer_id * 1000 + i
            result = buffer.enqueue(item, timeout=1.0)
            if result == BufferResult.SUCCESS:
                produced += 1
            elif result == BufferResult.BACKPRESSURE:
                time.sleep(0.001)  # Brief backoff
        return produced
    
    # Consumer function
    def consumer(items_to_consume: int, consumer_id: int):
        consumed = 0
        items = []
        
        while consumed < items_to_consume:
            # Try batch dequeue first
            result, batch = buffer.dequeue_batch(max_items=10, timeout=0.1)
            if result == BufferResult.SUCCESS:
                items.extend(batch)
                consumed += len(batch)
            else:
                # Single item dequeue
                result, item = buffer.dequeue(timeout=0.1)
                if result == BufferResult.SUCCESS:
                    items.append(item)
                    consumed += 1
                elif result == BufferResult.BUFFER_EMPTY:
                    time.sleep(0.001)
        
        return consumed, items
    
    print("Starting ring buffer benchmark...")
    start_time = time.time()
    
    # Run benchmark with multiple producers and consumers
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Start producers
        producer_futures = [
            executor.submit(producer, 1000, i) for i in range(4)
        ]
        
        # Start consumers  
        consumer_futures = [
            executor.submit(consumer, 1000, i) for i in range(4)
        ]
        
        # Wait for completion
        total_produced = sum(f.result() for f in producer_futures)
        consumer_results = [f.result() for f in consumer_futures]
        total_consumed = sum(result[0] for result in consumer_results)
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Get final metrics
    metrics = buffer.get_metrics()
    
    print(f"\nBenchmark Results:")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Total Produced: {total_produced}")
    print(f"Total Consumed: {total_consumed}")
    print(f"Throughput: {(total_produced + total_consumed) / duration:.0f} ops/sec")
    print(f"Peak Utilization: {metrics.peak_utilization:.1%}")
    print(f"Backpressure Events: {metrics.backpressure_events}")
    print(f"Failed Enqueues: {metrics.failed_enqueues}")
    
    buffer.shutdown()


# Integration with your decomposer configuration
def create_optimized_queue_system():
    """
    Create a complete queue system using your configuration
    """
    from task_analysis_config_enhanced import ADVANCED_QUEUE_CONFIG, DEFAULT_RESOURCE_CONSTRAINTS
    
    # Enhanced queue configs with ring buffer optimizations
    optimized_configs = {}
    
    for queue_name, config in ADVANCED_QUEUE_CONFIG.items():
        optimized_configs[queue_name] = {
            **config,
            'ring_buffer_capacity': config['max_depth'],
            'batch_processing': True,
            'priority_levels': config['priority_levels'],
            'backpressure_threshold': config['flow_control_threshold'],
            'metrics_enabled': True
        }
    
    # Create queue allocation manager
    queue_manager = EnhancedQueueAllocationManager(optimized_configs)
    
    # Configure backpressure handlers
    def handle_gpu_backpressure(utilization: float):
        if utilization > 0.9:
            print(f"GPU queue critically full: {utilization:.1%}")
            # Could trigger scale-up or load redistribution
    
    def handle_cpu_backpressure(utilization: float):
        if utilization > 0.85:
            print(f"CPU queue high utilization: {utilization:.1%}")
    
    # Register backpressure callbacks
    if 'gpu_queue' in queue_manager.queues:
        queue_manager.queues['gpu_queue'].buffer.register_backpressure_callback(
            handle_gpu_backpressure
        )
    
    if 'cpu_queue' in queue_manager.queues:
        queue_manager.queues['cpu_queue'].buffer.register_backpressure_callback(
            handle_cpu_backpressure
        )
    
    return queue_manager


# Performance monitoring and alerting
class RingBufferMonitor:
    """Advanced monitoring for ring buffer performance"""
    
    def __init__(self, queue_manager: EnhancedQueueAllocationManager):
        self.queue_manager = queue_manager
        self.monitoring_active = False
        self.alert_thresholds = {
            'utilization_critical': 0.95,
            'utilization_warning': 0.80,
            'throughput_min': 100,  # ops/sec
            'backpressure_max': 10   # events per minute
        }
    
    def start_monitoring(self, interval: float = 30.0):
        """Start continuous monitoring"""
        self.monitoring_active = True
        
        def monitor_loop():
            while self.monitoring_active:
                try:
                    self._check_system_health()
                    time.sleep(interval)
                except Exception as e:
                    print(f"Monitor error: {e}")
                    time.sleep(interval * 2)
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        print(f"Ring buffer monitoring started (interval: {interval}s)")
    
    def _check_system_health(self):
        """Check system health and trigger alerts"""
        health = self.queue_manager.get_system_health()
        
        for queue_name, queue_health in health['queues'].items():
            utilization = queue_health['utilization']
            
            # Check utilization alerts
            if utilization >= self.alert_thresholds['utilization_critical']:
                self._send_alert('CRITICAL', f"Queue {queue_name} at {utilization:.1%} capacity")
            elif utilization >= self.alert_thresholds['utilization_warning']:
                self._send_alert('WARNING', f"Queue {queue_name} at {utilization:.1%} capacity")
            
            # Check throughput
            throughput = queue_health['throughput']
            total_ops = throughput['enqueued'] + throughput['dequeued']
            if total_ops < self.alert_thresholds['throughput_min']:
                self._send_alert('INFO', f"Queue {queue_name} low throughput: {total_ops} ops")
    
    def _send_alert(self, level: str, message: str):
        """Send alert (customize for your alerting system)"""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] {level}: {message}")
        
        # Could integrate with:
        # - Prometheus alerts
        # - Slack notifications  
        # - Email alerts
        # - Custom webhook
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring_active = False


if __name__ == "__main__":
    # Example: Run benchmark
    benchmark_ring_buffer()
    
    # Example: Create production system
    print("\nCreating optimized queue system...")
    queue_system = create_optimized_queue_system()
    
    # Start monitoring
    monitor = RingBufferMonitor(queue_system)
    monitor.start_monitoring(interval=10.0)
    
    # Simulate some work
    print("Queue system ready. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down queue system...")
        
        # Graceful shutdown
        for queue in queue_system.queues.values():
            remaining = queue.shutdown(drain=True)
            print(f"Queue {queue.name} shutdown: {remaining} items remaining")
        
        monitor.stop_monitoring()
        print("Shutdown complete.")