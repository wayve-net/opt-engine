"""
Enhanced Task Analysis Configuration
Includes hierarchical patterns, adaptive rules, and ring buffer settings
"""

from enum import Enum
from typing import Dict, List, Any, Tuple

class TaskCategory(Enum):
    GPU_TASKS = "gpu_tasks"
    CPU_TASKS = "cpu_tasks" 
    IO_TASKS = "io_tasks"

class TaskPriority(Enum):
    HIGH = 3
    MEDIUM = 2
    LOW = 1

# Hierarchical task patterns for better classification
TASK_HIERARCHIES = {
    'video_rendering': ['video', 'rendering'],
    'video_encoding': ['video', 'compression'],
    'ml_inference': ['ml', 'ai'],
    'ml_training': ['ml', 'ai'],
    'image_processing': ['image', 'vision'],
    'data_analytics': ['data_processing', 'analytics'],
    'financial_modeling': ['financial', 'mathematics']
}

# Enhanced task patterns with fuzzy matching support
TASK_PATTERNS = {
    'video': {
        'keywords': ['video', 'encode', 'h264', 'h265', 'render', 'transcode', 'codec', 'ffmpeg', 'streaming'],
        'primary_category': TaskCategory.GPU_TASKS,
        'secondary_categories': [TaskCategory.CPU_TASKS, TaskCategory.IO_TASKS],
        'complexity_indicators': ['resolution', 'bitrate', 'frame_count', 'duration'],
        'base_weight': 2.0,
        'parallel_factor': 0.8
    },
    'ml': {
        'keywords': ['model', 'inference', 'neural', 'predict', 'tensorflow', 'pytorch', 'train', 'deep', 'ai'],
        'primary_category': TaskCategory.GPU_TASKS,
        'secondary_categories': [TaskCategory.CPU_TASKS, TaskCategory.IO_TASKS],
        'complexity_indicators': ['batch_size', 'data_size', 'model_size', 'layers'],
        'base_weight': 3.0,
        'parallel_factor': 0.9
    },
    'database': {
        'keywords': ['query', 'database', 'sql', 'select', 'insert', 'update', 'postgres', 'mysql', 'index'],
        'primary_category': TaskCategory.IO_TASKS,
        'secondary_categories': [TaskCategory.CPU_TASKS],
        'complexity_indicators': ['record_count', 'table_size', 'join_complexity', 'index_count'],
        'base_weight': 1.5,
        'parallel_factor': 0.4
    },
    'financial': {
        'keywords': ['finance', 'calculate', 'portfolio', 'risk', 'pricing', 'monte', 'carlo', 'var', 'trading'],
        'primary_category': TaskCategory.CPU_TASKS,
        'secondary_categories': [TaskCategory.IO_TASKS],
        'complexity_indicators': ['portfolio_size', 'simulation_count', 'time_horizon', 'instruments'],
        'base_weight': 2.5,
        'parallel_factor': 0.7
    },
    'image': {
        'keywords': ['image', 'filter', 'resize', 'crop', 'process', 'opencv', 'pil', 'transform', 'vision'],
        'primary_category': TaskCategory.GPU_TASKS,
        'secondary_categories': [TaskCategory.CPU_TASKS],
        'complexity_indicators': ['image_count', 'resolution', 'operation_count', 'channels'],
        'base_weight': 1.8,
        'parallel_factor': 0.85
    },
    'data_processing': {
        'keywords': ['csv', 'excel', 'json', 'xml', 'parse', 'transform', 'etl', 'data', 'clean'],
        'primary_category': TaskCategory.CPU_TASKS,
        'secondary_categories': [TaskCategory.IO_TASKS],
        'complexity_indicators': ['file_size', 'record_count', 'transformation_complexity', 'columns'],
        'base_weight': 1.2,
        'parallel_factor': 0.6
    }
}

# Enhanced category performance with adaptive speedup
CATEGORY_SPEEDUP = {
    TaskCategory.GPU_TASKS: 8.0,
    TaskCategory.CPU_TASKS: 4.0,
    TaskCategory.IO_TASKS: 1.5
}

# Adaptive resource constraints based on system monitoring
DEFAULT_RESOURCE_CONSTRAINTS = {
    'max_gpu_tasks': 4,
    'max_cpu_cores': 8,
    'memory_limit_gb': 16,
    'network_bandwidth_mbps': 1000,
    'storage_iops': 10000,
    'ring_buffer_size': 1024,
    'queue_timeout_ms': 100
}

# Enhanced dynamic weighting with contextual factors
DYNAMIC_WEIGHT_FACTORS = {
    'cpu_load_multiplier': 1.2,
    'memory_usage_multiplier': 1.3,
    'network_latency_multiplier': 1.1,
    'gpu_utilization_multiplier': 1.4,
    'queue_depth_multiplier': 1.1,  # Adjust based on queue congestion
    'time_of_day_multiplier': 0.9   # Lower weights during peak hours
}

# Advanced queue configuration for external queue systems
ADVANCED_QUEUE_CONFIG = {
    'gpu_queue': {
        'max_depth': 2000,
        'max_batch_size': 100,
        'flow_control_threshold': 0.8,
        'flow_control_rate': 20,  # items per second when flow control active
        'priority_levels': 5,
        'batching_window_ms': 100,
        'allocation_strategy': 'priority_aware'
    },
    'cpu_queue': {
        'max_depth': 5000,
        'max_batch_size': 200,
        'flow_control_threshold': 0.85,
        'flow_control_rate': 50,
        'priority_levels': 3,
        'batching_window_ms': 50,
        'allocation_strategy': 'load_balanced'
    },
    'io_queue': {
        'max_depth': 1000,
        'max_batch_size': 50,
        'flow_control_threshold': 0.7,
        'flow_control_rate': 10,
        'priority_levels': 3,
        'batching_window_ms': 200,
        'allocation_strategy': 'batch_optimized'
    }
}

# Central Task Repository configuration
REPOSITORY_CONFIG = {
    'connection_pool_size': 10,
    'heartbeat_interval_seconds': 30,
    'task_batch_sizes': {
        'low_load': 5,
        'medium_load': 10, 
        'high_load': 20
    },
    'decomposer_scaling': {
        'min_decomposers': 2,
        'max_decomposers': 16,
        'scale_up_threshold': 100,  # tasks in repository
        'scale_down_threshold': 10,
        'cooldown_minutes': 5
    },
    'retry_policies': {
        'max_retries': 3,
        'backoff_multiplier': 2.0,
        'max_backoff_seconds': 300
    }
}

# Contextual complexity rules with relationships between parameters
COMPLEXITY_RULES = {
    'video': {
        'frame_count': {'low': 100, 'medium': 1000, 'high': 5000},
        'resolution': {'low': '720p', 'medium': '1080p', 'high': '4k'},
        'bitrate': {'low': '1mbps', 'medium': '5mbps', 'high': '20mbps'},
        # Relationship: frame_count * resolution_factor * bitrate_factor
        'complexity_function': lambda fc, res, br: fc * {'720p': 1, '1080p': 2.25, '4k': 16}.get(res, 1) * (br / 5)
    },
    'ml': {
        'data_size': {'low': 1000, 'medium': 100000, 'high': 1000000},
        'batch_size': {'low': 16, 'medium': 64, 'high': 256},
        'model_size': {'low': 10, 'medium': 100, 'high': 1000},  # MB
        'complexity_function': lambda ds, bs, ms: (ds / bs) * (ms / 100) ** 0.5
    },
    'financial': {
        'portfolio_size': {'low': 50, 'medium': 500, 'high': 5000},
        'simulation_count': {'low': 1000, 'medium': 10000, 'high': 100000},
        'time_horizon': {'low': 30, 'medium': 252, 'high': 1260},  # days
        'complexity_function': lambda ps, sc, th: ps * (sc ** 0.8) * (th / 252)
    },
    'image': {
        'image_count': {'low': 10, 'medium': 100, 'high': 1000},
        'resolution': {'low': 480, 'medium': 1080, 'high': 2160},  # height in pixels
        'operation_count': {'low': 2, 'medium': 5, 'high': 10},
        'complexity_function': lambda ic, res, oc: ic * (res / 1080) ** 1.5 * oc
    }
}

# Enhanced priority assignment with context awareness
PRIORITY_RULES = {
    'high_priority_keywords': ['urgent', 'critical', 'emergency', 'real-time', 'live', 'immediate'],
    'low_priority_keywords': ['batch', 'background', 'offline', 'non-urgent', 'scheduled', 'deferred'],
    'medium_priority_keywords': ['standard', 'normal', 'regular', 'default'],
    'default_priority': TaskPriority.MEDIUM,
    'context_modifiers': {
        'time_sensitive': TaskPriority.HIGH,
        'resource_intensive': TaskPriority.MEDIUM,
        'background_process': TaskPriority.LOW
    }
}

# Memoization configuration for recurring tasks
MEMOIZATION_CONFIG = {
    'enabled': True,
    'max_cache_size': 1000,
    'ttl_seconds': 3600,  # 1 hour
    'similarity_threshold': 0.95,
    'cache_hit_speedup': 10.0
}

# Adaptive parallelism configuration
PARALLELISM_CONFIG = {
    'min_chunk_size': 10,
    'max_chunks': 16,
    'load_balancing_factor': 0.8,
    'resource_utilization_target': 0.85,
    'dynamic_adjustment': True
}

# Queue monitoring thresholds
QUEUE_MONITORING = {
    'warning_threshold': 0.7,  # Warn when queue is 70% full
    'critical_threshold': 0.9,  # Critical when queue is 90% full
    'backpressure_threshold': 0.95,  # Apply backpressure at 95%
    'metrics_interval_seconds': 30
}

# External service endpoints for the optimization engine
SERVICE_ENDPOINTS = {
    'resource_manager': 'tcp://localhost:5555',
    'load_balancer': 'tcp://localhost:5556',
    'execution_engine': 'tcp://localhost:5557',
    'monitoring_service': 'tcp://localhost:5558'
}

# Performance optimization flags
OPTIMIZATION_FLAGS = {
    'use_graph_optimization': True,
    'enable_critical_path_minimization': True,
    'apply_adaptive_chunking': True,
    'use_predictive_scaling': True,
    'enable_task_fusion': True  # Combine small related tasks
}