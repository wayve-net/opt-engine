# data_types.py
"""
Type definitions and data structures for the dataset builder.
Uses dataclasses and TypedDicts for better structure consistency.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, TypedDict
from enum import Enum
import hashlib
from data_config import ResourceType, CriticalityLevel, ComplexityLevel

class SourceType(Enum):
    GITHUB = "github_repository"
    DOCUMENTATION = "documentation"
    TUTORIAL = "tutorial"
    SYNTHETIC = "synthetic_generation"
    TRACES = "execution_traces"
    DLOG_EXPERIMENTS = "dlog_experiments"

@dataclass
class CodeAnalysis:
    """Structure for analyzed code metadata."""
    functions: List[str] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    lines_of_code: int = 0
    complexity_score: float = 0.0
    content_hash: str = ""

    def __post_init__(self):
        if not self.content_hash and hasattr(self, '_content'):
            self.content_hash = hashlib.sha256(self._content.encode()).hexdigest()[:16]

# --- Deprecated for the DLOG PIPELINE ---
@dataclass
class DecompositionStep:
    """Structure for a single decomposition step."""
    step: int
    action: str
    expected_time: str
    resource: str
    criticality: str
    parallelizable: bool = False
    dependencies: List[int] = field(default_factory=list)

@dataclass
class Decomposition:
    """Structure for complete operation decomposition."""
    operation: str
    category: str
    steps: List[DecompositionStep]
    complexity: str
    estimated_total_time: str
    parallelizable_steps: List[int] = field(default_factory=list)
    critical_path_length: str = "0μs"
    metadata: Dict[str, Any] = field(default_factory=dict)
# ---------------------------------------

# --- NEW DATA TYPES FOR THE DLOG PROBLEM ---
@dataclass
class HybridPivotSelectionData:
    """
    Structured output for a model trained to predict optimal pivots
    for the hybrid BMS-SP solver.
    """
    # A list of recommended pivots, sorted from best to worst.
    recommended_pivots: List[int] = field(default_factory=list)
    
    # Heuristics or features used to rank the pivots.
    metadata: Dict[str, Any] = field(default_factory=dict)
    
@dataclass
class TrainingExample:
    """
    The core training example structure. Now updated to be flexible
    for different output types.
    """
    instruction: str
    input: Dict[str, Any]
    output: Any # Can be Decomposition or HybridPivotSelectionData
    metadata: Dict[str, Any] = field(default_factory=dict)

# TypedDicts for raw data structures (for JSON compatibility)
class GitHubFileData(TypedDict):
    source: str
    repo: str
    file_path: str
    content: str
    imports: List[str]
    category: str
    function_signatures: List[str]
    execution_traces: str

class DocumentationData(TypedDict):
    source: str
    url: str
    content: Dict[str, Any]
    tags: List[str]

class TutorialData(TypedDict):
    source: str
    url: str
    content: Dict[str, Any]
    natural_language_description: List[str]

class TracesData(TypedDict):
    source: str
    executable_path: str
    syscalls: List[Dict[str, Any]]
    timing_data: Dict[str, float]
    resource_usage: Dict[str, Any]

@dataclass
class ProcessingConfig:
    """Configuration for processing"""
    enable_deduplication: bool = True
    enable_parallel_processing: bool = False
    max_workers: int = 4
    max_code_length: int = 3000
    min_code_length: int = 100
    balance_categories: bool = False
    enhance_with_traces: bool = False
    target_examples: int = 1000
    
@dataclass
class DatasetStatistics:
    total_examples: int = 0
    source_distribution: Dict[str, int] = field(default_factory=dict)
    category_distribution: Dict[str, int] = field(default_factory=dict)
    complexity_distribution: Dict[str, int] = field(default_factory=dict)
    resource_usage: Dict[str, int] = field(default_factory=dict)
    avg_code_length: float = 0.0
    median_code_length: float = 0.0
    most_common_functions: List[tuple] = field(default_factory=list)
    most_common_classes: List[tuple] = field(default_factory=list)
    duplicate_count: int = 0
    processing_time_ms: float = 0.0




























