# generators.py
"""
Fixed generator classes with correct imports.
"""

import hashlib
import random
from typing import List, Dict, Optional, Any, Set
from abc import ABC, abstractmethod
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from data_types import (
    TrainingExample, Decomposition, DecompositionStep, CodeAnalysis,
    SourceType, ProcessingConfig
)
from data_config import (
    CATEGORY_PATTERNS, FUNCTION_PATTERN, CLASS_PATTERN, IMPORT_PATTERN,
    IO_KEYWORDS, GPU_KEYWORDS, CPU_KEYWORDS, MEMORY_KEYWORDS, NETWORK_KEYWORDS,
    ResourceType, CriticalityLevel, ComplexityLevel, STEP_TEMPLATES,
    SYNTHETIC_OPERATIONS, StepTemplate, TimingRange  # Fixed imports
)

logger = logging.getLogger(__name__)

class BaseGenerator(ABC):
    """Base class for all training example generators."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.seen_hashes: Set[str] = set()
        self.generated_count = 0
        
    @abstractmethod
    def generate_examples(self, data: List[Dict]) -> List[TrainingExample]:
        """Generate training examples from input data."""
        pass
    
    def _categorize_code(self, code: str, metadata: Dict = None) -> str:
        """Enhanced code categorization using precompiled patterns."""
        if not code:
            return "general_networking"
            
        # Check metadata for additional context
        if metadata:
            file_path = metadata.get('file_path', '').lower()
            repo = metadata.get('repo', '').lower()
            
            # Context-based categorization
            if any(sec in file_path for sec in ['ssl', 'tls', 'crypto']) or 'security' in repo:
                return "network_security"
            elif any(dns in file_path for dns in ['dns', 'resolver']):
                return "dns_resolution"
            elif any(ws in file_path for ws in ['websocket', 'ws']):
                return "websockets"
        
        # Pattern matching with priority ordering
        category_scores = {}
        for category, pattern in CATEGORY_PATTERNS.items():
            matches = len(pattern.findall(code))
            if matches > 0:
                category_scores[category] = matches
        
        if category_scores:
            return max(category_scores, key=category_scores.get)
        return "general_networking"
    
    def _analyze_code(self, code: str) -> CodeAnalysis:
        """Extract comprehensive code analysis."""
        functions = FUNCTION_PATTERN.findall(code)
        classes = CLASS_PATTERN.findall(code)
        imports = IMPORT_PATTERN.findall(code)
        
        # Flatten tuples from regex groups
        functions = [f for f in functions if f]
        classes = [c for c in classes if c]
        imports = [imp[0] or imp[1] for imp in imports if any(imp)]
        
        lines_of_code = len([line for line in code.split('\n') if line.strip()])
        
        # Simple complexity scoring based on control structures
        complexity_indicators = ['if', 'for', 'while', 'try', 'except', 'with', 'async', 'await']
        complexity_score = sum(code.lower().count(indicator) for indicator in complexity_indicators)
        complexity_score = min(complexity_score / max(lines_of_code, 1) * 10, 10.0)
        
        content_hash = hashlib.sha256(code.encode()).hexdigest()[:16]
        
        return CodeAnalysis(
            functions=functions,
            classes=classes,
            imports=imports,
            lines_of_code=lines_of_code,
            complexity_score=complexity_score,
            content_hash=content_hash
        )
    
    def _classify_resource_type(self, action: str) -> ResourceType:
        """Classify action by primary resource type."""
        action_lower = action.lower()
        
        if any(keyword in action_lower for keyword in IO_KEYWORDS):
            return ResourceType.IO
        elif any(keyword in action_lower for keyword in NETWORK_KEYWORDS):
            return ResourceType.NETWORK
        elif any(keyword in action_lower for keyword in GPU_KEYWORDS):
            return ResourceType.GPU
        elif any(keyword in action_lower for keyword in MEMORY_KEYWORDS):
            return ResourceType.MEMORY
        else:
            return ResourceType.CPU
    
    def _calculate_criticality(self, action: str, category: str) -> CriticalityLevel:
        """Determine criticality level based on action and category."""
        action_lower = action.lower()
        
        critical_keywords = ['verify', 'validate', 'authenticate', 'encrypt', 'decrypt', 'security']
        high_keywords = ['connect', 'send', 'receive', 'establish', 'handshake']
        low_keywords = ['initialize', 'prepare', 'cleanup', 'close']
        
        if any(keyword in action_lower for keyword in critical_keywords):
            return CriticalityLevel.CRITICAL
        elif any(keyword in action_lower for keyword in high_keywords):
            return CriticalityLevel.HIGH
        elif any(keyword in action_lower for keyword in low_keywords):
            return CriticalityLevel.LOW
        else:
            return CriticalityLevel.MEDIUM
    
    def _generate_realistic_timing(self, resource: ResourceType, criticality: CriticalityLevel, 
                                  base_complexity: float = 1.0) -> int:
        """Generate realistic timing estimates based on resource type and criticality."""
        base_times = {
            ResourceType.CPU: (10, 1000),      # 10μs - 1ms
            ResourceType.GPU: (100, 10000),    # 100μs - 10ms  
            ResourceType.IO: (1000, 500000),   # 1ms - 500ms
            ResourceType.NETWORK: (5000, 2000000),  # 5ms - 2s
            ResourceType.MEMORY: (5, 500)      # 5μs - 500μs
        }
        
        criticality_multipliers = {
            CriticalityLevel.LOW: 0.5,
            CriticalityLevel.MEDIUM: 1.0,
            CriticalityLevel.HIGH: 2.0,
            CriticalityLevel.CRITICAL: 4.0
        }
        
        min_time, max_time = base_times[resource]
        multiplier = criticality_multipliers[criticality] * base_complexity
        
        adjusted_min = int(min_time * multiplier)
        adjusted_max = int(max_time * multiplier)
        
        return random.randint(adjusted_min, adjusted_max)
    
    def _is_duplicate(self, content: str) -> bool:
        """Check for duplicate content using hashing."""
        if not self.config.enable_deduplication:
            return False
            
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        if content_hash in self.seen_hashes:
            return True
        self.seen_hashes.add(content_hash)
        return False

class GitHubGenerator(BaseGenerator):
    """Generator for GitHub repository code examples."""
    
    def generate_examples(self, github_data: List[Dict]) -> List[TrainingExample]:
        """Generate training examples from GitHub repository data."""
        logger.info(f"Processing {len(github_data)} GitHub files...")
        
        if self.config.enable_parallel_processing:
            return self._generate_parallel(github_data)
        else:
            return self._generate_sequential(github_data)
    
    def _generate_parallel(self, github_data: List[Dict]) -> List[TrainingExample]:
        """Generate examples using parallel processing."""
        examples = []
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(self._create_github_example, file_data): file_data 
                for file_data in github_data
            }
            
            for future in as_completed(futures):
                try:
                    example = future.result()
                    if example:
                        examples.append(example)
                except Exception as e:
                    logger.error(f"Error processing GitHub file: {e}")
        
        logger.info(f"Generated {len(examples)} GitHub examples")
        return examples
    
    def _generate_sequential(self, github_data: List[Dict]) -> List[TrainingExample]:
        """Generate examples sequentially."""
        examples = []
        
        for file_data in github_data:
            try:
                example = self._create_github_example(file_data)
                if example:
                    examples.append(example)
            except Exception as e:
                logger.error(f"Error processing GitHub file: {e}")
        
        logger.info(f"Generated {len(examples)} GitHub examples")
        return examples
    
    def _create_github_example(self, file_data: Dict) -> Optional[TrainingExample]:
        """Create a training example from GitHub file data."""
        code = file_data.get("content", "")
        file_path = file_data.get("file_path", "")
        repo = file_data.get("repo", "")
        
        # Validate code length
        if not (self.config.min_code_length <= len(code) <= self.config.max_code_length):
            return None
        
        # Check for duplicates
        if self._is_duplicate(code):
            return None
        
        # Extract metadata
        metadata = {
            "file_path": file_path,
            "repo": repo,
            "imports": file_data.get("imports", []),
            "function_signatures": file_data.get("function_signatures", [])
        }
        
        category = self._categorize_code(code, metadata)
        code_analysis = self._analyze_code(code)
        
        # Generate decomposition with realistic steps
        decomposition = self._generate_enhanced_decomposition(code, category, code_analysis)
        
        # Create training example
        code_sample = self._truncate_code(code)
        
        example = TrainingExample(
            instruction=f"Analyze and decompose this {category} code from a production repository",
            input=f"Repository: {repo}\nFile: {file_path}\n\nCode:\n{code_sample}",
            output=decomposition,
            metadata={
                "source": SourceType.GITHUB.value,
                "category": category,
                "complexity": decomposition.complexity,
                "code_analysis": code_analysis,
                "repo_context": {"repo": repo, "file_path": file_path}
            }
        )
        
        self.generated_count += 1
        return example
    
    def _truncate_code(self, code: str) -> str:
        """Intelligently truncate code while preserving structure."""
        if len(code) <= self.config.max_code_length:
            return code
        
        # Try to truncate at function boundaries
        lines = code.split('\n')
        truncated_lines = []
        current_length = 0
        
        for line in lines:
            if current_length + len(line) > self.config.max_code_length - 100:  # Leave room for "..."
                truncated_lines.append("    # ... (truncated)")
                break
            truncated_lines.append(line)
            current_length += len(line) + 1  # +1 for newline
        
        return '\n'.join(truncated_lines)
    
    def _generate_enhanced_decomposition(self, code: str, category: str, 
                                       analysis: CodeAnalysis) -> Decomposition:
        """Generate enhanced decomposition with realistic timing and dependencies."""
        
        # Get base template or create default
        templates = STEP_TEMPLATES.get(category, [])
        if not templates:
            templates = self._create_generic_templates(category, analysis)
        
        steps = []
        total_time = 0
        parallelizable_steps = []
        
        for i, template in enumerate(templates):
            # Generate realistic timing based on code complexity
            timing_us = self._generate_realistic_timing(
                ResourceType(template.resource.value),
                CriticalityLevel(template.criticality.value),
                analysis.complexity_score / 5.0  # Normalize complexity
            )
            
            step = DecompositionStep(
                step=i + 1,
                action=template.action,
                expected_time=f"{timing_us}μs",
                resource=template.resource.value,
                criticality=template.criticality.value,
                parallelizable=template.parallelizable,
                dependencies=template.dependencies.copy()
            )
            
            steps.append(step)
            total_time += timing_us
            
            if template.parallelizable:
                parallelizable_steps.append(i + 1)
        
        # Calculate critical path
        critical_path_time = self._calculate_critical_path(steps)
        
        # Determine complexity based on steps and code analysis
        complexity = self._determine_complexity(len(steps), analysis.complexity_score)
        
        return Decomposition(
            operation=f"{category.replace('_', ' ').title()} Implementation",
            category=category,
            steps=steps,
            complexity=complexity.value,
            estimated_total_time=f"{total_time}μs",
            parallelizable_steps=parallelizable_steps,
            critical_path_length=f"{critical_path_time}μs",
            metadata={
                "functions_found": analysis.functions,
                "classes_found": analysis.classes,
                "imports": analysis.imports,
                "lines_of_code": analysis.lines_of_code,
                "complexity_score": analysis.complexity_score
            }
        )
    
    def _create_generic_templates(self, category: str, analysis: CodeAnalysis):
        """Create generic templates when specific ones aren't available."""
        # Fixed import reference
        base_actions = [
            "Initialize components and variables",
            "Process input data and parameters", 
            "Execute core networking operations",
            "Handle response and error conditions",
            "Cleanup resources and connections"
        ]
        
        templates = []
        for i, action in enumerate(base_actions):
            resource = ResourceType.IO if "network" in action.lower() else ResourceType.CPU
            criticality = CriticalityLevel.HIGH if "error" in action.lower() else CriticalityLevel.MEDIUM
            
            templates.append(StepTemplate(
                action=action,
                resource=resource,
                criticality=criticality,
                timing=TimingRange(100, 5000, 1000),
                dependencies=[i-1] if i > 0 else []
            ))
        
        return templates
    
    def _calculate_critical_path(self, steps: List[DecompositionStep]) -> int:
        """Calculate critical path timing considering dependencies."""
        step_times = {}
        
        for step in steps:
            step_time = int(step.expected_time.replace('μs', ''))
            
            if not step.dependencies:
                step_times[step.step] = step_time
            else:
                max_dep_time = max(step_times.get(dep, 0) for dep in step.dependencies)
                step_times[step.step] = max_dep_time + step_time
        
        return max(step_times.values()) if step_times else 0
    
    def _determine_complexity(self, step_count: int, code_complexity: float) -> ComplexityLevel:
        """Determine overall complexity based on multiple factors."""
        if step_count >= 8 or code_complexity >= 7.0:
            return ComplexityLevel.EXTREME
        elif step_count >= 6 or code_complexity >= 5.0:
            return ComplexityLevel.HIGH
        elif step_count >= 4 or code_complexity >= 3.0:
            return ComplexityLevel.MEDIUM
        else:
            return ComplexityLevel.LOW


class DocumentationGenerator(BaseGenerator):
    """Generator for documentation examples."""
    
    def generate_examples(self, docs_data: List[Dict]) -> List[TrainingExample]:
        """Generate training examples from documentation sources."""
        logger.info(f"Processing {len(docs_data)} documentation sources...")
        
        examples = []
        for doc_data in docs_data:
            try:
                doc_examples = self._create_documentation_examples(doc_data)
                examples.extend(doc_examples)
            except Exception as e:
                logger.error(f"Error processing documentation: {e}")
        
        logger.info(f"Generated {len(examples)} documentation examples")
        return examples
    
    def _create_documentation_examples(self, doc_data: Dict) -> List[TrainingExample]:
        """Create training examples from documentation content."""
        examples = []
        content = doc_data.get("content", {})
        url = doc_data.get("url", "")
        tags = doc_data.get("tags", [])
        
        code_blocks = content.get("code", [])
        text_content = content.get("text", "")
        
        for i, code_block in enumerate(code_blocks[:3]):  # Limit to 3 per doc
            if len(code_block.strip()) < 50:
                continue
                
            if self._is_duplicate(code_block):
                continue
            
            category = self._categorize_code(code_block)
            analysis = self._analyze_code(code_block)
            
            # Create documentation-specific decomposition
            decomposition = self._create_documentation_decomposition(
                code_block, category, analysis, url, tags
            )
            
            example = TrainingExample(
                instruction="Explain and decompose this documented network programming pattern",
                input=f"Documentation source: {url}\n\nCode example:\n{code_block}",
                output=decomposition,
                metadata={
                    "source": SourceType.DOCUMENTATION.value,
                    "category": "reference_material",
                    "url": url,
                    "tags": tags
                }
            )
            
            examples.append(example)
            self.generated_count += 1
        
        return examples
    
    def _create_documentation_decomposition(self, code: str, category: str, 
                                          analysis: CodeAnalysis, url: str, 
                                          tags: List[str]) -> Decomposition:
        """Create decomposition for documentation examples."""
        
        steps = [
            DecompositionStep(
                step=1,
                action="Study the documented concept and API usage",
                expected_time="200μs",
                resource=ResourceType.CPU.value,
                criticality=CriticalityLevel.MEDIUM.value
            ),
            DecompositionStep(
                step=2,
                action="Understand implementation patterns and best practices",
                expected_time="400μs", 
                resource=ResourceType.CPU.value,
                criticality=CriticalityLevel.HIGH.value,
                dependencies=[1]
            ),
            DecompositionStep(
                step=3,
                action="Apply knowledge to practical implementation",
                expected_time="600μs",
                resource=ResourceType.CPU.value,
                criticality=CriticalityLevel.HIGH.value,
                dependencies=[2]
            )
        ]
        
        return Decomposition(
            operation=f"Documentation Study: {category.title()}",
            category="documentation_reference",
            steps=steps,
            complexity=ComplexityLevel.LOW.value,
            estimated_total_time="1200μs",
            parallelizable_steps=[],
            critical_path_length="1200μs",
            metadata={
                "reference_url": url,
                "tags": tags,
                "code_analysis": analysis
            }
        )


class TutorialGenerator(BaseGenerator):
    """Generator for tutorial examples."""
    
    def generate_examples(self, tutorial_data: List[Dict]) -> List[TrainingExample]:
        """Generate training examples from tutorial content."""
        logger.info(f"Processing {len(tutorial_data)} tutorial sources...")
        
        examples = []
        for tutorial in tutorial_data:
            try:
                tutorial_examples = self._create_tutorial_examples(tutorial)
                examples.extend(tutorial_examples)
            except Exception as e:
                logger.error(f"Error processing tutorial: {e}")
        
        logger.info(f"Generated {len(examples)} tutorial examples")
        return examples
    
    def _create_tutorial_examples(self, tutorial: Dict) -> List[TrainingExample]:
        """Create training examples from tutorial content."""
        examples = []
        content = tutorial.get("content", {})
        url = tutorial.get("url", "")
        descriptions = tutorial.get("natural_language_description", [])
        
        code_blocks = content.get("code", [])
        
        for i, code_block in enumerate(code_blocks[:2]):  # Limit to 2 per tutorial
            if len(code_block.strip()) < 50:
                continue
                
            if self._is_duplicate(code_block):
                continue
            
            description = descriptions[i] if i < len(descriptions) else "Tutorial example"
            category = self._categorize_code(code_block)
            analysis = self._analyze_code(code_block)
            
            decomposition = self._create_tutorial_decomposition(
                code_block, category, analysis, description, url
            )
            
            example = TrainingExample(
                instruction="Break down this tutorial example into structured learning steps",
                input=f"Tutorial context: {description}\n\nTutorial source: {url}\n\nCode example:\n{code_block}",
                output=decomposition,
                metadata={
                    "source": SourceType.TUTORIAL.value,
                    "category": "educational_breakdown",
                    "tutorial_url": url,
                    "learning_context": description
                }
            )
            
            examples.append(example)
            self.generated_count += 1
        
        return examples
    
    def _create_tutorial_decomposition(self, code: str, category: str, 
                                     analysis: CodeAnalysis, description: str,
                                     url: str) -> Decomposition:
        """Create decomposition for tutorial examples."""
        
        steps = [
            DecompositionStep(
                step=1,
                action="Understand the problem context and requirements",
                expected_time="150μs",
                resource=ResourceType.CPU.value,
                criticality=CriticalityLevel.MEDIUM.value
            ),
            DecompositionStep(
                step=2,
                action="Break down the solution approach and methodology",
                expected_time="300μs",
                resource=ResourceType.CPU.value,
                criticality=CriticalityLevel.HIGH.value,
                dependencies=[1]
            ),
            DecompositionStep(
                step=3,
                action="Implement and test the solution",
                expected_time="800μs",
                resource=ResourceType.IO.value,
                criticality=CriticalityLevel.HIGH.value,
                dependencies=[2]
            ),
            DecompositionStep(
                step=4,
                action="Validate results and handle edge cases",
                expected_time="400μs",
                resource=ResourceType.CPU.value,
                criticality=CriticalityLevel.MEDIUM.value,
                dependencies=[3]
            )
        ]
        
        return Decomposition(
            operation=f"Tutorial Learning: {category.title()}",
            category="educational_content",
            steps=steps,
            complexity=ComplexityLevel.MEDIUM.value,
            estimated_total_time="1650μs",
            parallelizable_steps=[],
            critical_path_length="1650μs",
            metadata={
                "learning_context": description,
                "tutorial_url": url,
                "code_analysis": analysis
            }
        )


class SyntheticGenerator(BaseGenerator):
    """Generator for synthetic operation examples."""
    
    def generate_examples(self, operation_count: int = None) -> List[TrainingExample]:
        """Generate synthetic operation examples."""
        logger.info(f"Generating synthetic operation examples...")
        
        examples = []
        for operation_name, operation_data in SYNTHETIC_OPERATIONS.items():
            try:
                example = self._create_synthetic_example(operation_name, operation_data)
                examples.append(example)
                self.generated_count += 1
            except Exception as e:
                logger.error(f"Error creating synthetic example for {operation_name}: {e}")
        
        logger.info(f"Generated {len(examples)} synthetic examples")
        return examples
    
    def _create_synthetic_example(self, operation_name: str, 
                                operation_data: Dict) -> TrainingExample:
        """Create a synthetic training example."""
        
        templates = operation_data["templates"]
        category = operation_data["category"]
        complexity = operation_data["complexity"]
        
        steps = []
        total_time = 0
        parallelizable_steps = []
        
        for i, template in enumerate(templates):
            timing_us = self._generate_realistic_timing(
                template.resource,
                template.criticality
            )
            
            step = DecompositionStep(
                step=i + 1,
                action=template.action,
                expected_time=f"{timing_us}μs",
                resource=template.resource.value,
                criticality=template.criticality.value,
                parallelizable=template.parallelizable,
                dependencies=template.dependencies.copy()
            )
            
            steps.append(step)
            total_time += timing_us
            
            if template.parallelizable:
                parallelizable_steps.append(i + 1)
        
        critical_path_time = self._calculate_critical_path(steps)
        
        decomposition = Decomposition(
            operation=operation_name,
            category=category,
            steps=steps,
            complexity=complexity.value,
            estimated_total_time=f"{total_time}μs",
            parallelizable_steps=parallelizable_steps,
            critical_path_length=f"{critical_path_time}μs",
            metadata={"synthetic": True}
        )
        
        return TrainingExample(
            instruction=f"Decompose the network operation: {operation_name}",
            input=f"Analyze the step-by-step process for: {operation_name}",
            output=decomposition,
            metadata={
                "source": SourceType.SYNTHETIC.value,
                "category": category,
                "complexity": complexity.value
            }
        )
    
    def _calculate_critical_path(self, steps: List[DecompositionStep]) -> int:
        """Calculate critical path timing considering dependencies."""
        step_times = {}
        
        for step in steps:
            step_time = int(step.expected_time.replace('μs', ''))
            
            if not step.dependencies:
                step_times[step.step] = step_time
            else:
                max_dep_time = max(step_times.get(dep, 0) for dep in step.dependencies)
                step_times[step.step] = max_dep_time + step_time
        
        return max(step_times.values()) if step_times else 0