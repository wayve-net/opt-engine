# config.py
"""
Configuration module for dataset builder.
Separates patterns, templates, and constants from core logic.
"""

import re
from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum

class ResourceType(Enum):
    CPU = "CPU"
    GPU = "GPU"
    IO = "I/O"
    NETWORK = "NETWORK"
    MEMORY = "MEMORY"

class CriticalityLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ComplexityLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"

@dataclass
class TimingRange:
    min_us: int
    max_us: int
    typical_us: int

@dataclass
class StepTemplate:
    action: str
    resource: ResourceType
    criticality: CriticalityLevel
    timing: TimingRange
    parallelizable: bool = False
    dependencies: List[int] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

# Category patterns (precompiled regex)
CATEGORY_PATTERNS = {
    "socket_programming": re.compile(r'\b(socket|bind|listen|accept|connect|send|recv|sendall|recvfrom)\b', re.IGNORECASE),
    "http_requests": re.compile(r'\b(requests|urllib|http\.client|httplib|GET|POST|PUT|DELETE)\b', re.IGNORECASE),
    "async_networking": re.compile(r'\b(async|await|asyncio|aiohttp|aiofiles|uvloop)\b', re.IGNORECASE),
    "api_clients": re.compile(r'\b(api|rest|json|endpoint|client|graphql|oauth)\b', re.IGNORECASE),
    "websockets": re.compile(r'\b(websocket|ws:|wss:|socket\.io)\b', re.IGNORECASE),
    "network_security": re.compile(r'\b(ssl|tls|cert|crypto|auth|jwt|oauth|https)\b', re.IGNORECASE),
    "dns_resolution": re.compile(r'\b(dns|resolver|nslookup|dig|getaddrinfo|gethostbyname)\b', re.IGNORECASE),
    "tcp_udp": re.compile(r'\b(tcp|udp|datagram|stream|SOCK_STREAM|SOCK_DGRAM)\b', re.IGNORECASE),
    "network_monitoring": re.compile(r'\b(pcap|wireshark|tcpdump|netstat|iperf|ping|traceroute)\b', re.IGNORECASE),
    "proxy_tunneling": re.compile(r'\b(proxy|tunnel|socks|http_proxy|vpn|tor)\b', re.IGNORECASE),
    "quic_http3": re.compile(r'\b(quic|http3|h3|aioquic)\b', re.IGNORECASE),
    "grpc": re.compile(r'\b(grpc|protobuf|proto|rpc)\b', re.IGNORECASE)
}

# Precompiled extraction patterns
FUNCTION_PATTERN = re.compile(r'(?:def|function|func)\s+(\w+)\s*\(', re.IGNORECASE)
CLASS_PATTERN = re.compile(r'class\s+(\w+)(?:\([^)]*\))?:', re.IGNORECASE)
IMPORT_PATTERN = re.compile(r'(?:import\s+(\w+(?:\.\w+)*)|from\s+(\w+(?:\.\w+)*)\s+import)', re.IGNORECASE)

# Keyword classifications for resource type inference
IO_KEYWORDS = [
    "send", "receive", "read", "write", "download", "upload", "fetch",
    "connect", "listen", "accept", "bind", "lookup", "request", "response",
    "socket", "network", "tcp", "udp", "http", "dns", "ping", "curl",
    "stream", "buffer", "file", "disk", "database", "cache"
]

GPU_KEYWORDS = [
    "render", "visualize", "cryptographic", "hash", "encrypt", "decrypt",
    "complex calculation", "parallel", "matrix", "tensor", "cuda", "opencl",
    "gpu", "acceleration", "compute", "shader"
]

CPU_KEYWORDS = [
    "parse", "process", "calculate", "analyze", "decompress", "compress",
    "verify", "validate", "prepare", "generate", "create", "construct",
    "compile", "serialize", "deserialize", "transform", "sort", "search",
    "algorithm", "compute", "execute"
]

MEMORY_KEYWORDS = [
    "malloc", "alloc", "free", "garbage", "gc", "memory", "heap", "stack",
    "buffer", "cache", "pool", "mmap"
]

NETWORK_KEYWORDS = [
    "bandwidth", "latency", "throughput", "congestion", "routing", "switching",
    "firewall", "load balancer", "cdn", "dns", "dhcp"
]

# Step templates for different categories
STEP_TEMPLATES = {
    "http_requests": [
        StepTemplate("Parse URL and extract components", ResourceType.CPU, CriticalityLevel.MEDIUM, 
                    TimingRange(20, 100, 50)),
        StepTemplate("Prepare request headers and payload", ResourceType.CPU, CriticalityLevel.MEDIUM,
                    TimingRange(30, 150, 80)),
        StepTemplate("Resolve DNS for hostname", ResourceType.IO, CriticalityLevel.HIGH,
                    TimingRange(5000, 50000, 15000), parallelizable=True),
        StepTemplate("Establish TCP connection", ResourceType.IO, CriticalityLevel.HIGH,
                    TimingRange(10000, 200000, 50000), dependencies=[2]),
        StepTemplate("Send HTTP request", ResourceType.IO, CriticalityLevel.HIGH,
                    TimingRange(1000, 20000, 5000), dependencies=[3]),
        StepTemplate("Receive HTTP response headers", ResourceType.IO, CriticalityLevel.HIGH,
                    TimingRange(2000, 100000, 20000), dependencies=[4]),
        StepTemplate("Receive and buffer response body", ResourceType.IO, CriticalityLevel.MEDIUM,
                    TimingRange(5000, 500000, 100000), dependencies=[5]),
        StepTemplate("Parse response and handle status codes", ResourceType.CPU, CriticalityLevel.MEDIUM,
                    TimingRange(50, 500, 200), dependencies=[6])
    ],
    
    "socket_programming": [
        StepTemplate("Create socket object", ResourceType.CPU, CriticalityLevel.MEDIUM,
                    TimingRange(10, 50, 25)),
        StepTemplate("Configure socket options", ResourceType.CPU, CriticalityLevel.LOW,
                    TimingRange(20, 100, 60)),
        StepTemplate("Bind socket to address and port", ResourceType.IO, CriticalityLevel.HIGH,
                    TimingRange(100, 1000, 300), dependencies=[0, 1]),
        StepTemplate("Listen for incoming connections", ResourceType.IO, CriticalityLevel.HIGH,
                    TimingRange(50, 500, 150), dependencies=[2]),
        StepTemplate("Accept client connection", ResourceType.IO, CriticalityLevel.HIGH,
                    TimingRange(1000, 50000, 10000), dependencies=[3]),
        StepTemplate("Send/receive data packets", ResourceType.IO, CriticalityLevel.HIGH,
                    TimingRange(500, 100000, 20000), dependencies=[4]),
        StepTemplate("Handle connection cleanup and close", ResourceType.CPU, CriticalityLevel.MEDIUM,
                    TimingRange(100, 1000, 400), dependencies=[5])
    ],
    
    "async_networking": [
        StepTemplate("Initialize event loop", ResourceType.CPU, CriticalityLevel.HIGH,
                    TimingRange(100, 1000, 400)),
        StepTemplate("Create async context manager", ResourceType.CPU, CriticalityLevel.MEDIUM,
                    TimingRange(50, 300, 150)),
        StepTemplate("Schedule multiple coroutines", ResourceType.CPU, CriticalityLevel.MEDIUM,
                    TimingRange(200, 2000, 800), parallelizable=True, dependencies=[0, 1]),
        StepTemplate("Handle concurrent I/O operations", ResourceType.IO, CriticalityLevel.HIGH,
                    TimingRange(1000, 200000, 50000), parallelizable=True, dependencies=[2]),
        StepTemplate("Aggregate results from coroutines", ResourceType.CPU, CriticalityLevel.MEDIUM,
                    TimingRange(100, 1000, 500), dependencies=[3]),
        StepTemplate("Cleanup async resources and event loop", ResourceType.CPU, CriticalityLevel.MEDIUM,
                    TimingRange(200, 2000, 600), dependencies=[4])
    ],
    
    "network_security": [
        StepTemplate("Initialize SSL/TLS context", ResourceType.CPU, CriticalityLevel.HIGH,
                    TimingRange(500, 5000, 2000)),
        StepTemplate("Load certificates and private keys", ResourceType.IO, CriticalityLevel.CRITICAL,
                    TimingRange(1000, 10000, 4000)),
        StepTemplate("Perform certificate chain validation", ResourceType.CPU, CriticalityLevel.CRITICAL,
                    TimingRange(2000, 20000, 8000), dependencies=[1]),
        StepTemplate("Execute cryptographic handshake", ResourceType.CPU, CriticalityLevel.CRITICAL,
                    TimingRange(5000, 50000, 20000), dependencies=[0, 2]),
        StepTemplate("Establish secure communication channel", ResourceType.IO, CriticalityLevel.HIGH,
                    TimingRange(1000, 15000, 6000), dependencies=[3]),
        StepTemplate("Encrypt/decrypt data packets", ResourceType.CPU, CriticalityLevel.HIGH,
                    TimingRange(100, 5000, 1500), dependencies=[4])
    ],
    
    "dns_resolution": [
        StepTemplate("Parse and validate domain name", ResourceType.CPU, CriticalityLevel.MEDIUM,
                    TimingRange(20, 200, 80)),
        StepTemplate("Check local DNS cache", ResourceType.MEMORY, CriticalityLevel.LOW,
                    TimingRange(10, 100, 30)),
        StepTemplate("Query configured DNS resolver", ResourceType.IO, CriticalityLevel.HIGH,
                    TimingRange(5000, 100000, 25000), parallelizable=True),
        StepTemplate("Handle recursive DNS resolution", ResourceType.IO, CriticalityLevel.HIGH,
                    TimingRange(10000, 300000, 80000), dependencies=[2]),
        StepTemplate("Parse DNS response packets", ResourceType.CPU, CriticalityLevel.MEDIUM,
                    TimingRange(50, 500, 200), dependencies=[3]),
        StepTemplate("Cache DNS results", ResourceType.MEMORY, CriticalityLevel.LOW,
                    TimingRange(30, 300, 100), dependencies=[4])
    ],
    
    "websockets": [
        StepTemplate("Send HTTP upgrade request", ResourceType.IO, CriticalityLevel.HIGH,
                    TimingRange(1000, 10000, 4000)),
        StepTemplate("Validate WebSocket upgrade headers", ResourceType.CPU, CriticalityLevel.HIGH,
                    TimingRange(100, 1000, 400), dependencies=[0]),
        StepTemplate("Complete WebSocket handshake protocol", ResourceType.IO, CriticalityLevel.HIGH,
                    TimingRange(2000, 20000, 8000), dependencies=[1]),
        StepTemplate("Establish bidirectional communication", ResourceType.IO, CriticalityLevel.MEDIUM,
                    TimingRange(500, 5000, 2000), dependencies=[2]),
        StepTemplate("Handle WebSocket frame encoding/decoding", ResourceType.CPU, CriticalityLevel.MEDIUM,
                    TimingRange(50, 1000, 300), dependencies=[3]),
        StepTemplate("Manage connection keepalive and ping/pong", ResourceType.IO, CriticalityLevel.LOW,
                    TimingRange(1000, 30000, 10000), parallelizable=True, dependencies=[3])
    ]
}

# Synthetic operation templates for comprehensive coverage
SYNTHETIC_OPERATIONS = {
    "TCP 3-way handshake": {
        "category": "tcp_connection_management",
        "complexity": ComplexityLevel.MEDIUM,
        "templates": [
            StepTemplate("Client sends SYN packet with ISN", ResourceType.IO, CriticalityLevel.HIGH,
                        TimingRange(100, 2000, 600)),
            StepTemplate("Server receives SYN and allocates TCB", ResourceType.CPU, CriticalityLevel.HIGH,
                        TimingRange(50, 500, 200), dependencies=[0]),
            StepTemplate("Server sends SYN-ACK response", ResourceType.IO, CriticalityLevel.HIGH,
                        TimingRange(100, 2000, 600), dependencies=[1]),
            StepTemplate("Client receives SYN-ACK and sends ACK", ResourceType.IO, CriticalityLevel.HIGH,
                        TimingRange(100, 2000, 600), dependencies=[2]),
            StepTemplate("Connection established and ready", ResourceType.CPU, CriticalityLevel.MEDIUM,
                        TimingRange(20, 200, 80), dependencies=[3])
        ]
    },
    
    "QUIC Connection Establishment": {
        "category": "quic_http3",
        "complexity": ComplexityLevel.HIGH,
        "templates": [
            StepTemplate("Generate connection ID and initial keys", ResourceType.CPU, CriticalityLevel.HIGH,
                        TimingRange(500, 5000, 2000)),
            StepTemplate("Send Initial packet with TLS ClientHello", ResourceType.IO, CriticalityLevel.HIGH,
                        TimingRange(1000, 10000, 4000), dependencies=[0]),
            StepTemplate("Receive and process server Initial packet", ResourceType.IO, CriticalityLevel.HIGH,
                        TimingRange(1000, 15000, 6000), dependencies=[1]),
            StepTemplate("Complete TLS handshake over QUIC", ResourceType.CPU, CriticalityLevel.CRITICAL,
                        TimingRange(5000, 50000, 20000), dependencies=[2]),
            StepTemplate("Establish 1-RTT protected connection", ResourceType.IO, CriticalityLevel.HIGH,
                        TimingRange(500, 8000, 3000), dependencies=[3])
        ]
    },
    
    "gRPC Service Call": {
        "category": "grpc",
        "complexity": ComplexityLevel.MEDIUM,
        "templates": [
            StepTemplate("Serialize protobuf message", ResourceType.CPU, CriticalityLevel.MEDIUM,
                        TimingRange(100, 2000, 600)),
            StepTemplate("Establish HTTP/2 stream", ResourceType.IO, CriticalityLevel.HIGH,
                        TimingRange(500, 8000, 3000)),
            StepTemplate("Send gRPC headers and metadata", ResourceType.IO, CriticalityLevel.MEDIUM,
                        TimingRange(200, 3000, 1000), dependencies=[0, 1]),
            StepTemplate("Stream request message", ResourceType.IO, CriticalityLevel.MEDIUM,
                        TimingRange(1000, 50000, 15000), dependencies=[2]),
            StepTemplate("Receive and deserialize response", ResourceType.CPU, CriticalityLevel.MEDIUM,
                        TimingRange(200, 3000, 1000), dependencies=[3]),
            StepTemplate("Handle gRPC status and trailers", ResourceType.CPU, CriticalityLevel.LOW,
                        TimingRange(50, 500, 200), dependencies=[4])
        ]
    }
}

# Dataset balancing configuration
DATASET_BALANCE_CONFIG = {
    "min_examples_per_category": 5,
    "max_examples_per_category": 100,
    "preferred_complexity_distribution": {
        ComplexityLevel.LOW: 0.2,
        ComplexityLevel.MEDIUM: 0.5,
        ComplexityLevel.HIGH: 0.25,
        ComplexityLevel.EXTREME: 0.05
    }
}

# File processing configuration
FILE_PROCESSING_CONFIG = {
    "max_code_length": 3000,  # characters
    "min_code_length": 100,   # characters
    "max_examples_per_source": 50,
    "deduplication_enabled": True,
    "parallel_processing": True,
    "max_workers": 4
}