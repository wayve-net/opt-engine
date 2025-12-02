# training_data_generator.py
import random
from dataclasses import dataclass
from typing import List, Dict
from enum import Enum

class OperationType(Enum):
    CPU = 0
    GPU = 1
    IO = 2

@dataclass
class SubTask:
    name: str
    operation: OperationType
    estimate_time: str
    subtasks: List['SubTask'] = None

@dataclass 
class NetworkOperation:
    id: str
    description: str
    subtasks: List[SubTask]

def generate_network_operations_training_data():
    """Generate training examples using your schema structure"""
    
    training_examples = []
    
    # Example 1: HTTP GET Request
    http_get = NetworkOperation(
        id="http_get_001",
        description="Make HTTP GET request to REST API endpoint",
        subtasks=[
            SubTask(
                name="DNS Resolution", 
                operation=OperationType.IO,
                estimate_time="50ms",
                subtasks=[
                    SubTask("Query local DNS cache", OperationType.CPU, "100ns"),
                    SubTask("UDP query to DNS server", OperationType.IO, "45ms"),
                    SubTask("Parse DNS response", OperationType.CPU, "5ms")
                ]
            ),
            SubTask(
                name="TCP Connection Establishment",
                operation=OperationType.IO, 
                estimate_time="100ms",
                subtasks=[
                    SubTask("Create socket", OperationType.CPU, "10µs"),
                    SubTask("TCP 3-way handshake", OperationType.IO, "99ms"),
                    SubTask("Allocate buffers", OperationType.CPU, "1µs")
                ]
            ),
            SubTask(
                name="HTTP Request Construction",
                operation=OperationType.CPU,
                estimate_time="50µs", 
                subtasks=[
                    SubTask("Build headers", OperationType.CPU, "20µs"),
                    SubTask("Serialize request body", OperationType.CPU, "30µs")
                ]
            ),
            SubTask(
                name="Send Request",
                operation=OperationType.IO,
                estimate_time="5ms"
            ),
            SubTask(
                name="Receive Response", 
                operation=OperationType.IO,
                estimate_time="20ms",
                subtasks=[
                    SubTask("Read response headers", OperationType.IO, "2ms"),
                    SubTask("Read response body", OperationType.IO, "18ms")
                ]
            ),
            SubTask(
                name="Parse Response",
                operation=OperationType.CPU,
                estimate_time="200µs",
                subtasks=[
                    SubTask("Parse JSON", OperationType.CPU, "150µs"),
                    SubTask("Create response object", OperationType.CPU, "50µs")
                ]
            )
        ]
    )
    
    # Generate corresponding code
    http_get_code = '''
import requests
import time

def make_http_request(url):
    """
    Task: Make HTTP GET request to REST API endpoint
    Decomposition:
    1. DNS Resolution (50ms, I/O)
    2. TCP Connection (100ms, I/O) 
    3. HTTP Request Construction (50µs, CPU)
    4. Send Request (5ms, I/O)
    5. Receive Response (20ms, I/O)
    6. Parse Response (200µs, CPU)
    """
    try:
        response = requests.get(url, timeout=30)
        return response.json()
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return None
'''
    
    # Example 2: Socket Server
    socket_server = NetworkOperation(
        id="socket_server_001", 
        description="Create TCP socket server with connection handling",
        subtasks=[
            SubTask(
                name="Socket Creation",
                operation=OperationType.CPU,
                estimate_time="10µs"
            ),
            SubTask(
                name="Bind to Address",
                operation=OperationType.IO,
                estimate_time="100µs"
            ),
            SubTask(
                name="Listen for Connections",
                operation=OperationType.IO, 
                estimate_time="1ns",  # blocking call
                subtasks=[
                    SubTask("Set listen queue", OperationType.CPU, "1µs")
                ]
            ),
            SubTask(
                name="Accept Connection",
                operation=OperationType.IO,
                estimate_time="varies",  # blocking
                subtasks=[
                    SubTask("Accept system call", OperationType.IO, "50µs"),
                    SubTask("Create client socket", OperationType.CPU, "5µs")
                ]
            ),
            SubTask(
                name="Handle Client Data", 
                operation=OperationType.IO,
                estimate_time="varies",
                subtasks=[
                    SubTask("Receive data", OperationType.IO, "1-100ms"),
                    SubTask("Process data", OperationType.CPU, "10-1000µs"), 
                    SubTask("Send response", OperationType.IO, "1-50ms")
                ]
            )
        ]
    )
    
    socket_server_code = '''
import socket
import threading

def create_tcp_server(host='localhost', port=8080):
    """
    Task: Create TCP socket server with connection handling
    Decomposition:
    1. Socket Creation (10µs, CPU)
    2. Bind to Address (100µs, I/O)
    3. Listen for Connections (1ns, I/O blocking)
    4. Accept Connection (varies, I/O)
    5. Handle Client Data (varies, I/O + CPU)
    """
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server_socket.bind((host, port))
        server_socket.listen(5)
        print(f"Server listening on {host}:{port}")
        
        while True:
            client_socket, address = server_socket.accept()
            print(f"Connection from {address}")
            
            # Handle client in separate thread
            client_thread = threading.Thread(
                target=handle_client, 
                args=(client_socket,)
            )
            client_thread.start()
            
    except Exception as e:
        print(f"Server error: {e}")
    finally:
        server_socket.close()

def handle_client(client_socket):
    """Handle individual client connection"""
    try:
        while True:
            data = client_socket.recv(1024)
            if not data:
                break
            
            # Echo back the data
            client_socket.send(data)
            
    except Exception as e:
        print(f"Client error: {e}")
    finally:
        client_socket.close()
'''
    
    # Example 3: Async HTTP Client
    async_client = NetworkOperation(
        id="async_http_001",
        description="Asynchronous HTTP client with connection pooling",
        subtasks=[
            SubTask(
                name="Initialize Event Loop",
                operation=OperationType.CPU,
                estimate_time="100µs"
            ),
            SubTask(
                name="Create Connection Pool",
                operation=OperationType.CPU,
                estimate_time="50µs",
                subtasks=[
                    SubTask("Allocate pool memory", OperationType.CPU, "20µs"),
                    SubTask("Initialize connection limits", OperationType.CPU, "30µs")
                ]
            ),
            SubTask(
                name="Concurrent Requests",
                operation=OperationType.IO,
                estimate_time="varies",
                subtasks=[
                    SubTask("Schedule coroutines", OperationType.CPU, "10µs"),
                    SubTask("DNS resolution (cached)", OperationType.CPU, "1µs"),
                    SubTask("Reuse existing connections", OperationType.CPU, "5µs"),
                    SubTask("Parallel I/O operations", OperationType.IO, "20-200ms")
                ]
            ),
            SubTask(
                name="Response Aggregation",
                operation=OperationType.CPU, 
                estimate_time="100µs"
            )
        ]
    )
    
    async_client_code = '''
import asyncio
import aiohttp
from typing import List

async def fetch_multiple_urls(urls: List[str]):
    """
    Task: Asynchronous HTTP client with connection pooling
    Decomposition:
    1. Initialize Event Loop (100µs, CPU)
    2. Create Connection Pool (50µs, CPU)
    3. Concurrent Requests (varies, I/O + CPU)
    4. Response Aggregation (100µs, CPU)
    """
    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=100)
    ) as session:
        
        tasks = []
        for url in urls:
            task = asyncio.create_task(fetch_url(session, url))
            tasks.append(task)
        
        # Wait for all requests to complete
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        return responses

async def fetch_url(session, url):
    """Fetch single URL with error handling"""
    try:
        async with session.get(url) as response:
            return {
                'url': url,
                'status': response.status,
                'data': await response.text()
            }
    except Exception as e:
        return {
            'url': url,
            'error': str(e)
        }
'''

    # Create training format
    training_examples = [
        {
            "task_decomposition": http_get,
            "implementation": http_get_code,
            "category": "synchronous_http"
        },
        {
            "task_decomposition": socket_server,
            "implementation": socket_server_code, 
            "category": "socket_programming"
        },
        {
            "task_decomposition": async_client,
            "implementation": async_client_code,
            "category": "asynchronous_programming"
        }
    ]
    
    return training_examples

def format_for_training(examples):
    """Convert examples to training text format"""
    
    training_texts = []
    
    for example in examples:
        # Create instruction-following format
        text = f"""
<TASK_DECOMPOSITION>
{example['task_decomposition'].description}

Subtasks:
"""
        for subtask in example['task_decomposition'].subtasks:
            text += f"- {subtask.name} ({subtask.estimate_time}, {subtask.operation.name})\n"
            if subtask.subtasks:
                for sub in subtask.subtasks:
                    text += f"  - {sub.name} ({sub.estimate_time}, {sub.operation.name})\n"
        
        text += f"""
</TASK_DECOMPOSITION>

<IMPLEMENTATION>
{example['implementation']}
</IMPLEMENTATION>

<CATEGORY>{example['category']}</CATEGORY>
"""
        training_texts.append(text)
    
    return training_texts

# Generate comprehensive training dataset
def generate_full_dataset():
    """Generate 1000+ training examples across network operations"""
    
    categories = [
        "http_requests", "socket_programming", "async_networking",
        "api_clients", "websockets", "error_handling", 
        "authentication", "connection_pooling", "rate_limiting"
    ]
    
    # Generate variations for each category
    all_examples = []
    for category in categories:
        # Generate 100+ examples per category with variations
        examples = generate_category_examples(category, count=100)
        all_examples.extend(examples)
    
    return all_examples

def generate_category_examples(category, count=100):
    """Generate multiple examples for a specific category"""
    # Implementation would create variations of network operations
    # with different parameters, error conditions, optimizations
    pass

if __name__ == "__main__":
    # Generate sample training data
    examples = generate_network_operations_training_data()
    training_texts = format_for_training(examples)
    
    # Save to files
    for i, text in enumerate(training_texts):
        with open(f"training_example_{i+1}.txt", "w") as f:
            f.write(text)
    
    print(f"Generated {len(training_texts)} training examples")
    print(f"Total characters: {sum(len(text) for text in training_texts):,}")