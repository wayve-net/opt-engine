# optimized_tokenizer_training.py
import os
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors, normalizers
from transformers import PreTrainedTokenizerFast

def create_optimized_network_tokenizer(text_files, vocab_size=8192, output_dir="./optimized_network_tokenizer"):
    """Train an optimized tokenizer for network programming with better subword handling"""
    
    # Validate inputs
    for file_path in text_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Training file not found: {file_path}")
    
    print(f"Training optimized tokenizer on {len(text_files)} files...")
    
    # Initialize BPE model
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    
    # Add normalization for better handling of network data
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFD(),  # Unicode normalization
        normalizers.StripAccents(),  # Remove accents
    ])
    
    # Improved pre-tokenization for network/code content
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(pattern=r'\s+', behavior='isolated'),  # Split on whitespace
        pre_tokenizers.Split(pattern=r'[^\w\s<>/\.\-:]+', behavior='isolated'),  # Punctuation
        pre_tokenizers.ByteLevel(add_prefix_space=False)  # ByteLevel last
    ])
    
    # Enhanced special tokens for network programming
    special_tokens = [
        # Core tokens
        "<pad>", "<unk>", "<s>", "</s>", 
        
        # Structural markers
        "<CODE>", "</CODE>", "<LOG>", "</LOG>", 
        "<PACKET>", "</PACKET>", "<ERROR>", "</ERROR>",
        "<CONFIG>", "</CONFIG>", "<TRACE>", "</TRACE>",
        
        # Network lifecycle
        "<CONNECT>", "<DISCONNECT>", "<BIND>", "<LISTEN>", "<ACCEPT>",
        "<HANDSHAKE>", "<SYN>", "<ACK>", "<FIN>", "<RST>",
        "<SEND>", "<RECV>", "<TRANSMIT>", "<RECEIVE>",
        "<REQUEST>", "<RESPONSE>", "<TIMEOUT>", "<RETRY>",
        
        # HTTP methods and status
        "<GET>", "<POST>", "<PUT>", "<DELETE>", "<HEAD>", "<OPTIONS>",
        "<OK>", "<ERROR>", "<REDIRECT>", "<NOT_FOUND>", "<FORBIDDEN>",
        "<SERVER_ERROR>", "<UNAUTHORIZED>", "<BAD_REQUEST>",
        
        # Protocols
        "<HTTP>", "<HTTPS>", "<TCP>", "<UDP>", "<ICMP>", "<DNS>",
        "<TLS>", "<SSL>", "<WEBSOCKET>", "<GRPC>", "<MQTT>", "<FTP>",
        
        # Network components
        "<SERVER>", "<CLIENT>", "<ROUTER>", "<SWITCH>", "<GATEWAY>",
        "<FIREWALL>", "<LOAD_BALANCER>", "<PROXY>",
        
        # Data formats
        "<JSON>", "<XML>", "<YAML>", "<PROTOBUF>", "<BINARY>",
        
        # Common network patterns
        "<IP>", "<PORT>", "<MAC>", "<URL>", "<DOMAIN>", "<HOSTNAME>",
        "<PAYLOAD>", "<HEADER>", "<BODY>", "<PARAMS>", "<QUERY>",
        
        # Log levels
        "<DEBUG>", "<INFO>", "<WARN>", "<ERROR>", "<FATAL>", "<TRACE>",
        
        # Time and identifiers  
        "<TIMESTAMP>", "<SESSION_ID>", "<TRACE_ID>", "<USER_ID>",
        
        # Programming constructs common in networking
        "<FUNCTION>", "<CLASS>", "<METHOD>", "<VARIABLE>", "<CONSTANT>",
        "<IMPORT>", "<EXPORT>", "<ASYNC>", "<AWAIT>", "<CALLBACK>",
        
        # Network operations
        "<POLL>", "<SELECT>", "<EPOLL>", "<KQUEUE>", "<IOCP>",
        "<SOCKET>", "<BUFFER>", "<STREAM>", "<PIPELINE>", "<QUEUE>"
    ]
    
    print(f"Using {len(special_tokens)} special tokens")
    
    # Enhanced trainer configuration
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=2,  # Balanced frequency
        show_progress=True,
        continuing_subword_prefix="",
        end_of_word_suffix="",
        limit_alphabet=1000,  # Limit alphabet size for better merges
    )
    
    # Train the tokenizer
    print("Starting training...")
    tokenizer.train(text_files, trainer)
    print("Training completed!")
    
    # Enhanced post-processor
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save tokenizer
    tokenizer.save(os.path.join(output_dir, "tokenizer.json"))
    
    # Convert to HuggingFace format
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        clean_up_tokenization_spaces=True
    )
    
    fast_tokenizer.save_pretrained(output_dir)
    
    # Statistics
    vocab = tokenizer.get_vocab()
    special_in_vocab = len([t for t in special_tokens if t in vocab])
    
    print(f"\n=== Tokenizer Statistics ===")
    print(f"Final vocabulary size: {len(vocab)}")
    print(f"Special tokens in vocab: {special_in_vocab}/{len(special_tokens)}")
    print(f"Regular tokens: {len(vocab) - special_in_vocab}")
    
    return fast_tokenizer

def analyze_tokenization_quality(tokenizer, test_cases):
    """Analyze how well the tokenizer handles different types of network content"""
    
    print("\n=== Tokenization Quality Analysis ===")
    
    total_tokens = 0
    total_chars = 0
    
    for category, examples in test_cases.items():
        print(f"\n--- {category} ---")
        category_tokens = 0
        category_chars = 0
        
        for example in examples:
            tokens = tokenizer.tokenize(example)
            total_tokens += len(tokens)
            total_chars += len(example)
            category_tokens += len(tokens)
            category_chars += len(example)
            
            print(f"Input: {example}")
            print(f"Tokens ({len(tokens)}): {tokens}")
            print(f"Compression ratio: {len(example)/len(tokens):.2f} chars/token")
            print()
        
        print(f"Category average: {category_chars/len(examples)/category_tokens*len(examples):.2f} chars/token")
    
    print(f"\n=== Overall Statistics ===")
    print(f"Average compression: {total_chars/total_tokens:.2f} chars/token")
    print(f"Total tokens: {total_tokens}, Total chars: {total_chars}")

# Enhanced test cases
test_cases = {
    "Network Operations": [
        "socket.connect(('192.168.1.1', 8080))",
        "HTTP/1.1 200 OK\\r\\nContent-Type: application/json",
        "TCP connection established on port 443",
        "<CONNECT>192.168.1.1:80</CONNECT>"
    ],
    
    "Error Messages": [
        "ConnectionError: [Errno 111] Connection refused",
        "TimeoutError: timed out waiting for response",
        "SSL handshake failed: certificate verification failed"
    ],
    
    "Log Entries": [
        "2024-01-15 10:30:25 INFO [server.py:127] Client connected from 10.0.0.45",
        "ERROR: Failed to bind socket to 0.0.0.0:8080 - Address already in use",
        "<LOG><TIMESTAMP>2024-01-15T10:30:25Z</TIMESTAMP><LEVEL>INFO</LEVEL><MESSAGE>Connection accepted</MESSAGE></LOG>"
    ],
    
    "Code Snippets": [
        "async def send_data(websocket, data): await websocket.send(json.dumps(data))",
        "def create_server(host='localhost', port=8080): return HTTPServer((host, port), RequestHandler)",
        "with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s: s.bind(('', port))"
    ],
    
    "Network Protocols": [
        "GET /api/users?limit=10&offset=20 HTTP/1.1",
        "POST /webhook HTTP/1.1\\r\\nContent-Type: application/json\\r\\n\\r\\n{\"event\": \"user.created\"}",
        "DNS query for example.com returned 93.184.216.34"
    ]
}

if __name__ == "__main__":
    training_files = ["network_logs.txt", "network_code.py"]  # Your files
    
    try:
        # Train optimized tokenizer
        tokenizer = create_optimized_network_tokenizer(
            text_files=training_files,
            vocab_size=8192,
            output_dir="./optimized_network_tokenizer"
        )
        
        # Analyze quality
        analyze_tokenization_quality(tokenizer, test_cases)
        
    except Exception as e:
        print(f"Error: {e}")
