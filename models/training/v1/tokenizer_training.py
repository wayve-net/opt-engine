# tokenizer_training.py
import os
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors
from transformers import PreTrainedTokenizerFast

def train_network_tokenizer(text_files, vocab_size=8192, output_dir="./network_tokenizer"):
    """Train tokenizer optimized for network programming"""
    
    # Validate input files exist and are not empty
    if not text_files:
        raise ValueError("No training files provided")
    
    for file_path in text_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Training file not found: {file_path}")
        if os.path.getsize(file_path) == 0:
            raise ValueError(f"Training file is empty: {file_path}")
    
    print(f"Training tokenizer on {len(text_files)} files...")
    
    # Use BPE with network-specific vocabulary
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    
    # Pre-tokenization optimized for code and network data
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    
    # Define categorized special tokens based on network task lifecycle
    structural_tokens = [
        "<pad>", "<unk>", "<s>", "</s>", "<CODE>", "</CODE>",
        "<LOG>", "</LOG>", "<PACKET>", "</PACKET>", "<ERROR>", "</ERROR>"
    ]
 
    lifecycle_tokens = [
        "<CONNECT>", "<BIND>", "<LISTEN>", "<ACCEPT>", "<HANDSHAKE>", 
        "<NEGOTIATE>", "<SYN>", "<ACK>", "<FIN>",
        "<SEND>", "<RECV>", "<TRANSMIT>", "<RECEIVE>", 
        "<REQUEST>", "<RESPONSE>", "<GET>", "<POST>", "<PUT>", "<DELETE>", 
        "<POLL>", "<PING>",
        "<CLOSE>", "<TIMEOUT>", "<SUCCESS>", "<FAILURE>", "<PENDING>", 
        "<RETRY>", "<DEADLINE_EXCEEDED>", "<OK>", "<CONNECTION_REFUSED>", 
        "<PROTOCOL_ERROR>", "<UNAUTHORIZED>", "<FORBIDDEN>", "<NOT_FOUND>", 
        "<SERVER_ERROR>"
    ]
    
    protocol_tokens = [
        "<HTTP/1.1>", "<HTTP/2>", "<TCP>", "<UDP>", "<ICMP>", "<ARP>", 
        "<QUIC>", "<WEBSOCKET>", "<gRPC>", "<MQTT>", "<COAP>", "<FTP>", 
        "<SSH>", "<BGP>", "<OSPF>", "<DNS>", "<DHCP>", "<TLS>", "<SSL>", 
        "<IPsec>", "<RTP>", "<SIP>",
        "<JSON>", "<XML>", "<YAML>", "<PROTOBUF>", "<AVRO>", "<CBOR>", 
        "<GZIP>", "<ZLIB>"
    ]
    
    resource_tokens = [
        "<SERVER>", "<CLIENT>", "<PORT>", "<ADDRESS>", "<IP>", "<MAC>", 
        "<HOSTNAME>", "<ROUTER>", "<SWITCH>", "<GATEWAY>", "<FIREWALL>", 
        "<LOAD_BALANCER>", "<VPN>", "<DNS_SERVER>", "<NAT_DEVICE>",
        "<PAYLOAD>", "<HEADER>"
    ]

    log_packet_tokens = [
        "<PACKET_HEADER>", "<PACKET_BODY>", "<SOURCE_IP>", "<DESTINATION_IP>", 
        "<SOURCE_PORT>", "<DESTINATION_PORT>", "<PROTOCOL_TYPE>", 
        "<PACKET_SIZE>", "<PAYLOAD_LENGTH>",
        "<TIMESTAMP>", "<LEVEL>", "<MESSAGE>", "<TRACE_ID>", "<SESSION_ID>",
        "<EVENT_ID>", "<SYSLOG>", "<JSON_LOG>"
    ]

    # Combine all lists into a single list for the trainer
    special_tokens = (
        structural_tokens + 
        lifecycle_tokens + 
        protocol_tokens + 
        resource_tokens +
        log_packet_tokens
    )
    
    print(f"Using {len(special_tokens)} special tokens")
    
    # Configure trainer with more aggressive settings
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=1,  # Lowered from 2 to capture more tokens
        show_progress=True,
        continuing_subword_prefix="",
        end_of_word_suffix=""
    )
    
    # Train the tokenizer
    print("Starting training...")
    tokenizer.train(text_files, trainer)
    print("Training completed!")
    
    # Add post-processor for proper token handling
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the raw tokenizer first
    tokenizer.save(os.path.join(output_dir, "tokenizer.json"))
    print(f"Raw tokenizer saved to {output_dir}/tokenizer.json")
    
    # Convert to HuggingFace format with proper configuration
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>"
    )
    
    # Save HuggingFace tokenizer
    fast_tokenizer.save_pretrained(output_dir)
    print(f"HuggingFace tokenizer saved to {output_dir}")
    
    # Print some statistics
    vocab = tokenizer.get_vocab()
    print(f"Final vocabulary size: {len(vocab)}")
    print(f"Special tokens in vocab: {len([t for t in special_tokens if t in vocab])}")
    
    return fast_tokenizer

def test_tokenizer(tokenizer, test_strings):
    """Test the trained tokenizer on sample network-related strings"""
    print("\n=== Tokenizer Testing ===")
    
    for test_str in test_strings:
        tokens = tokenizer.tokenize(test_str)
        ids = tokenizer.encode(test_str)
        decoded = tokenizer.decode(ids)
        
        print(f"\nInput: {test_str}")
        print(f"Tokens: {tokens}")
        print(f"Token IDs: {ids}")
        print(f"Decoded: {decoded}")
        print(f"Token count: {len(tokens)}")

# Example usage and testing
if __name__ == "__main__":
    # Training files
    training_files = [
        "scrapfile.txt",
        "synfile.txt",
        # Add more files as needed
    ]
    
    # Test strings to verify tokenizer behavior
    test_strings = [
        "TCP connection established on port 8080",
        "<CONNECT>192.168.1.1:80</CONNECT>",
        "HTTP/1.1 200 OK",
        "def send_packet(data): return socket.send(data)",
        "ERROR: Connection refused on 127.0.0.1:3000"
    ]
    
    try:
        # Train the tokenizer
        tokenizer = train_network_tokenizer(
            text_files=training_files,
            vocab_size=8192,
            output_dir="./network_tokenizer"
        )
        
        # Test the tokenizer
        test_tokenizer(tokenizer, test_strings)
        
        print("\n=== Tokenizer training completed successfully! ===")
        
    except Exception as e:
        print(f"Error during tokenizer training: {e}")
        print("Please check:")
        print("1. Training files exist and contain data")
        print("2. You have write permissions for the output directory")
        print("3. Required libraries are installed (tokenizers, transformers)")