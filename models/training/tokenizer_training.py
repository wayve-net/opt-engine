# tokenizer_training.py
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

def train_network_tokenizer(text_files, vocab_size=8192):
    """Train tokenizer optimized for network programming"""
    
    # Use BPE with network-specific vocabulary
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    
    # Pre-tokenization optimized for code
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
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
    
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=2
    )
    
    tokenizer.train(text_files, trainer)
    
    # Convert to HuggingFace format
    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    fast_tokenizer.save_pretrained("./network_tokenizer")
    
    return fast_tokenizer
