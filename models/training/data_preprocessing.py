# data_preprocessing.py
# this is where we add all the training examples
import re
import ast
import os
import json
import csv

def preprocess_network_code(code_text):
    """Clean and structure network programming code"""
    
    # Remove excessive whitespace but preserve code structure
    code_text = re.sub(r'\n\s*\n\s*\n', '\n\n', code_text)
    
    # Add special tokens for different types of network operations
    patterns = {
        r'(requests\.|urllib\.|http\.client)': '<HTTP>',
        r'(socket\.|asyncio\.)': '<SOCKET>',
        r'(try:|except:|raise)': '<ERROR>',
        r'(def |class |import )': '<CODE>'
    }
    
    for pattern, token in patterns.items():
        code_text = re.sub(pattern, f'{token}\\1', code_text)
    
    return code_text

def create_instruction_dataset():
    """Create instruction-following dataset for network tasks"""
    
    examples = [
        {
            "instruction": "Write Python code to make an HTTP GET request",
            "input": "URL: https://api.example.com/data",
            "output": "import requests\nresponse = requests.get('https://api.example.com/data')\ndata = response.json()"
        },
        {
            "instruction": "Create a simple TCP socket server",
            "input": "Port: 8080",
            "output": "import socket\ns = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\ns.bind(('localhost', 8080))\ns.listen(5)"
        }
        # Add hundreds more examples
    ]
    
    return examples

def save_dataset_to_folder(dataset, base_folder, filename="dataset.json"):
    """Saves the dataset to a specified folder in JSON format."""
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
        
    file_path = os.path.join(base_folder, filename)
    
    with open(file_path, "w") as f:
        json.dump(dataset, f, indent=4)
        
    print(f"Dataset saved to '{file_path}'")

def preprocess_multiple_files(input_folder, output_folder):
    """
    Reads all supported files from an input folder, preprocesses their code content,
    and saves the preprocessed content as individual files in an output folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        file_extension = os.path.splitext(filename)[1].lower()
        
        raw_code = None
        
        # Read file content based on extension
        if file_extension == '.json':
            try:
                with open(input_path, 'r', encoding='utf-8') as infile:
                    data = json.load(infile)
                    # Assuming the code is in a key named 'output' or similar
                    code_texts = [item.get('output', '') for item in data]
                    raw_code = '\n\n'.join(code_texts)
            except Exception as e:
                print(f"Error reading JSON file '{filename}': {e}")
                continue
        
        elif file_extension == '.csv':
            try:
                with open(input_path, 'r', newline='', encoding='utf-8') as infile:
                    reader = csv.reader(infile)
                    # Assuming the code is in a specific column, e.g., the second column (index 1)
                    # You might need to adjust this depending on your CSV structure
                    code_texts = [row[1] for row in reader if len(row) > 1]
                    raw_code = '\n\n'.join(code_texts)
            except Exception as e:
                print(f"Error reading CSV file '{filename}': {e}")
                continue
                
        elif file_extension in ['.txt', '.py']:
            try:
                with open(input_path, 'r', encoding='utf-8') as infile:
                    raw_code = infile.read()
            except Exception as e:
                print(f"Error reading text/py file '{filename}': {e}")
                continue
        
        else:
            print(f"Skipping unsupported file type: '{filename}'")
            continue

        if raw_code:
            preprocessed_code = preprocess_network_code(raw_code)
            output_path = os.path.join(output_folder, f"preprocessed_{os.path.basename(filename).split('.')[0]}.txt")
            
            try:
                with open(output_path, 'w', encoding='utf-8') as outfile:
                    outfile.write(preprocessed_code)
                print(f"Successfully preprocessed '{filename}' and saved to '{output_path}'")
            except Exception as e:
                print(f"Failed to write preprocessed file for '{filename}': {e}")

if __name__ == "__main__":

    # Example usage of the preprocessor
    sample_code = """
import socket

def my_socket_function():

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect(('localhost', 8080))
    except Exception as e:
        print(f"Error: {e}")
    """
    
    preprocessed_code = preprocess_network_code(sample_code)
    print("--- Preprocessed Code ---")
    print(preprocessed_code)

    # Example usage of the dataset creator and saver
    dataset = create_instruction_dataset()
    print("\n--- Instruction Dataset ---")
    print(json.dumps(dataset, indent=2))
    
    # Save the dataset to a new folder named 'output'
    save_dataset_to_folder(dataset, "output")
