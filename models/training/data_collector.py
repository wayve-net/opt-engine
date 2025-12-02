# data_collector.py
import os
import requests
import json
import csv
import subprocess
import shutil
from pathlib import Path
import time
# Add-ons
from bs4 import BeautifulSoup
from lxml import etree
import kaggle
import zipfile
import io
import pandas as pd
import gzip

def collect_network_code():
    """Collect network-related code and documentation sources"""
    
    # GitHub repositories focused on networking
    network_repos = [
        "requests/requests",
        "aio-libs/aiohttp", 
        "urllib3/urllib3",
        "psf/requests-oauthlib",
        "python-websockets/websockets",
        "encode/httpx",
        "scrapy/scrapy"
        # open-source codebases
        "prometheus/prometheus",
        "wireshark/wireshark",
        "zabbix/zabbix"
    ]
    
    # Stack Overflow network programming questions API endpoint
    stack_exchange_api = {
        "url": "https://api.stackexchange.com/2.3/questions",
        "tags": ["python", "networking", "http", "socket", "rest"]
    }
    
    # Documentation and RFC sources for scraping
    docs_sources = [
        "https://docs.python.org/3/library/socket.html",
        "https://www.w3.org/Protocols/rfc2616/rfc2616.html", 
        "https://aws.amazon.com/what-is/restful-api/",
        "https://developer.mozilla.org/en-US/docs/Web/API/Websockets_API"
    ]

    # IETF RFCs (Request for Comments)
    rfcs = [
        791, # Internet Protocol (IPv4)
        793, # Transmission Control Protocol (TCP)
        2616, # Hypertext Transfer Protocol -- HTTP/1.1
        1918, # Address Allocation for private Internets
        8200, # Internet Protocol, Version 6 (IPv6)
    ]

    # Dataset slugs from Kaggle
    kaggle_datasets = {
        "network_traffic": "jsphyg/network-traffic-dataset-in-csv",
        "intrusion_detection": "datasets/ravikumargattu/network-traffic-dataset",
        "iot_traffic_data": "faisalhadi/iot-device-network-logs",
        "network_behavior_analysis": "crawford/computer-network-traffic",
        "protocol_specific_data": "ahmetts/ping-data",
        "labeled_traffic": "rojas/labeled-network-traffic-flows-114-applications",
    }

    # Stanford Large Network Dataset Collection (SNAP)
    snap_datasets = {
        "stanford_web_graph": "https://snap.stanford.edu/data/web-Stanford.txt.gz"
    }

    # New sources for network and cybersecurity datasets
    web_datasets = {
        # These are top-level pages. You'll need to find specific file URLs.
        "unb_datasets": "https://www.cs.unb.ca/~alashkar/Data-sets.asp",
        "lanl_datasets": "https://csr.lanl.gov/data/",
        "internet_traffic_archive": "http://ita.ee.lbl.gov/html/traces.html"
    }

    # New sources for open-source projects and code repositories
    code_repositories = [
        "https://www.talosintelligence.com/software",
        "https://packetpushers.net/blog/open-source-networking-projects/",
        "https://gist.github.com/stefanbschneider/96602bb3c8b256b90058d59f337a0e59"
    ]

    # New sources for public APIs
    api_explorers = [
        "https://www.postman.com/explore",
        "https://developers.google.com/apis-explorer"
    ]
    
    return network_repos, stack_exchange_api, docs_sources, rfcs, kaggle_datasets, snap_datasets, web_datasets, code_repositories, api_explorers 

def get_github_code(repo_name):
    """Fetches Pyhton code from a Github repository."""
    print(f"Fetching code from Github repo: {repo_name}")
    api_url = f"https://api.github.com/repos/{repo_name}/contents"
    headers = {"Accept": "application/vnd.github.v3.raw"}

    try:
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        contents = response.json()

        # Example: fetch the first 5 Python files
        python_files = [c for c in contents if c['name'].endswith('.py')][:5]

        for file in python_files:
            file_response = requests.get(file['url'], headers=headers)
            if file_response.status_code == 200:
                print(f" - Downloaded {file['name']}")
                yield file_response.text

    except requests.exceptions.RequestException as e:
        print(f"Error fetching Github data: {e}")

def get_stack_overflow_data(api_info):
    """Fetches questions and answers from the Stack Exchange API."""
    print(f"Fetching data from Stack Exchange...")
    params = {
        "site": "stackoverflow",
        "tagged": ";".join(api_info["tags"]),
        "sort": "activity",
        "filter": "withbody"
    }

    try:
        response = requests.get(api_info["url"], params=params)
        response.raise_for_status()
        data = response.json()
        
        for item in data.get("items", []):
            yield {
                "question": item["title"],
                "body": item["body"]
            }
            if "answers" in item:
                for answer in item["answers"]:
                    yield {
                        "question": item["title"],
                        "answer_body": answer["body"]
                    }
                    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching Stack Exchange data: {e}")

def scrape_documentation(url):
    """Scrapes a single web page for text content."""
    print(f"Scraping documentation from: {url}")
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Use BeautifulSoup to parse the HTML and extract text
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract the main content (e.g., from a specific container)
        # This part requires inspecting the specific page's HTML structure
        # Example using a common pattern for main content
        main_content = soup.find('main') or soup.find('article') or soup.find('body')
        
        if main_content:
            return main_content.get_text(separator=' ', strip=True)
            
    except requests.exceptions.RequestException as e:
        print(f"Error scraping documentation: {e}")
    return ""

def get_rfcs(rfc_numbers):
    """Fetches the plain text content of IETF RFC documents."""
    print("Fetching IETF RFC documents...")
    for rfc_number in rfc_numbers:
        url = f"https://datatracker.ietf.org/doc/rfc{rfc_number}/"
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            # The RFCs page has a pre-formatted text block
            soup = BeautifulSoup(response.text, 'html.parser')
            content = soup.find('pre', class_='rfc-text')
            
            if content:
                print(f"  - Downloaded RFC {rfc_number}")
                yield content.get_text()

        except requests.exceptions.RequestException as e:
            print(f"Error fetching RFC {rfc_number}: {e}")

def download_kaggle_dataset(dataset_slug, download_path):
    """
    Downloads and unzips a Kaggle dataset using the kaggle CLI.
    NOTE: Requires the 'kaggle' command-line tool to be installed and configured.
    """
    print(f"Downloading Kaggle dataset: {dataset_slug} to {download_path}")
    try:
        # The `kaggle` command will automatically download and unzip to the CWD
        subprocess.run(
            ["kaggle", "datasets", "download", dataset_slug, "--unzip", "-p", str(download_path)],
            check=True,
            capture_output=True,
            text=True
        )
        print("  - Download successful.")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading Kaggle dataset: {e.stderr}")
        print("Please ensure you have configured your Kaggle API credentials.")
        return False
    return True

def download_snap_dataset(url, category, categories):
    """
    Downloads and processes a gzipped network dataset from the SNAP collection.
    """
    print(f"Downloading SNAP dataset from {url}")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Decompress the gzipped content in memory
        with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as gz_file:
            # Read a small portion of the file to save disk space
            # and to demonstrate the file format
            content = gz_file.read(1024 * 10).decode('utf-8')
            
            if content:
                print("  - SNAP dataset downloaded and processed.")
                categories[category].append(content)
                return True
                
    except requests.exceptions.RequestException as e:
        print(f"Error downloading SNAP dataset: {e}")
        return False

def download_web_dataset(url, download_path):
    """
    Downloads a file from a given URL to a specified path.
    """
    print(f"Downloading file from {url} to {download_path}")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(download_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("  - Download successful.")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading from {url}: {e}")
        return False

def create_training_dataset():
    """Create structured training dataset"""
    
    # Directory structure
    raw_data_dir = Path("data/raw")
    os.makedirs(raw_data_dir, exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    network_repos, stack_exchange_api, docs_sources, rfcs, kaggle_datasets, snap_datasets, web_datasets, code_repositories, api_explorers = collect_network_code()

    # Collect and organize data by categories:
    categories = {
        "socket_programming": [],
        "http_requests": [],
        "async_networking": [],
        "api_clients": [],
        "network_protocols": [],
        "error_handling": [],
        "security": [],
        "traffic_analysis": [],
        "network_graphs": [],
        "cybersecutity_datasets": []
    }

    # 1. Collect data from GitHub repositories
    for repo in network_repos:
        for code in get_github_code(repo):
            # A more sophisticated model would classify this code, for simplicity we'll add it to all relevant categories
            if "socket" in code: categories["socket_programming"].append(code)
            if "http" in code or "requests" in code: categories["http_requests"].append(code)
            if "aio" in code: categories["async_networking"].append(code)
    
    # 2. Collect data from Stack Exchange API
    for item in get_stack_overflow_data(stack_exchange_api):
        if "answer_body" in item:
            # Questions and answers are a great source for problem-solution pairs
            categories["error_handling"].append(item["body"])
            categories["api_clients"].append(item["body"])
    
    # 3. Collect data from documentation sources
    for doc_url in docs_sources:
        text = scrape_documentation(doc_url)
        if text:
            # Organize text based on keywords from the URL or content
            if "socket" in doc_url: categories["socket_programming"].append(text)
            if "http" in doc_url: categories["network_protocols"].append(text)
            if "api" in doc_url: categories["api_clients"].append(text)
            if "websocket" in doc_url: categories["network_protocols"].append(text)

    # 4. Collect data from IETF RFCs
    for rfc_text in get_rfcs(rfcs):
        categories["network_protocols"].append(rfc_text)

    # 5. Collect data from Kaggle datasets
    for category, dataset_slug in kaggle_datasets.items():
        # A separate directory for each dataset helps with cleanup
        dataset_dir = raw_data_dir / dataset_slug.replace("/", "_")
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Download and unzip the dataset using the kaggle CLI
        if download_kaggle_dataset(dataset_slug, dataset_dir):
            # After download, find and process the CSV files
            for file in dataset_dir.glob("*.csv"):
                print(f"Reading and processing {file.name}...")
                with open(file, "r", encoding="utf-8") as f:
                    # Read the first few lines as a string for training
                    csv_data = "".join(f.readlines()[:50])
                    categories[category].append(csv_data)

    # 6. Collect data from SNAP datasets
    for category, url in snap_datasets.items():
        download_snap_dataset(url, "network_graphs", categories)
            
    # 7. Collect data from web datasets
    # This section demonstrates how to handle specific web dataset sources
    for dataset_name, base_url in web_datasets.items():
        if "unb_datasets" in dataset_name:
            print(f"Manually inspecting {base_url} to find dataset links.")
            # Example logic for UNB datasets, which requires web scraping to find file links.
            # This is a conceptual example, as the exact URLs would change.
            try:
                response = requests.get(base_url)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                # Find all links that point to files (e.g., .zip, .pcap)
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    if href.endswith(('.zip', '.pcap', '.tar.gz')):
                        file_url = href if href.startswith('http') else f"https://www.cs.unb.ca/~alashkar/{href}"
                        file_name = os.path.basename(file_url)
                        download_path = raw_data_dir / "unb" / file_name
                        os.makedirs(download_path.parent, exist_ok=True)
                        download_web_dataset(file_url, download_path)
            except requests.exceptions.RequestException as e:
                print(f"Could not scrape UNB dataset page: {e}")
        
        elif "lanl_datasets" in dataset_name:
            # Example: Hardcoding a known direct download URL if available
            lanl_dataset_url = "https://csr.lanl.gov/data/2017-network-flows.zip"
            download_path = raw_data_dir / "lanl" / "2017-network-flows.zip"
            os.makedirs(download_path.parent, exist_ok=True)
            download_web_dataset(lanl_dataset_url, download_path)

    # Save raw data to files
    for category, data_list in categories.items():
        if data_list:
            file_path = raw_data_dir / f"{category}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data_list, f, indent=4)
                print(f"Saved {len(data_list)} items to {file_path}")

    return categories

if __name__ == "__main__":
    create_training_dataset()