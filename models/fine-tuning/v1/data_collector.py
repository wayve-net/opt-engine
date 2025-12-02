# data_collector.py
"""
Data collector using Scrapling for robust web scraping.
This file is responsible for fetching raw data from various sources.
"""

import os
import requests
import json
from pathlib import Path
import time
from typing import List, Dict, Optional, Generator
from urllib.parse import urljoin, urlparse
import logging
import re
from bs4 import BeautifulSoup
import subprocess
from concurrent.futures import ThreadPoolExecutor

# Enhanced scraping with Scrapling
try:
    from scrapling.fetchers import Fetcher, StealthyFetcher, PlayWrightFetcher
    SCRAPLING_AVAILABLE = True
except ImportError:
    print("⚠️  Scrapling not installed. Install with: pip install scrapling")
    SCRAPLING_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataCollector:
    """
    Data collector with Scrapling integration for robust scraping.
    """
    
    def __init__(self, use_stealth: bool = True):
        self.use_stealth = use_stealth
        self.scraped_urls = set()  # Track scraped URLs to avoid duplicates
        self.output_dir = Path("data") # Store findings in a 'data' folder
        
        if SCRAPLING_AVAILABLE:
            StealthyFetcher.auto_match = True
            logger.info("✅ Scrapling initialized with auto-matching enabled")
        else:
            logger.warning("⚠️  Using fallback scraping methods")
            
        self.output_dir.mkdir(exist_ok=True)
        logger.info(f"📁 Created output directory: {self.output_dir}")

        # Initialize separate files for structured output
        self.github_output_file = self.output_dir / "github_raw.json"
        self.docs_output_file = self.output_dir / "docs_raw.json"
        self.tutorials_output_file = self.output_dir / "tutorials_raw.json"
        self.traces_output_file = self.output_dir / "traces_raw.json"
        
        # Clear existing files for a clean run
        for f in [self.github_output_file, self.docs_output_file, self.tutorials_output_file, self.traces_output_file]:
            if f.exists():
                os.remove(f)

        self.networking_keywords = re.compile(r'socket|ssl|http|dns|asyncio|requests|aiohttp|urllib|curl|bind|connect|listen|accept|send|recv|tls|quic|tcp|udp|ip|network|netlink|epoll|kqueue|select', re.IGNORECASE)
        self.networking_imports = re.compile(r'import\s+(socket|ssl|asyncio|http\.client|urllib|requests|aiohttp|curl)|from\s+(socket|ssl|asyncio|http\.client|urllib|requests|aiohttp|curl)', re.IGNORECASE)
        self.file_extensions = ('.py', '.c', '.h', '.go', '.rs')

    def collect_network_code_sources(self) -> Dict[str, List[str]]:
        """A curated list of network programming sources."""
        return {
            "github_repos": [
                "https://github.com/kennethreitz/requests",
                "https://github.com/aio-libs/aiohttp",
                "https://github.com/scrapy/scrapy",
                "https://github.com/scapy/scapy",
                "https://github.com/ktbyers/netmiko",
                "https://github.com/ansible/ansible",
                "https://github.com/google/go-libpcap",
                "https://github.com/wg/wrk",
                "https://github.com/systemd/systemd", # For netlink/system level networking
                "https://github.com/DNS-OARC/dnsmasq",
                "https://github.com/nginx/nginx",
                "https://github.com/OpenVPN/openvpn",
                "https://github.com/curl/curl",
                "https://github.com/torvalds/linux" # For low-level kernel networking
            ],
            "documentation": [
                "https://www.rfc-editor.org/",
                "https://datatracker.ietf.org/",
                "https://www.juniper.net/documentation/",
                "https://man7.org/linux/man-pages/man2/socket.2.html",
                "https://man7.org/linux/man-pages/man2/connect.2.html",
                "https://man7.org/linux/man-pages/man7/epoll.7.html",
                "https://www.wireshark.org/docs/wsug_html_chunked/ChBuildProtocols.html",
            ],
            "tutorials": [
                "https://realpython.com/async-io-in-python/",
                "https://www.fullstackpython.com/network-programming.html",
                "https://pymotw.com/3/", # Python Module of the Week
                "https://www.geeksforgeeks.org/python-networking/",
                "https://docs.python.org/3/howto/sockets.html"
            ]
        }

    def _extract_content_with_selectors(self, soup, tags: list, is_tutorial: bool = False) -> Dict:
        """Extract content and optional natural language descriptions based on tags."""
        content = {'text': '', 'code': [], 'natural_language_description': []}
        for tag in tags:
            for element in soup.select(tag):
                text_content = element.get_text(strip=True)
                if element.name in ['pre', 'code'] and text_content:
                    content['code'].append(text_content)
                else:
                    content['text'] += text_content + '\\n'
                    # Capture problem descriptions from tutorials
                    if is_tutorial and self.networking_keywords.search(text_content):
                        # Heuristic: capture the sentence or paragraph before a code block
                        prev_p = element.find_previous(['p', 'h1', 'h2', 'h3'])
                        if prev_p and prev_p.get_text(strip=True):
                            content['natural_language_description'].append(prev_p.get_text(strip=True))

        return content

    def scrape_github_repository_files(self, repo_url: str) -> Generator[Dict, None, None]:
        """Scrapes relevant network-related code files from a GitHub repository."""
        try:
            logger.info(f"🔍 Scraping GitHub repo: {repo_url}")
            owner, repo_name = repo_url.split('/')[-2:]
            api_url = f"https://api.github.com/repos/{owner}/{repo_name}/git/trees/main?recursive=1"
            
            headers = {"Accept": "application/vnd.github.v3.json"}
            response = requests.get(api_url, headers=headers)
            response.raise_for_status()
            tree = response.json().get('tree', [])
            
            for item in tree:
                path = item['path']
                if any(path.endswith(ext) for ext in self.file_extensions) and item['type'] == 'blob':
                    raw_url = f"https://raw.githubusercontent.com/{owner}/{repo_name}/main/{path}"
                    file_content_response = requests.get(raw_url)
                    if file_content_response.status_code == 200:
                        content = file_content_response.text
                        if self.networking_keywords.search(content) or self.networking_imports.search(content):
                            logger.info(f"✅ Found relevant file: {path}")
                            file_meta = {
                                "source": "github",
                                "repo": repo_url,
                                "file_path": path,
                                "content": content,
                                "imports": self.networking_imports.findall(content),
                                "category": "general_networking", # Placeholder, can be refined with more advanced logic
                                "function_signatures": re.findall(r'def\s+(\w+)\s*\(|func\s+(\w+)\s*\(', content)
                            }
                            # Optional: Instrumentation hooks for execution traces (not executable in this environment)
                            # You would run a separate process here, e.g., subprocess.run(['strace', 'python', path])
                            # And capture the output to 'traces_raw.json'
                            file_meta['execution_traces'] = "Instrumentation hooks (strace/eBPF) would be integrated here."
                            yield file_meta
                    time.sleep(1) # Be a good citizen, don't spam the API
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Failed to scrape GitHub repo {repo_url}: {e}")

    def scrape_documentation_sites(self, urls: List[str]) -> Generator[Dict, None, None]:
        """Scrapes official documentation sites for network-related content."""
        logger.info("📚 Collecting from documentation sites...")
        for url in urls:
            if url in self.scraped_urls:
                continue
            self.scraped_urls.add(url)
            try:
                fetcher = StealthyFetcher() if self.use_stealth and SCRAPLING_AVAILABLE else Fetcher()
                logger.info(f"Fetching {url}")
                page_content = fetcher.get(url, cache=True).content
                soup = BeautifulSoup(page_content, 'html.parser')
                
                content = self._extract_content_with_selectors(soup, ['p', 'pre', 'code'])
                
                yield {
                    "source": "documentation",
                    "url": url,
                    "content": content,
                    "tags": ["reference", "hints"]
                }
            except Exception as e:
                logger.error(f"❌ Failed to scrape documentation site {url}: {e}")
            time.sleep(3)

    def scrape_tutorial_sites(self, urls: List[str]) -> Generator[Dict, None, None]:
        """Scrapes tutorial sites, capturing both code and natural language explanations."""
        logger.info("🎯 Collecting from tutorial sites...")
        for url in urls:
            if url in self.scraped_urls:
                continue
            self.scraped_urls.add(url)
            try:
                fetcher = StealthyFetcher() if self.use_stealth and SCRAPLING_AVAILABLE else Fetcher()
                logger.info(f"Fetching {url}")
                page_content = fetcher.get(url, cache=True).content
                soup = BeautifulSoup(page_content, 'html.parser')
                
                content = self._extract_content_with_selectors(soup, ['p', 'pre', 'code', 'h1', 'h2', 'h3'], is_tutorial=True)
                
                yield {
                    "source": "tutorial",
                    "url": url,
                    "content": content,
                    "natural_language_description": content.get('natural_language_description', [])
                }
            except Exception as e:
                logger.error(f"❌ Failed to scrape tutorial site {url}: {e}")
            time.sleep(3)
    
    def scrape_rfcs(self) -> Generator[Dict, None, None]:
        """A simple scraping method for RFCs from rfc-editor.org."""
        logger.info("📄 Collecting RFCs...")
        try:
            rfc_search_url = "https://www.rfc-editor.org/rfc/rfc793.txt" # Example for TCP RFC
            response = requests.get(rfc_search_url)
            response.raise_for_status()
            
            yield {
                "source": "rfc",
                "url": rfc_search_url,
                "content": response.text,
                "tags": ["protocol_specification"]
            }
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Failed to scrape RFCs: {e}")

    def collect_raw_data(self):
        """Collects raw data and saves it to separate structured files."""
        collected_data = {
            "github_files": [],
            "documentation": [],
            "tutorials": [],
            "rfcs": []
        }
        sources = self.collect_network_code_sources()

        # Use a ThreadPoolExecutor for concurrent scraping
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            logger.info("📂 Collecting from GitHub repositories...")
            for repo in sources["github_repos"]:
                futures.append(executor.submit(self._run_github_scrape, repo))
            
            logger.info("📚 Collecting from documentation sites...")
            futures.append(executor.submit(self._run_docs_scrape, sources["documentation"]))

            logger.info("🎯 Collecting from tutorial sites...")
            futures.append(executor.submit(self._run_tutorials_scrape, sources["tutorials"]))
            
            logger.info("📄 Collecting RFCs...")
            futures.append(executor.submit(self._run_rfcs_scrape))

            for future in futures:
                result = future.result()
                if result:
                    if result['type'] == 'github':
                        collected_data["github_files"].extend(result['data'])
                    elif result['type'] == 'docs':
                        collected_data["documentation"].extend(result['data'])
                    elif result['type'] == 'tutorials':
                        collected_data["tutorials"].extend(result['data'])
                    elif result['type'] == 'rfcs':
                        collected_data["rfcs"].extend(result['data'])

        self._save_to_json(self.github_output_file, collected_data["github_files"])
        self._save_to_json(self.docs_output_file, collected_data["documentation"] + collected_data["rfcs"])
        self._save_to_json(self.tutorials_output_file, collected_data["tutorials"])

        logger.info("✅ Raw data collection completed and saved to separate files.")
        return collected_data
    
    def _run_github_scrape(self, repo):
        try:
            return {"type": "github", "data": list(self.scrape_github_repository_files(repo))}
        except Exception as e:
            logger.error(f"Error scraping GitHub repo {repo}: {e}")
            return None

    def _run_docs_scrape(self, urls):
        try:
            return {"type": "docs", "data": list(self.scrape_documentation_sites(urls))}
        except Exception as e:
            logger.error(f"Error scraping docs: {e}")
            return None

    def _run_tutorials_scrape(self, urls):
        try:
            return {"type": "tutorials", "data": list(self.scrape_tutorial_sites(urls))}
        except Exception as e:
            logger.error(f"Error scraping tutorials: {e}")
            return None

    def _run_rfcs_scrape(self):
        try:
            return {"type": "rfcs", "data": list(self.scrape_rfcs())}
        except Exception as e:
            logger.error(f"Error scraping RFCs: {e}")
            return None

    def _save_to_json(self, file_path, data):
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)

if __name__ == "__main__":
    collector = DataCollector(use_stealth=True)
    collector.collect_raw_data()