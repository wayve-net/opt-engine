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

# Enhanced scraping with Scrapling
try:
    from scrapling.fetchers import Fetcher, StealthyFetcher, PlayWrightFetcher
    SCRAPLING_AVAILABLE = True
except ImportError:
    print("⚠️  Scrapling not installed. Install with: pip install scrapling")
    SCRAPLING_AVAILABLE = False

# Fallback imports
from bs4 import BeautifulSoup

# Set up logging
logging.basicConfig(level=logging.INFO)
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

    def collect_network_code_sources(self) -> Dict:
        """Enhanced version of collect_network_code with more sources"""
        
        network_repos = [
            "requests/requests", "aio-libs/aiohttp", "urllib3/urllib3", "encode/httpx",
            "scrapy/scrapy", "prometheus/prometheus", "nmap/nmap", "scapy/scapy",
            "pallets/flask", "django/django", "tiangolo/fastapi", "tornadoweb/tornado",
        ]
        docs_sources = [
            {"url": "https://docs.python.org/3/library/socket.html", "category": "socket_programming", "selectors": {"content": "div.body", "code_blocks": "pre, code"}},
            {"url": "https://requests.readthedocs.io/en/latest/", "category": "http_requests", "selectors": {"content": ".document", "code_blocks": ".highlight-python"}},
            {"url": "https://aiohttp.readthedocs.io/en/stable/", "category": "async_networking", "selectors": {"content": ".body", "code_blocks": ".highlight-python3"}},
        ]
        tutorial_sources = [
            {"url": "https://realpython.com/python-sockets/", "category": "socket_programming", "selectors": {"content": "article", "code": ".codehilite"}},
            {"url": "https://realpython.com/api-integration-in-python/", "category": "api_clients", "selectors": {"content": "article", "code": ".codehilite"}},
        ]
        
        return {
            "github_repos": network_repos,
            "documentation": docs_sources,
            "tutorials": tutorial_sources
        }
    
    def scrape_with_scrapling(self, url: str, selectors: Dict[str, str] = None, 
                             category: str = "general") -> Optional[Dict]:
        """Enhanced scraping using Scrapling with fallback"""
        
        if not SCRAPLING_AVAILABLE:
            return self._fallback_scrape(url, selectors)
        if url in self.scraped_urls:
            logger.info(f"⏭️  Skipping already scraped URL: {url}")
            return None
            
        try:
            page = StealthyFetcher.fetch(url, stealthy_headers=True, network_idle=True, timeout=30)
            if page.status != 200:
                logger.warning(f"⚠️  HTTP {page.status} for {url}")
                return None
            
            logger.info(f"✅ Successfully fetched: {url}")
            self.scraped_urls.add(url)
            content = self._extract_content_with_selectors(page, selectors or {})
            
            return {
                "url": url, "category": category, "status": page.status,
                "content": content, "title": page.css_first('title::text') or "No title",
                "scraped_at": time.time()
            }
        except Exception as e:
            logger.error(f"❌ Scrapling failed for {url}: {e}")
            return self._fallback_scrape(url, selectors)
    
    def _extract_content_with_selectors(self, page, selectors: Dict[str, str]) -> Dict:
        """Extract content using CSS selectors with Scrapling's features"""
        extracted = {}
        try:
            if "content" in selectors:
                extracted["main_content"] = [elem.get_all_text() for elem in page.css(selectors["content"])]
            if "code_blocks" in selectors:
                extracted["code_blocks"] = [elem.text for elem in page.css(selectors["code_blocks"]) if elem.text.strip()]
            if "examples" in selectors:
                extracted["examples"] = [elem.text for elem in page.css(selectors["examples"]) if elem.text.strip()]
            if not selectors:
                extracted["all_text"] = page.get_all_text(ignore_tags=('script', 'style', 'nav'))
                code_patterns = ['pre', 'code', '.highlight', '.codehilite', '.language-python']
                for pattern in code_patterns:
                    elements = page.css(pattern)
                    if elements:
                        extracted.setdefault("auto_code", []).extend([elem.text for elem in elements if elem.text.strip()])
        except Exception as e:
            logger.error(f"❌ Content extraction failed: {e}")
            extracted["error"] = str(e)
        return extracted
    
    def _fallback_scrape(self, url: str, selectors: Dict[str, str] = None) -> Optional[Dict]:
        """Fallback scraping method using requests + BeautifulSoup"""
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            content = {}
            if selectors:
                for key, selector in selectors.items():
                    elements = soup.select(selector)
                    content[key] = [elem.get_text(strip=True) for elem in elements]
            else:
                content["text"] = soup.get_text(separator=' ', strip=True)
            return {"url": url, "status": response.status_code, "content": content, "title": soup.title.string or "No title", "scraped_at": time.time()}
        except Exception as e:
            logger.error(f"❌ Fallback scraping failed for {url}: {e}")
            return None

    def scrape_github_repository_files(self, repo_name: str, file_extensions: List[str] = None) -> Generator[Dict, None, None]:
        """Enhanced GitHub repository scraping"""
        if file_extensions is None:
            file_extensions = ['.py', '.md']
        api_url = f"https://api.github.com/repos/{repo_name}/contents"
        headers = {"Accept": "application/vnd.github.v3+json"}
        try:
            response = requests.get(api_url, headers=headers, timeout=30)
            response.raise_for_status()
            contents = response.json()
            target_files = [c for c in contents if c['type'] == 'file' and any(c['name'].endswith(ext) for ext in file_extensions)][:10]
            logger.info(f"📁 Found {len(target_files)} target files in {repo_name}")
            for file_info in target_files:
                try:
                    file_response = requests.get(file_info['download_url'], timeout=30)
                    if file_response.status_code == 200:
                        yield {"repo": repo_name, "filename": file_info['name'], "path": file_info['path'], "content": file_response.text, "scraped_at": time.time()}
                        time.sleep(0.5)
                except Exception as e:
                    logger.error(f"❌ Failed to fetch {file_info['name']}: {e}")
        except Exception as e:
            logger.error(f"❌ Failed to access repository {repo_name}: {e}")
    
    def scrape_documentation_sites(self, doc_sources: List[Dict]) -> Generator[Dict, None, None]:
        """Scrape documentation sites with intelligent content extraction"""
        for doc_info in doc_sources:
            logger.info(f"📖 Scraping documentation: {doc_info['url']}")
            result = self.scrape_with_scrapling(doc_info['url'], doc_info.get('selectors', {}), doc_info.get('category', 'documentation'))
            if result and result.get('content'):
                yield result
            time.sleep(2)

    def scrape_tutorial_sites(self, tutorial_sources: List[Dict]) -> Generator[Dict, None, None]:
        """Scrape programming tutorial sites"""
        for tutorial in tutorial_sources:
            logger.info(f"🎓 Scraping tutorial: {tutorial['url']}")
            result = self.scrape_with_scrapling(tutorial['url'], tutorial.get('selectors', {}), tutorial.get('category', 'tutorial'))
            if result and result.get('content'):
                if 'code' in result['content']:
                    result['code_examples'] = result['content']['code']
                yield result
            time.sleep(3)

    def collect_raw_data(self):
        """Collects raw data and saves it to a file."""
        collected_data = {"github_files": [], "documentation": [], "tutorials": []}
        sources = self.collect_network_code_sources()
        
        logger.info("📂 Collecting from GitHub repositories...")
        for repo in sources["github_repos"][:5]:
            for file_data in self.scrape_github_repository_files(repo):
                collected_data["github_files"].append(file_data)
        
        logger.info("📚 Collecting from documentation sites...")
        for doc_data in self.scrape_documentation_sites(sources["documentation"][:3]):
            collected_data["documentation"].append(doc_data)
        
        logger.info("🎯 Collecting from tutorial sites...")
        for tutorial_data in self.scrape_tutorial_sites(sources["tutorials"][:3]):
            collected_data["tutorials"].append(tutorial_data)
        
        with open(self.output_dir / "collected_raw_data.json", 'w', encoding='utf-8') as f:
            json.dump(collected_data, f, indent=2, default=str)
        
        logger.info("✅ Raw data collection completed and saved to 'data/collected_raw_data.json'")
        return collected_data

if __name__ == "__main__":
    collector = DataCollector(use_stealth=True)
    collector.collect_raw_data()