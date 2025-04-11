"""
Truth Scope Article Collector

This module collects and analyzes web articles by searching for relevant content
based on news headlines and extracting keywords from article bodies.
It uses Google Custom Search API for finding related articles and a local scraper
service to extract content from web pages.

Features:
- Google Custom Search API integration for finding related articles
- Local scraper service integration for content extraction
- Keyword extraction from headlines and article bodies
- Error handling and logging
- Configurable parameters for searches and keyword extraction
"""

# Standard library imports
import os              # For environment variable access
import json            # For JSON parsing and serialization
import logging         # For logging messages and errors (suppressed in this version)
import concurrent.futures  # For parallel processing of URLs
import re              # For URL validation using regular expressions

# Third-party imports
import requests        # For making HTTP requests to APIs and web services
from dotenv import load_dotenv  # For loading environment variables from .env file
from requests.adapters import HTTPAdapter  # For adding retry capabilities to requests
from urllib3.util.retry import Retry  # For configuring retry behavior
from extractor import extract_keywords_yake  # Local module for keyword extraction
from scorer import calculate_similarity_scores  # Local module for similarity scoring

# Configure logging with null handler to suppress output
logging.basicConfig(  
    level=logging.CRITICAL,  # Only log critical errors (effectively suppressing most logs)
    handlers=[logging.NullHandler()]  # Direct logs to nowhere
)
logger = logging.getLogger(__name__)  # Get logger for this module

# Load environment variables from .env file for API credentials
load_dotenv()  # Load variables from .env file into environment

# Configuration constants for the module
SEARCH_ENGINE_ID = os.getenv('SEARCH_ENGINE_ID')  # Google Custom Search Engine ID
API_KEY = os.getenv('API_KEY')  # Google API Key for Custom Search API
MAX_KEYWORDS_HEADLINE = 4  # Maximum number of keywords to extract from headlines
MAX_KEYWORDS_ARTICLE = 5  # Maximum number of keywords to extract from article bodies
SCRAPER_URL = "http://127.0.0.1:5000/scrape"  # URL for local scraper service
REQUEST_TIMEOUT = 10  # Timeout for HTTP requests in seconds
MAX_RETRIES = 3  # Maximum number of retries for HTTP requests
TEST_HEADLINE = "Delhi weather sees sudden turn: Rain, dust storms bring temperatures down in capital"  # Default test headline

# Validate API credentials before proceeding
if not SEARCH_ENGINE_ID or not API_KEY:
    # API credentials are required for this module to function
    raise ValueError("Missing API credentials. Check your .env file for SEARCH_ENGINE_ID and API_KEY.")

def create_retry_session(retries=MAX_RETRIES, backoff_factor=0.3, status_forcelist=(500, 502, 503, 504)):
    """
    Create a requests session with retry capability for more robust HTTP requests.
    
    Args:
        retries (int): Number of times to retry failed requests
        backoff_factor (float): Backoff factor between retries (exponential backoff)
        status_forcelist (tuple): HTTP status codes that should trigger a retry
    
    Returns:
        requests.Session: Session object configured with retry capability
    """
    session = requests.Session()  # Create a new session
    retry = Retry(  # Configure retry behavior
        total=retries,  # Total number of retries to allow
        read=retries,  # Number of read error retries
        connect=retries,  # Number of connection error retries
        backoff_factor=backoff_factor,  # Backoff factor between retries
        status_forcelist=status_forcelist,  # Status codes that trigger a retry
    )
    adapter = HTTPAdapter(max_retries=retry)  # Create adapter with retry logic
    session.mount('http://', adapter)  # Apply adapter to HTTP requests
    session.mount('https://', adapter)  # Apply adapter to HTTPS requests
    return session  # Return the configured session

def is_valid_url(url):
    """
    Validate if a string is a properly formatted URL using regex pattern matching.
    
    Args:
        url (str): URL string to validate
    
    Returns:
        bool: True if URL is valid, False otherwise
    """
    # Comprehensive URL validation regex pattern
    pattern = re.compile(
        r'^(?:http|https)://'  # Must start with http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # Domain name
        r'localhost|'  # localhost
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # IP address
        r'(?::\d+)?'  # Optional port number
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)  # Path, query string, etc.
    return bool(pattern.match(url))  # Return True if URL matches pattern, False otherwise

def check_scraper_availability():
    """
    Check if the local scraper service is running and responding to requests.
    
    Makes a test request to the scraper service to verify its availability.
    
    Returns:
        bool: True if scraper service is available, False otherwise
    """
    try:
        # Make a simple test request with short timeout
        response = requests.post(
            SCRAPER_URL,  # Local scraper endpoint
            json={'url': 'https://example.com'},  # Simple test URL
            timeout=2  # Short timeout for quick check
        )
        # If we get any response, the service is running
        return True
    except requests.exceptions.RequestException:
        # Service is not running or not responding
        return False

def top_urls(headline, n=2):
    """
    Find top relevant URLs for a given headline using Google Custom Search API.
    
    Extracts keywords from the headline and uses them to search for related articles.
    
    Args:
        headline (str): The news headline to search for
        n (int): Maximum number of URLs to return (default: 2)
        
    Returns:
        dict: Dictionary with 'urls' list and optional 'error' information
    """
    # Validate input headline
    if not headline or not isinstance(headline, str):
        return {"urls": [], "error": "Invalid headline provided"}
    
    # Validate number of results requested
    if not isinstance(n, int) or n <= 0:
        n = 2  # Set to default value if invalid
    
    # Extract keywords from headline for better search results
    keywords = extract_keywords_yake(headline, MAX_KEYWORDS_HEADLINE, 1)
    query = " ".join(keywords)  # Join keywords into search query
    
    # Prepare API request parameters
    search_url = "https://www.googleapis.com/customsearch/v1"  # Google Custom Search API endpoint
    params = {
        "key": API_KEY,  # API key for authentication
        "cx": SEARCH_ENGINE_ID,  # Custom search engine ID
        "q": query,  # Search query built from extracted keywords
        "num": min(n, 10)  # Number of results (max 10 per API constraints)
    }
    session = create_retry_session()  # Create session with retry capability
    
    try:
        # Send request to Google Custom Search API
        response = session.get(search_url, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        data = response.json()  # Parse JSON response
        
        # Handle API errors
        if "error" in data:
            error_msg = data['error'].get('message', 'An error occurred')
            return {"urls": [], "error": f"API Error: {error_msg}"}
        
        # Extract URLs from search results
        urls = [item.get("link") for item in data.get("items", [])]
        valid_urls = [url for url in urls if url and is_valid_url(url)]  # Filter valid URLs
        
        return {"urls": valid_urls, "error": None}
    
    except requests.exceptions.RequestException as e:
        # Handle request errors (network issues, timeouts, etc.)
        return {"urls": [], "error": f"API Request Error: {e}"}

def scrape_article(url):
    """
    Scrape content from a single URL and extract keywords from the headline only.
    
    Uses the local scraper service to extract content from the URL,
    then extracts keywords from the headline for analysis.
    
    Args:
        url (str): URL to scrape
    
    Returns:
        tuple: (url, keywords_or_error) - URL and either keywords or error dict
    """
    # Validate URL format before proceeding
    if not is_valid_url(url):
        return url, {"error": "Invalid URL format"}
    
    try:
        # Create session with retry capability for reliability
        session = create_retry_session()
        
        # Request article content from local scraper service
        response = session.post(
            SCRAPER_URL,  # Local scraper endpoint
            json={'url': url},  # URL to scrape
            timeout=REQUEST_TIMEOUT  # Timeout to prevent hanging
        )
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Parse scraped data from JSON response
        scraped_data = response.json()
        
        # Extract keywords from the headline only
        if 'head' in scraped_data and scraped_data['head']:
            # Extract keywords from headline using YAKE algorithm
            keywords = extract_keywords_yake(scraped_data['head'], MAX_KEYWORDS_HEADLINE)
            return url, keywords
        else:
            # No headline found
            return url, []
            
    except requests.exceptions.RequestException as e:
        # Handle scraper communication errors (connection issues, timeouts)
        return url, {"error": f"Scraper Communication Error: {e}"}
        
    except json.JSONDecodeError as e:
        # Handle JSON parsing errors in scraper response
        return url, {"error": "Scraper JSON Decode Error"}

def scrape_articles(urls, parallel=True):
    """
    Scrape content from multiple URLs and extract keywords from each article.
    
    Processes multiple URLs either in parallel (using threads) or sequentially,
    extracting keywords from each article's headline for analysis.
    
    Args:
        urls (list): List of article URLs to scrape and analyze
        parallel (bool): Whether to process URLs in parallel (default: True)
        
    Returns:
        dict: Dictionary mapping each URL to keywords or error information
    """
    # Validate input
    if not urls:
        return {}  # Return empty dict if no URLs provided
    
    # Check if scraper service is available
    if not check_scraper_availability():
        # Return error for all URLs if scraper is not available
        return {url: {"error": "Scraper service is not available"} for url in urls}
    
    keywords_by_link = {}  # Dictionary to store results
    
    if parallel and len(urls) > 1:
        # Use ThreadPoolExecutor for parallel processing of URLs
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(5, len(urls))) as executor:
            # Submit all scraping tasks
            future_to_url = {executor.submit(scrape_article, url): url for url in urls}
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_url):
                url, result = future.result()  # Get result from completed task
                keywords_by_link[url] = result  # Store in results dictionary
    else:
        # Process URLs sequentially (one at a time)
        for url in urls:
            url, result = scrape_article(url)  # Scrape each article
            keywords_by_link[url] = result  # Store in results dictionary
    
    return keywords_by_link  # Return all results

def analyze_headline(headline=TEST_HEADLINE, max_urls=10):
    """
    Main function to analyze a headline and find related articles.
    
    This is the primary entry point for headline analysis. It extracts keywords,
    finds related articles, scrapes their content, and calculates similarity scores.
    
    Args:
        headline (str): The headline to analyze
        max_urls (int): Maximum number of URLs to retrieve and analyze
        
    Returns:
        dict: Analysis results with keywords, articles, and similarity scores
    """
    # Check if scraper service is available before proceeding
    if not check_scraper_availability():
        return {"error": "Scraper service is not available. Please start the service and try again."}
    
    # Extract keywords from the headline for comparison
    headline_keywords = extract_keywords_yake(headline, MAX_KEYWORDS_HEADLINE)
    
    # Search for relevant URLs based on headline
    search_result = top_urls(headline, max_urls)
    
    # Process search results
    if search_result.get("error"):
        # Return the error from the search process
        return {"error": search_result.get("error")}
    
    elif search_result.get("urls"):
        # Scrape articles to get keywords from each
        article_keywords = scrape_articles(search_result["urls"], parallel=True)
        
        # Calculate similarity scores using the scorer module
        similarity_scores = calculate_similarity_scores(headline_keywords, article_keywords)
        
        # Create enhanced results with both keywords and scores
        enhanced_results = {
            "headline": headline,
            "headline_keywords": headline_keywords,
            "articles": {}
        }
        
        # Process each article's keywords and scores
        for url, keywords in article_keywords.items():
            # Handle URLs with errors
            if isinstance(keywords, dict) and "error" in keywords:
                enhanced_results["articles"][url] = {
                    "keywords": [],
                    "error": keywords["error"],
                    "scores": {}
                }
                continue
                
            # Add keywords and scores for valid results
            enhanced_results["articles"][url] = {
                "keywords": keywords,
                "scores": similarity_scores.get(url, {})
            }
        
        # Return the complete analysis results
        return enhanced_results
    
    else:
        # No articles were found
        return {"error": "No relevant articles found"}


# Example code that runs when the module is executed directly
if __name__ == '__main__':
    # Use the default test headline
    headline = TEST_HEADLINE
    print(f"Using headline: {headline}")
    
    # Check if scraper service is running before proceeding
    if not check_scraper_availability():
        print("Error: Scraper service is not available. Please start the service and try again.")
        exit(1)
    
    # Get analysis results
    results = analyze_headline(headline)
    
    # Check for errors
    if "error" in results:
        print(f"Error: {results['error']}")
    else:
        # Print headline and its keywords
        print(f"Headline: {results['headline']}")
        print(f"Headline keywords: {results['headline_keywords']}")
        
        # Sort articles by combined score
        sorted_articles = sorted(
            results["articles"].items(),
            key=lambda x: x[1].get("scores", {}).get("combined_score", 0),
            reverse=True
        )
        
        # Print formatted results
        print("\nResults sorted by relevance:")
        for url, data in sorted_articles:
            print(f"\n{url}")
            if "error" in data:
                print(f"  Error: {data['error']}")
                continue
                
            print(f"  Keywords: {', '.join(data['keywords'])}")
            
            if "scores" in data and data["scores"]:
                print("  Similarity scores:")
                for metric, score in data["scores"].items():
                    print(f"    {metric}: {score:.3f}")
        
        # Also provide the raw JSON output for programmatic use
        print("\nRaw JSON output:")
        print(json.dumps(results, indent=2))