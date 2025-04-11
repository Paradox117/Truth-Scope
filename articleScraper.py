"""
Truth Scope Article Scraper

This module provides functionality for scraping web articles,
extracting their content, and preprocessing the text.

Key features:
- Flask-based web service for article scraping
- HTML parsing using BeautifulSoup
- Content extraction focusing on relevant article elements
- Text preprocessing to clean and normalize article content
- Error handling for robust operation

It runs as a service that accepts URLs via HTTP POST requests
and returns the extracted article content as JSON.
"""

# Import Flask for web service functionality
from flask import Flask, request, jsonify  # Web framework and request/response handling
from bs4 import BeautifulSoup  # HTML parsing library
import requests  # HTTP requests library
from extractor import preprocess_text  # Local text preprocessing function

# Create Flask app object without running it automatically
app = Flask(__name__)  # Initialize Flask application

def scrape_article(url):
    """
    Scrape and extract content from a given URL.
    
    Fetches the HTML content from the URL, parses it using BeautifulSoup,
    extracts useful text from the head and body, and applies preprocessing.
    
    Args:
        url (str): URL of the article to scrape
        
    Returns:
        dict: Dictionary containing head and body text, or error message
              - 'head': String with title and metadata text
              - 'body': String with main article content
              - 'error': Error message if scraping fails
    """
    try:
        # Set user agent to avoid being blocked by websites
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/122.0.0.0 Safari/537.36"  # Modern browser user agent
        }
        # Make HTTP request with timeout and headers
        response = requests.get(url, headers=headers, timeout=10)  # 10-second timeout
        response.raise_for_status()  # Raise exception for bad status codes (4XX, 5XX)

        # Parse HTML content using BeautifulSoup with html.parser
        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove noisy tags that typically contain non-article content
        # This improves relevance of extracted text
        for tag in soup(['script', 'style', 'nav', 'footer', 'aside', 'noscript']):
            tag.decompose()  # Remove tag and its contents from the DOM

        # Extract <head> text for metadata and title
        head_element = soup.head  # Get head element
        head_text = ''  # Initialize empty string for head text
        if head_element:
            # Join all text strings in head element with spaces
            head_raw_text = ' '.join(head_element.stripped_strings)
            # Preprocess head text to clean and normalize it
            head_text = preprocess_text(head_raw_text)

        # Extract meaningful content from <body>, focusing on relevant tags
        body_text = ''  # Initialize empty string for body text
        if soup.body:
            # Find all paragraph and heading tags (main content)
            content_tags = soup.body.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'])
            # Join all text from content tags with spaces
            body_raw_text = ' '.join(tag.get_text(strip=True) for tag in content_tags if tag.get_text(strip=True))
            # Preprocess body text to clean and normalize it
            body_text = preprocess_text(body_raw_text)

        # Return dictionary with extracted content
        return {"head": head_text, "body": body_text}

    except requests.exceptions.RequestException as e:
        # Handle request exceptions (timeouts, connection errors, etc.)
        return {"error": f"Request failed: {e}"}
    except Exception as e:
        # Handle any other exceptions
        return {"error": f"Something went wrong: {e}"}

# Define Flask route for scraping endpoint
@app.route('/scrape', methods=['POST'])
def scrape():
    """
    Flask endpoint for scraping articles via HTTP POST requests.
    
    Expects JSON request with a 'url' field containing the URL to scrape.
    Returns JSON with 'head' and 'body' fields containing extracted text,
    or an error message if scraping fails.
    
    Request format:
        {"url": "https://example.com/article"}
    
    Response format (success):
        {"head": "Article title and metadata", "body": "Article content"}
    
    Response format (error):
        {"error": "Error message"}
    
    Returns:
        flask.Response: JSON response with article content or error message
    """
    # Extract JSON data from request
    data = request.get_json()
    
    # Check if request contains URL
    if not data or 'url' not in data:
        return jsonify({"error": "No URL given"}), 400  # Bad request
    
    # Extract URL from request
    url = data['url']
    
    # Scrape article content
    result = scrape_article(url)
    
    # Return result as JSON
    return jsonify(result)

# Function to start the Flask server (called from main.py)
def start_server(host='127.0.0.1', port=5000, debug=False):
    """
    Start the Flask server for article scraping.
    
    This function is called from main.py to start the scraper service.
    It configures and runs the Flask application with the specified parameters.
    
    Args:
        host (str): Host address to bind to (default: '127.0.0.1')
        port (int): Port to listen on (default: 5000)
        debug (bool): Whether to run in debug mode (default: False)
        
    Returns:
        None: The function blocks while the server is running
    """
    # Run Flask application with specified parameters
    app.run(host=host, port=port, debug=debug)

# No automatic server start when imported as a module
# The server is started by main.py calling the start_server function
