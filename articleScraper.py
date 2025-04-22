"""
Truth Scope Article Scraper

This module provides functionality for scraping web articles,
extracting their content, and preprocessing the text.

Key features:
- Flask-based web service for article scraping
- HTML parsing using newspaper3k
- Content extraction focusing on relevant article elements
- Text preprocessing to clean and normalize article content
- Error handling for robust operation

It runs as a service that accepts URLs via HTTP POST requests
and returns the extracted article content as JSON.
"""

# Import Flask for web service functionality
from flask import Flask, request, jsonify  # Web framework and request/response handling
from extractor import preprocess_text  # Local text preprocessing function
from newspaper import Article  # Newspaper3k for article extraction

# Create Flask app object without running it automatically
app = Flask(__name__)  # Initialize Flask application

def scrape_article(url):
    """
    Scrape and extract content from a given URL using newspaper3k.
    
    Fetches the article using newspaper3k, extracts title and text,
    and applies preprocessing.
    
    Args:
        url (str): URL of the article to scrape
        
    Returns:
        dict: Dictionary containing head and body text, or error message
              - 'head': String with title and metadata text
              - 'body': String with main article content
              - 'error': Error message if scraping fails
    """
    try:
        article = Article(url)
        article.download()
        article.parse()
        # Preprocess title and main text
        head_text = preprocess_text(article.title)
        body_text = preprocess_text(article.text)
        return {"head": head_text, "body": body_text}
    except Exception as e:
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
