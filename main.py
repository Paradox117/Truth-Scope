"""
Truth Scope - Main Application

This is the primary entry point for the Truth Scope application that integrates all components:
- Article Scraper: Extracts content from web pages
- Keyword Extractor: Performs text analysis and keyphrase extraction
- Article Collector: Searches for related articles on the web
- Scorer: Calculates similarity and credibility between articles

The application reads a URL or headline from link.txt, analyzes it,
finds related articles, and determines an overall credibility score.
The results are saved as a comprehensive JSON report.
"""

# Standard library imports for core functionality
import os               # File path operations and environment variables
import sys              # System-specific parameters and functions 
import json             # JSON serialization and deserialization
import time             # Time-related functions for delays
import threading        # Thread-based parallelism for running the scraper service
import requests         # HTTP library for making web requests

# Local module imports for application components
from extractor import extract_keywords_yake, preprocess_text  # Text extraction utilities
from collector import analyze_headline, scrape_article, is_valid_url  # Article collection utilities
from articleScraper import start_server as start_scraper_server  # Web scraper service
from scorer import calculate_similarity_scores, aggregate_credibility_score  # Scoring utilities

# Constants used throughout the application
SCRAPER_URL = "http://127.0.0.1:5000/scrape"  # Local endpoint for article scraper service
SCRAPER_STARTUP_TIME = 20  # Seconds to wait for scraper server to start up completely

def load_input_from_file(file_path):
    """
    Read the first line from the specified file (contains URL or headline).
    
    Args:
        file_path (str): Path to the file containing the input text
        
    Returns:
        str or None: The content of the first line, or None if file not found/error
    """
    try:
        # Open file with UTF-8 encoding to handle special characters
        with open(file_path, 'r', encoding='utf-8') as file:
            # Read only the first line and remove any trailing/leading whitespace
            content = file.readline().strip()
            return content
    except Exception:
        # Return None if file not found or any other error occurs
        return None

def start_scraper_service():
    """
    Start the web scraper service in a background thread.
    
    Creates and starts a daemon thread running the Flask-based scraper service,
    which will automatically terminate when the main program exits.
    
    Returns:
        threading.Thread: The thread running the scraper service
    """
    # Create a daemon thread that will automatically terminate when main program exits
    server_thread = threading.Thread(
        target=start_scraper_server,  # Function to run in the thread
        kwargs={'host': '127.0.0.1', 'port': 5000, 'debug': False},  # Server parameters
        daemon=True  # Daemon threads automatically terminate when main program exits
    )
    # Start the thread to run the scraper service
    server_thread.start()
    # Wait for server to initialize (prevents race conditions)
    time.sleep(SCRAPER_STARTUP_TIME)
    return server_thread

def check_scraper_available():
    """
    Check if the scraper service is running and responding to requests.
    
    Makes a test request to the scraper service to verify it's operational.
    
    Returns:
        bool: True if scraper is available and responding, False otherwise
    """
    try:
        # Make a simple test request to the scraper service
        requests.post(SCRAPER_URL, json={'url': 'https://example.com'}, timeout=2)
        # If request completes without error, service is available
        return True
    except requests.exceptions.RequestException:
        # If request fails with any exception, service is not available
        return False

def process_url(url):
    """
    Process a URL by scraping its content and analyzing it for credibility.
    
    Extracts the headline from the URL content, then searches for related
    articles to analyze the credibility of the information.
    
    Args:
        url (str): URL to scrape and analyze
        
    Returns:
        dict: Analysis results including related articles and scores, or error information
    """
    try:
        # Request the scraper service to extract content from the URL
        response = requests.post(SCRAPER_URL, json={'url': url}, timeout=10)
        response.raise_for_status()  # Raise exception for 4XX/5XX status codes
        
        # Parse the JSON response from the scraper
        scraped_data = response.json()
        
        # If scraper encountered an error, return it
        if 'error' in scraped_data:
            return {"error": scraped_data['error']}
            
        # Check if a headline was successfully extracted
        if 'head' not in scraped_data or not scraped_data['head']:
            return {"error": "Failed to extract headline"}
            
        # Use the extracted headline for further analysis
        headline = scraped_data['head']
        
        # Analyze the headline to find related articles and assess credibility
        return analyze_headline(headline)
    
    except Exception as e:
        # Catch and return any unexpected errors
        return {"error": str(e)}

def generate_comprehensive_report(input_text, results, credibility_result):
    """
    Generate a comprehensive report with all analysis details.
    
    Combines all the information from the analysis into a structured
    report format with credibility assessment and source information.
    
    Args:
        input_text (str): Original URL or headline used for analysis
        results (dict): Raw analysis results containing article data
        credibility_result (dict): Credibility assessment data
        
    Returns:
        dict: Complete structured report with all analysis data
    """
    # Initialize the base report structure
    report = {
        # Input section contains the original query and its type
        "input": {
            "text": input_text,
            "type": "url" if is_valid_url(input_text) else "headline"
        },
        # Credibility section contains the overall assessment
        "credibility": {
            "headline": results.get("headline", ""),
            "keywords": results.get("headline_keywords", []),
            "total_score": credibility_result.get("total_score", 0),
            "credibility_level": credibility_result.get("credibility_level", "unknown"),
            "interpretation": credibility_result.get("interpretation", ""),
            "sources_analyzed": credibility_result.get("sources_analyzed", 0)
        },
        # Sources section will contain individual article data
        "sources": []
    }
    
    # Add up to 10 sources with their detailed scores
    if "articles" in results:
        # Sort articles by weighted score (descending) and take top 10
        sorted_articles = sorted(
            results["articles"].items(),
            key=lambda x: x[1].get("scores", {}).get("weighted_score", 0),
            reverse=True
        )[:10]  # Limit to top 10 sources
        
        # Process each source and add to the report
        for url, data in sorted_articles:
            scores = data.get("scores", {})
            # Create structured source information
            source_info = {
                "url": url,
                "title": data.get("title", ""),
                "raw_similarity": scores.get("raw_similarity", 0),
                "source_weight": scores.get("source_weight", 1.0),
                "weighted_score": scores.get("weighted_score", 0),
                "similarity_method": scores.get("similarity_method", "unknown")
            }
            report["sources"].append(source_info)
    
    # Add source weight information used in calculations
    if len(report["sources"]) > 0:
        weights_used = {}
        for source in report["sources"]:
            # Extract domain from URL for weight tracking
            domain = source["url"].split("//")[-1].split("/")[0]
            # Remove www. prefix for consistency
            if "www." in domain:
                domain = domain.replace("www.", "")
            weights_used[domain] = source["source_weight"]
        
        # Add weights to report for transparency
        report["weights_used"] = weights_used
    
    return report

def main():
    """
    Main application function that orchestrates the entire workflow.
    
    Handles the full pipeline from input loading to analysis to report generation.
    Returns the generated report structure which is also saved as JSON.
    
    Returns:
        dict: Generated credibility report or error information
    """
    # Start the scraper service in the background
    start_scraper_service()
    
    # Verify that the scraper service is running
    if not check_scraper_available():
        return {"error": "Scraper service unavailable"}
    
    # Load input from the link.txt file
    link_file = os.path.join(os.path.dirname(__file__), 'link.txt')
    input_text = load_input_from_file(link_file)
    
    # Check if input was successfully loaded
    if not input_text:
        return {"error": "No input found in link.txt"}
    
    # Process input differently based on whether it's a URL or headline
    if is_valid_url(input_text):
        # Process as a URL by scraping content first
        results = process_url(input_text)
    else:
        # Process directly as a headline
        results = analyze_headline(input_text)
    
    # Check if any errors occurred during processing
    if "error" in results:
        # Create minimal report with error information
        report = {
            "input": input_text,
            "error": results["error"],
            "credibility_level": "unknown",
            "interpretation": "Unable to assess credibility due to error"
        }
    else:
        # Calculate credibility score from the analysis results
        credibility_result = calculate_credibility_score(results)
        
        # Generate comprehensive report with all details
        report = generate_comprehensive_report(input_text, results, credibility_result)
    
    # Save the report to a JSON file
    try:
        report_file = os.path.join(os.path.dirname(__file__), 'credibility_report.json')
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)  # Pretty-print with 2-space indentation
    except Exception as e:
        return {"error": f"Failed to save report: {str(e)}"}
    
    # Return the generated report
    return report

def calculate_credibility_score(results):
    """
    Calculate the overall credibility score from analysis results.
    
    Extracts scores from all analyzed articles and calculates an
    aggregate credibility assessment with score and interpretation.
    
    Args:
        results (dict): Analysis results containing article data and scores
        
    Returns:
        dict: Credibility assessment with score and interpretation
    """
    # Handle error cases
    if "error" in results:
        return {
            "error": results["error"],
            "total_score": 0,
            "credibility_level": "unknown",
            "interpretation": "Unable to assess credibility due to error"
        }
        
    # Handle case with no articles found
    if "articles" not in results or not results["articles"]:
        return {
            "error": "No related articles found for comparison",
            "total_score": 0,
            "credibility_level": "unknown",
            "interpretation": "Unable to assess credibility (no articles found)"
        }
    
    # Extract all valid article scores
    article_scores = {}
    for url, data in results["articles"].items():
        if "scores" in data:
            article_scores[url] = data["scores"]
    
    # Calculate aggregated credibility score using the scorer module
    credibility_assessment = aggregate_credibility_score(article_scores)
    
    # Create credibility result dictionary with all relevant information
    credibility_result = {
        "headline": results.get("headline", ""),
        "headline_keywords": results.get("headline_keywords", []),
        "total_score": credibility_assessment["total_score"],
        "credibility_level": credibility_assessment["credibility_level"],
        "interpretation": credibility_assessment["interpretation"],
        "sources_analyzed": credibility_assessment["sources_analyzed"]
    }
    
    return credibility_result

# Entry point when script is run directly
if __name__ == "__main__":
    try:
        # Run the main function
        main()
        # Exit with success code
        sys.exit(0)
    except Exception:
        # Exit with error code if any uncaught exception occurs
        sys.exit(1)