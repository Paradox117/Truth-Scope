"""
Truth Scope Scorer Module

This module provides functions for comparing keyphrases and calculating
similarity scores between articles based on their keyword content.
It offers multiple similarity metrics with a prioritized approach:
1. Semantic similarity using transformers (when available)
2. Jaccard similarity and Overlap coefficient as fallbacks

The module features graceful degradation if advanced NLP libraries 
are not available, maintaining functionality with simpler methods.
"""

# Standard library imports
import re  # For regular expression pattern matching and text cleaning
import numpy as np  # For numerical operations on arrays
from collections import Counter  # For efficient counting of elements

# Try to import advanced NLP libraries with fallbacks for each
try:
    # NLTK for linguistic preprocessing (stemming and stopword removal)
    from nltk.stem import PorterStemmer  # For word stemming
    from nltk.corpus import stopwords  # For stopword filtering
    NLTK_AVAILABLE = True  # Flag indicating NLTK is available
except ImportError:
    NLTK_AVAILABLE = False  # Flag indicating NLTK is not available

try:
    # Sentence transformers for semantic similarity using embeddings
    from sentence_transformers import SentenceTransformer  # For text embeddings
    TRANSFORMERS_AVAILABLE = True  # Flag indicating transformers are available
except ImportError:
    TRANSFORMERS_AVAILABLE = False  # Flag indicating transformers are not available

# Initialize advanced NLP components if available
if NLTK_AVAILABLE:
    try:
        # Initialize stemmer for reducing words to their root form
        stemmer = PorterStemmer()  # Create stemmer instance
        # Load English stopwords (common words like "the", "and", etc.)
        stop_words = set(stopwords.words('english'))  # Create set for efficient lookup
    except:
        # Handle case where NLTK data is not downloaded
        import nltk  # Import NLTK for downloading resources
        nltk.download('stopwords')  # Download stopwords resource
        stop_words = set(stopwords.words('english'))  # Try loading stopwords again
        
if TRANSFORMERS_AVAILABLE:
    try:
        # Load a lightweight sentence transformer model for semantic similarity
        # all-MiniLM-L6-v2 offers good balance between performance and speed
        model = SentenceTransformer('all-MiniLM-L6-v2')  # Initialize model
    except:
        # Mark transformers as unavailable if loading fails
        TRANSFORMERS_AVAILABLE = False  # Set flag to false


def preprocess_keyphrases(keyphrases, advanced=True):
    """
    Enhanced preprocessing of keyphrases to prepare for similarity comparison.
    
    Converts phrasal keywords into individual words with preprocessing options:
    - Basic: Lowercase conversion and word splitting
    - Advanced: Additional stemming, stopword removal, and character filtering
    
    Args:
        keyphrases (list): List of keyphrases to process
        advanced (bool): Whether to use advanced preprocessing (default: True)
        
    Returns:
        list: List of processed keywords ready for comparison
    """
    # Handle empty input case
    if not keyphrases:
        return []  # Return empty list for empty input
        
    # Basic preprocessing (used when advanced=False or NLTK is unavailable)
    if not advanced or not NLTK_AVAILABLE:
        keywords = []  # Initialize empty list for processed keywords
        for phrase in keyphrases:  # Process each keyphrase
            if phrase:  # Skip empty phrases
                # Convert to lowercase and split into words
                words = phrase.lower().split()  # Split by whitespace
                keywords.extend(words)  # Add words to result list
        return keywords  # Return processed keywords
    
    # Advanced preprocessing with NLTK (when available and requested)
    keywords = []  # Initialize empty list for processed keywords
    for phrase in keyphrases:  # Process each keyphrase
        if phrase:  # Skip empty phrases
            # Convert to lowercase and split into words
            words = phrase.lower().split()  # Split by whitespace
            
            # Process each word with advanced techniques
            for word in words:
                # Remove non-alphabetic characters using regex
                word = re.sub(r'[^a-z]', '', word)  # Keep only a-z
                
                # Apply stemming and filter stopwords
                if word and word not in stop_words:  # Skip empty words and stopwords
                    stemmed = stemmer.stem(word)  # Reduce to root form
                    keywords.append(stemmed)  # Add processed word to result list
    
    return keywords  # Return processed keywords


def jaccard_similarity(main_keywords, other_keywords):
    """
    Calculate the Jaccard similarity between two lists of keywords.
    
    Jaccard similarity is defined as the size of the intersection divided
    by the size of the union of two sets. It ranges from 0 to 1, where 0 means
    no overlap and 1 means identical sets.
    
    Args:
        main_keywords (list): First list of keywords
        other_keywords (list): Second list of keywords
        
    Returns:
        float: Jaccard similarity score (0-1)
    """
    # Convert to lowercase sets for case-insensitive comparison
    set_main = {word.lower() for word in main_keywords if word}  # Filter out empty strings
    set_other = {word.lower() for word in other_keywords if word}  # Filter out empty strings
    
    # Calculate intersection and union sizes
    intersection = len(set_main & set_other)  # Size of common elements
    union = len(set_main | set_other)  # Size of all unique elements
    
    # Calculate similarity score
    return intersection / union if union != 0 else 0  # Avoid division by zero


def overlap_coefficient(main_keywords, other_keywords):
    """
    Calculate the Overlap Coefficient between two lists of keywords.
    
    Overlap coefficient is defined as the size of the intersection divided
    by the size of the smaller set. It prioritizes matches in the smaller set
    and is useful when sets differ significantly in size.
    
    Args:
        main_keywords (list): First list of keywords
        other_keywords (list): Second list of keywords
        
    Returns:
        float: Overlap coefficient score (0-1)
    """
    # Convert to lowercase sets for case-insensitive comparison
    set_main = {word.lower() for word in main_keywords if word}  # Filter out empty strings
    set_other = {word.lower() for word in other_keywords if word}  # Filter out empty strings
    
    # Calculate intersection size and smaller set size
    intersection = len(set_main & set_other)  # Size of common elements
    smaller_set_size = min(len(set_main), len(set_other))  # Size of the smaller set
    
    # Calculate similarity score
    return intersection / smaller_set_size if smaller_set_size != 0 else 0  # Avoid division by zero


def semantic_similarity(main_keyphrases, other_keyphrases):
    """
    Calculate semantic similarity using neural sentence embeddings.
    
    This function uses a transformer-based model to convert text into
    high-dimensional vectors, then calculates cosine similarity between them.
    This captures semantic meaning beyond exact word matches.
    
    Args:
        main_keyphrases (list): First list of keyphrases
        other_keyphrases (list): Second list of keyphrases
        
    Returns:
        float: Semantic similarity score (0-1)
    """
    # Return 0 if transformers are not available
    if not TRANSFORMERS_AVAILABLE:
        return 0.0  # Fallback value when transformers aren't available
        
    # Convert keyphrase lists to single text strings for embedding
    main_text = " ".join(main_keyphrases)  # Join phrases with spaces
    other_text = " ".join(other_keyphrases)  # Join phrases with spaces
    
    # Generate sentence embeddings (vector representations)
    main_embedding = model.encode([main_text])[0]  # Encode first text
    other_embedding = model.encode([other_text])[0]  # Encode second text
    
    # Calculate cosine similarity between embeddings
    # Formula: cos(θ) = (A·B)/(||A||×||B||)
    similarity = np.dot(main_embedding, other_embedding) / (
        np.linalg.norm(main_embedding) * np.linalg.norm(other_embedding)  # Vector magnitudes
    )
    
    # Return similarity as float (converting from numpy types if needed)
    return float(similarity)  # Ensure Python float return type


def calculate_similarity_scores(main_keyphrases, links_keyphrases_dict, use_advanced=True):
    """
    Calculate credibility scores using similarity multiplied by source credibility weight.
    
    Args:
        main_keyphrases (list): Keyphrases from the main article.
        links_keyphrases_dict (dict): Dictionary mapping links to their keyphrases.
        use_advanced (bool): Whether to use semantic similarity.

    Returns:
        dict: Dictionary mapping each link to its weighted score and metrics.
    """
    from urllib.parse import urlparse

    # Update the source_weights dictionary with dramatically increased values for credible sources only
    source_weights = {
        # Government Sources (Highest Boost - 10x)
        "cdc.gov": 10.0,           # US Centers for Disease Control
        "nih.gov": 10.0,           # National Institutes of Health
        "who.int": 10.0,           # World Health Organization
        "un.org": 10.0,            # United Nations
        "europa.eu": 10.0,         # European Union
        "nasa.gov": 10.0,          # NASA
        "noaa.gov": 10.0,          # National Oceanic and Atmospheric Administration
        "education.gov": 10.0,      # US Department of Education
        "defense.gov": 10.0,       # US Department of Defense
        "state.gov": 10.0,         # US Department of State
        "treasury.gov": 10.0,      # US Department of Treasury
        "fbi.gov": 10.0,           # Federal Bureau of Investigation
        "cia.gov": 10.0,           # Central Intelligence Agency
        "whitehouse.gov": 10.0,    # The White House
        "congress.gov": 10.0,      # US Congress
        "supreme.court.gov": 10.0,  # US Supreme Court
        "nist.gov": 10.0,          # National Institute of Standards and Technology
        "usgs.gov": 10.0,          # US Geological Survey
        "epa.gov": 10.0,           # Environmental Protection Agency
        "fda.gov": 10.0,           # Food and Drug Administration
        
        # Indian Government Sources (10x)
        "india.gov.in": 10.0,      # National Portal of India
        "mygov.in": 10.0,          # MyGov India
        "nic.in": 10.0,            # National Informatics Centre
        "meity.gov.in": 10.0,      # Ministry of Electronics and IT
        "mohfw.gov.in": 10.0,      # Ministry of Health and Family Welfare
        "mea.gov.in": 10.0,        # Ministry of External Affairs
        "mod.gov.in": 10.0,        # Ministry of Defence
        "mha.gov.in": 10.0,        # Ministry of Home Affairs
        "pib.gov.in": 10.0,        # Press Information Bureau
        "rbi.org.in": 10.0,        # Reserve Bank of India
        "supremecourt.gov.in": 10.0, # Supreme Court of India
        "censusindia.gov.in": 10.0, # Census of India
        "data.gov.in": 10.0,       # Open Government Data Platform
        "niti.gov.in": 10.0,       # NITI Aayog
        "isro.gov.in": 10.0,       # Indian Space Research Organisation
        "drdo.gov.in": 10.0,       # Defence Research and Development Organisation
        "education.gov.in": 10.0,   # Ministry of Education
        "nhm.gov.in": 10.0,        # National Health Mission
        
        # Research and Educational Institutions (.edu domains - 8x)
        "harvard.edu": 8.0,
        "mit.edu": 8.0,
        "stanford.edu": 8.0,
        "berkeley.edu": 8.0,
        "columbia.edu": 8.0,
        "princeton.edu": 8.0,
        "yale.edu": 8.0,
        "caltech.edu": 8.0,
        "cornell.edu": 8.0,
        "ox.ac.uk": 8.0,
        "cam.ac.uk": 8.0,
        "imperial.ac.uk": 8.0,
        "edinburgh.ac.uk": 8.0,
        "iisc.ac.in": 8.0,  # Indian Institute of Science
        # Fact-Checking Organizations (Very High Boost)
        "snopes.com": 6.0,
        "factcheck.org": 5.8,
        "politifact.com": 5.6,
        "altnews.in": 5.6,
        "boomlive.in": 5.6,
        "factchecker.in": 5.5,
        "reporters-lab.org": 5.4,
        "climatefeedback.org": 5.4,
        "verificat.cat": 5.4,
        "vishvasnews.com": 5.4,
        "newschecker.in": 5.4,
        "webqoof.com": 5.3,
        "factcrescendo.com": 5.3,

        # Reputable International News Sources (High Boost)
        "bbc.com": 5.0,
        "reuters.com": 5.0,
        "theguardian.com": 4.5,
        "nytimes.com": 4.5,
        "apnews.com": 4.2,
        "wsj.com": 4.2,
        "economist.com": 4.2,
        "cfr.org": 4.0,
        "npr.org": 4.0,
        "pbs.org": 4.0,
        "cnn.com": 3.8,
        "euronews.com": 3.8,
        "ft.com": 3.8,
        "bloomberg.com": 3.8,
        "cbsnews.com": 3.5,
        "nbcnews.com": 3.5,
        "abcnews.go.com": 3.3,
        "globalnews.ca": 3.2,
        "smh.com.au": 3.2,
        "theage.com.au": 3.2,
        "stuff.co.nz": 3.2,
        "aljazeera.com": 3.2,
        "france24.com": 3.2,
        "dw.com": 3.2,

        # Reputable Indian News Sources (High - Moderate Boost)
        "thehindu.com": 3.5,
        "indianexpress.com": 3.5,
        "livemint.com": 3.3,
        "scroll.in": 3.2,
        "thewire.in": 3.2,
        "theprint.in": 3.0,
        "newslaundry.com": 3.0,
        "caravanmagazine.in": 2.8,
        "tribuneindia.com": 2.8,
        "telegraphindia.com": 2.8,
        "business-standard.com": 2.8,
        "financialexpress.com": 2.8,
        "outlookindia.com": 2.7,
        "timesofindia.indiatimes.com": 2.5,
        "hindustantimes.com": 2.5,
        "economictimes.indiatimes.com": 2.5,
        "thebridge.in": 2.4,
        "thequint.com": 2.4,
        "indiatoday.in": 2.4,
        "aninews.in": 2.2,
        "ndtv.com": 2.2,
        "indiaspend.com": 2.2,
        "pib.gov.in": 2.2,
        "prsindia.org": 2.2,
        "moneycontrol.com": 2.1,
        "firstpost.com": 2.1,
        "newindianexpress.com": 2.1,
        "deccanherald.com": 2.1,
        "dnaindia.com": 2.0,
        "downtoearth.org.in": 2.0,
        "thehindubusinessline.com": 2.0,

        # Regional Indian News Sources (Neutral to Slight Boost)
        "mathrubhumi.com": 1.8,
        "manoramaonline.com": 1.8,
        "anandabazar.com": 1.8,
        "eenadu.net": 1.8,
        "dailythanthi.com": 1.8,
        "amarujala.com": 1.7,
        "jagran.com": 1.7,
        "bhaskar.com": 1.7,
        "sakshi.com": 1.7,
        "lokmat.com": 1.7,
        "punjabkesari.in": 1.6,
        "sandesh.com": 1.6,
        "asomiyapratidin.in": 1.6,
        "prabhatkhabar.com": 1.6,
        "kashmirobserver.net": 1.6,

        # Less Reliable Sources (UNCHANGED)
        "opindia.com": 0.85,
        "swarajyamag.com": 0.85,
        "tfipost.com": 0.8,
        "postcard.news": 0.7,
        "rightlog.in": 0.7,
        "kreately.in": 0.7,
        "pgurus.com": 0.7,
        "organiser.org": 0.8,
        "intellectualkshatriya.com": 0.7,
        "fakingnews.com": 0.5,  # Satire site
        "nationalherald.com": 0.85,
        "thestatesman.com": 0.9,
        
        # Tabloids and Entertainment-focused (UNCHANGED)
        "mid-day.com": 0.9,
        "mumbaimirror.com": 0.9,
        "bollywoodhungama.com": 0.8,
        "pinkvilla.com": 0.8,
        "filmfare.com": 0.8,
        "sportskeeda.com": 0.9,
        
        # Clickbait and Dubious Sources (UNCHANGED)
        "greatgameindia.com": 0.6,
        "thedailyswitch.com": 0.6,
        "newsbharati.com": 0.7,
        "hindupost.in": 0.7,
        "mynation.net": 0.6,

        # Video-based News Sources (SELECTIVELY INCREASED)
        "ndtv.com/videos": 2.0,  # Was 1.0
        "timesnownews.com": 0.95,  # UNCHANGED
        "news18.com": 0.95,  # UNCHANGED
        "abplive.com": 0.95,  # UNCHANGED
        "republicworld.com": 0.85,  # UNCHANGED
        "zeenews.india.com": 0.9,  # UNCHANGED
        "tv9bharatvarsh.com": 0.9,  # UNCHANGED
        "indiatvnews.com": 0.85,  # UNCHANGED
        "news24online.com": 0.85,  # UNCHANGED
    }

    default_multiplier = 1.0
    main_keywords = preprocess_keyphrases(main_keyphrases, advanced=use_advanced)
    scores = {}

    for link, keyphrases in links_keyphrases_dict.items():
        scores[link] = {}
        
        # Determine source weight
        try:
            domain = urlparse(link).netloc
            parts = domain.split('.')
            if len(parts) > 2 and parts[-2] != 'co':
                base_domain = ".".join(parts[-2:])
            elif len(parts) > 3 and parts[-2] == 'co':
                base_domain = ".".join(parts[-3:])
            else:
                base_domain = domain

            weight = source_weights.get(base_domain, default_multiplier)
        except Exception:
            weight = default_multiplier

        scores[link]["source_weight"] = weight

        # Handle error cases
        if isinstance(keyphrases, dict) and "error" in keyphrases:
            scores[link]["weighted_score"] = 0.0
            scores[link]["error"] = keyphrases["error"]
            continue

        # Calculate similarity based on available methods
        if use_advanced and TRANSFORMERS_AVAILABLE:
            similarity = semantic_similarity(main_keyphrases, keyphrases)
            scores[link]["similarity_method"] = "semantic"
        else:
            # Fall back to traditional metrics with weighted combination
            other_keywords = preprocess_keyphrases(keyphrases, advanced=use_advanced)
            jaccard = jaccard_similarity(main_keywords, other_keywords)
            overlap = overlap_coefficient(main_keywords, other_keywords)
            
            # Use weighted average of traditional metrics (60% jaccard, 40% overlap)
            similarity = (jaccard * 0.6) + (overlap * 0.4)
            
            scores[link]["jaccard_similarity"] = round(jaccard, 3)
            scores[link]["overlap_coefficient"] = round(overlap, 3)
            scores[link]["similarity_method"] = "traditional"

        # Store raw similarity and compute weighted score
        scores[link]["raw_similarity"] = round(similarity, 3)
        scores[link]["weighted_score"] = round(similarity * weight, 3)

    return scores


def aggregate_credibility_score(scores, thresholds=None):
    """
    Sums all weighted similarity scores and evaluates overall credibility.

    Args:
        scores (dict): Output from calculate_similarity_scores.
        thresholds (dict, optional): Custom thresholds for different credibility levels.

    Returns:
        dict: Contains total score and credibility assessment.
    """
    if thresholds is None:
        thresholds = {
            "high": 12.0,    # High credibility threshold
            "moderate": 8.0,  # Moderate credibility threshold
            "fair": 5.0,     # Fair credibility threshold
            "low": 2.0       # Low credibility threshold
        }
    
    # Sum all weighted scores (skip errors)
    valid_scores = [
        info.get("weighted_score", 0.0)
        for info in scores.values()
        if "error" not in info
    ]
    
    # Handle case with no valid scores
    if not valid_scores:
        return {
            "total_score": 0.0,
            "credibility_level": "unknown",
            "interpretation": "Unable to assess credibility (no valid sources)"
        }
    
    total_score = sum(valid_scores)
    
    # Determine credibility level based on thresholds
    if total_score >= thresholds["high"]:
        level = "high"
        interpretation = "High credibility - Information is well-supported by reliable sources"
    elif total_score >= thresholds["moderate"]:
        level = "moderate"
        interpretation = "Moderate credibility - Information has good support from credible sources"
    elif total_score >= thresholds["fair"]:
        level = "fair"
        interpretation = "Fair credibility - Some support from reliable sources"
    elif total_score >= thresholds["low"]:
        level = "low"
        interpretation = "Low credibility - Limited support from reliable sources"
    else:
        level = "very_low"
        interpretation = "Very low credibility - Minimal or no support from reliable sources"
    
    # Return comprehensive assessment
    return {
        "total_score": round(total_score, 3),
        "credibility_level": level,
        "interpretation": interpretation,
        "sources_analyzed": len(valid_scores)
    }


# Example usage (runs when script is executed directly)
if __name__ == "__main__":
    # Test data with sample keyphrases
    main_keyphrases = [
        "Rain",  # Sample keyword 1
        "Delhi",  # Sample keyword 2
        "turn",  # Sample keyword 3
        "dust"  # Sample keyword 4
    ]
    
    # Sample articles with their keyphrases for testing
    links_keyphrases_dict = {
        "https://www.bbc.com/news/delhi-weather": [  # BBC link (high credibility)
            "dust storms bring",
            "storms bring temperatures",
            "Delhi weather",
            "sudden turn"
        ],
        "https://timesofindia.indiatimes.com/weather": [  # TOI link (medium credibility)
            "dust storm brings",
            "Delhi NCR breaths",
            "storm brings respite",
            "NCR breaths easy"
        ],
        "https://example-news.com/delhi": [  # Unknown source (default credibility)
            "Dust Bowl",
            "Bowl",
            "Dust",
            "India"
        ]
    }

    # Print information about available methods
    print("Available similarity methods:")
    print(f"- Basic preprocessing: Always available")  # Always works
    print(f"- NLTK advanced preprocessing: {NLTK_AVAILABLE}")  # Depends on installation
    print(f"- Semantic similarity: {TRANSFORMERS_AVAILABLE}")  # Depends on installation
    print()

    # Calculate similarity scores for test data
    similarity_scores = calculate_similarity_scores(
        main_keyphrases,  # Main article keyphrases
        links_keyphrases_dict,  # Other articles' keyphrases
        use_advanced=True  # Use advanced methods if available
    )
    
    # Get the single aggregated credibility score
    credibility_score = aggregate_credibility_score(similarity_scores)
    
    # Sort links by combined score (best matches first)
    sorted_links = sorted(
        similarity_scores.items(),  # Items to sort
        key=lambda x: x[1]["weighted_score"],  # Sort by weighted score
        reverse=True  # Descending order (highest scores first)
    )
    
    # Display results in human-readable format
    for link, scores in sorted_links:
        print(f"\n{link}:")  # Print link name
        for metric, score in scores.items():  # Print each metric
            print(f"  {metric}: {score}")  # Display score
    
    # Display the overall credibility score
    print(f"\nAggregated Credibility Score: {credibility_score['total_score']}")
    print(f"Credibility Level: {credibility_score['credibility_level']}")
    print(f"Interpretation: {credibility_score['interpretation']}")