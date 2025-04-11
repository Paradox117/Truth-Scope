"""
Keyword Extraction Module for Truth Scope

This module provides functions for extracting keywords from text content using different
extraction techniques. It supports both guided extraction with semantic understanding 
(KeyBERT) and statistical extraction (YAKE).

Main functionalities:
- Guided keyword extraction using KeyBERT with domain-specific seed words
- Unsupervised keyword extraction using YAKE
- Text preprocessing for cleaning input before extraction
"""

# Import necessary libraries
from keybert import KeyBERT  # For semantically guided keyword extraction
import yake  # For unsupervised, statistical keyword extraction
import re  # For regular expression operations in text preprocessing

# Initialize KeyBERT model with a sentence-transformer model
# 'all-mpnet-base-v2' is a powerful model with strong semantic understanding
kw_model = KeyBERT('all-mpnet-base-v2')  # Load model only once for reuse across function calls


def extract_guided_keywords(article_text, seed_words=['economy', 'inflation', 'budget', 'GDP', 'investment', 'startup', 'rupee', 'RBI', 'election', 'coalition', 'manifesto', 'democracy', 'parliament', 'governance', 'monsoon', 'climate', 'pollution', 'sustainability', 'renewable', 'carbon', 'technology', 'artificial intelligence', 'digital', 'cybersecurity', 'UPI', '5G', 'healthcare', 'vaccine', 'pandemic', 'telemedicine', 'hospital', 'insurance', 'education', 'NEP', 'university', 'online learning', 'skill development', 'cricket', 'Olympics', 'athlete', 'tournament', 'championship', 'Bollywood', 'OTT', 'cinema', 'streaming', 'box office', 'diplomacy', 'bilateral', 'security', 'trade agreement', 'defense', 'infrastructure', 'metro', 'smart city', 'housing', 'urbanization', 'supreme court', 'legislation', 'verdict', 'amendment', 'judicial', 'agriculture', 'farmer', 'crop', 'MSP', 'food security', 'unemployment', 'workforce', 'labor', 'migration', 'industry', 'stock market', 'interest rate', 'fiscal deficit', 'taxation', 'GST', 'military', 'border', 'strategic', 'defense deal', 'naval', 'vaccination', 'medical research', 'virus', 'corruption', 'transparency', 'accountability', 'lokpal', 'vigilance', 'entrepreneur', 'funding', 'innovation', 'venture capital', 'ecommerce', 'terrorism', 'internal security', 'intelligence', 'extremism', 'border security', 'reservation', 'social justice', 'inclusion', 'minority', 'affirmative action', 'water crisis', 'river linking', 'groundwater', 'dam', 'irrigation', 'cryptocurrency', 'fintech', 'digital payment', 'banking', 'financial inclusion', 'tourism', 'heritage', 'wildlife', 'ecotourism', 'hospitality', 'space program', 'satellite', 'ISRO', 'mission', 'aerospace', 'transport', 'electric vehicle', 'highway', 'railway', 'aviation', 'government', 'policy', 'minister', 'regulation', 'reform', 'leadership', 'dissent', 'debate', 'opposition', 'constituency', 'campaign', 'administration', 'judiciary', 'federal', 'state', 'finance', 'market', 'trade', 'fiscal', 'tax', 'commerce', 'manufacturing', 'economic', 'recession', 'recovery', 'growth', 'jobs', 'union', 'road', 'construction', 'urban', 'development', 'realestate', 'robotics', 'blockchain', 'automation', 'internet', 'application', 'software', 'hardware', 'telecom', 'mobile', 'research', 'science', 'laboratory', 'discovery', 'astrophysics', 'quantum', 'nuclear', 'energy', 'solar', 'wind', 'biofuel', 'conservation', 'biodiversity', 'green', 'ecology', 'treatment', 'medicine', 'doctor', 'care', 'wellness', 'nutrition', 'disease', 'mental', 'therapy', 'fitness', 'match', 'coach', 'IPL', 'score', 'record', 'event', 'training', 'film', 'actor', 'actress', 'drama', 'music', 'celebrity', 'festival', 'review', 'award', 'art', 'theatre', 'culture', 'literature', 'dance', 'reality', 'show', 'school', 'curriculum', 'exam', 'scholarship', 'learning', 'classroom', 'teacher', 'pedagogy', 'community', 'activism', 'protest', 'rights', 'equality', 'election reforms', 'coalition government', 'federalism', 'judicial activism', 'anti-corruption', 'reservation policy', 'caste dynamics', 'minority rights', 'border disputes', 'national security', 'diplomatic relations', 'RTI activism', 'GST reforms', 'inflation trends', 'FDI inflows', 'MSME sector', 'agricultural GDP', 'startup ecosystem', 'unicorn valuations', 'rural entrepreneurship', 'formalization push', 'skill gap', 'gig economy', 'PPP projects', 'AI governance', 'semiconductor push', 'deep-tech startups', 'data localization', 'edtech adoption', 'drone regulations', '6G readiness', 'coal dependency', 'air quality', 'water scarcity', 'climate resilience', 'solar adoption', 'EV infrastructure', 'carbon markets', 'Himalayan ecology', 'coastal erosion', 'waste management', 'green hydrogen', 'gender equality', 'urban migration', 'farmer distress', 'healthcare access', 'digital divide', 'religious harmony', 'mental health', 'ageing population', 'nutrition schemes', 'tribal rights', 'sanitation drive', 'rural unemployment', 'Bollylywood trends', 'OTT censorship', 'cricket economy', 'yoga diplomacy', 'religious tourism', 'regional cinema', 'fusion cuisine', 'fast fashion', 'matrimonial apps', 'vernacular content', 'heritage conservation', 'festival economy', 'IIT placements', 'STEM initiatives', 'global rankings', 'reservation in education', 'philanthropic funding', 'academic collaborations', 'rural literacy', 'EdTech mergers', 'port modernization', 'rural electrification', 'logistics network', 'optical fiber', 'warehousing boom', 'transit-oriented development', 'organic farming', 'crop insurance', 'warehouse receipts', 'agri-tech', 'fertilizer subsidies', 'food processing', 'land leasing', 'drought mitigation', 'farmer producer organizations', 'soil health', 'SAARC relations', 'diaspora engagement', 'Indo-Pacific strategy', 'strategic autonomy', 'defense exports', 'soft power', 'remittance flows', 'global south', 'climate negotiations', 'dollar-rupee dynamics', 'energy diplomacy', 'privacy laws', 'cybercrime', 'consumer rights', 'land acquisition', 'IPR disputes', 'marriage laws', 'free speech', 'right to education', 'refugee policy', 'surrogacy laws', 'anticipatory bail']):
    """
    Extract keywords using the guided KeyBERT method with a predefined list of seed words.
    
    This function leverages a pretrained language model to identify semantically relevant 
    keywords in the text that are conceptually similar to the provided seed words.
    The approach helps in domain-specific topic extraction.
    
    Args: 
        article_text (str): The content to extract keywords from (optimized for news articles)
        seed_words (list): List of domain-specific seed words to guide the extraction process.
                          Default includes a comprehensive list of common Indian news topics.
                          
    Returns:
        list: A list of extracted keywords (without scores)
    
    Note:
        The function uses the MaxSum algorithm to ensure diversity in the extracted keywords.
    """
    
    # Extract keywords using KeyBERT with the following parameters:
    # - keyphrase_ngram_range: Extract phrases of 1-3 words
    # - stop_words: Remove English stopwords
    # - highlight: Don't highlight keywords in text
    # - top_n: Extract 10 most relevant keywords
    # - use_maxsum: Use MaxSum algorithm for diversity
    # - nr_candidates: Consider 20 candidates before selecting final keywords
    # - seed_keywords: Use the provided seed words for guidance
    keywords = kw_model.extract_keywords(
        article_text,
        keyphrase_ngram_range=(1, 3),  # Extract single words to 3-word phrases
        stop_words='english',  # Remove common English stopwords
        highlight=False,  # Don't highlight keywords in text
        top_n=10,  # Return top 10 keywords
        use_maxsum=True,  # Use MaxSum algorithm for diversity
        nr_candidates=20,  # Consider 20 candidates before selection
        seed_keywords=seed_words  # Use provided seed words for guidance
    )
    
    # Return only the keywords without their scores
    return [keyword for keyword, score in keywords]


def extract_keywords_yake(article_text, top_n=10, n=3):
    """
    Extract keywords using the YAKE (Yet Another Keyword Extractor) method.
    
    YAKE is an unsupervised, statistical method that relies on text features
    like word position, frequency, and co-occurrence to identify important keywords.
    Unlike KeyBERT, it doesn't require a pretrained language model.
    
    Args: 
        article_text (str): The text content to analyze for keyword extraction
        top_n (int): Number of keywords to extract (default: 10)
        n (int): Maximum number of words in each keyword phrase (default: 3)
                    
    Returns: 
        list: A list of extracted keywords (without scores)
        
    Note:
        YAKE is particularly effective for short to medium-length texts and
        extracts keywords based on statistical features rather than semantics.
    """
    
    # Initialize YAKE keyword extractor with parameters:
    # - lan: Language of the text
    # - n: Maximum ngram size (up to n-word phrases)
    # - top: Number of keywords to extract
    yake_model = yake.KeywordExtractor(
        lan="en",    # English language
        n=n,         # Extract up to n-word phrases
        top=top_n    # Extract specified number of keywords
    )
    
    # Extract keywords with YAKE (returns keywords with scores)
    keywords = yake_model.extract_keywords(article_text)
    
    # Return only the keywords without their scores
    # YAKE scores are inverse - lower scores mean better keywords
    return [keyword for keyword, score in keywords]


def preprocess_text(text):
    """
    Preprocess text for better keyword extraction results.
    
    This function cleans input text by removing excessively long words 
    (often garbage or formatting artifacts), joining the filtered words,
    and removing special characters that might interfere with extraction.
    
    Args:
        text (str): Raw text input that needs preprocessing
        
    Returns:
        str: Cleaned and processed text ready for keyword extraction
        
    Note:
        Preprocessing is essential for improving extraction quality by
        reducing noise and normalizing the text.
    """
    
    # Split text into individual words
    words = text.split()
    
    # Filter out excessively long words (often URLs, IDs, or garbage text)
    filtered_words = [word for word in words if len(word) <= 25]
    
    # Rejoin the filtered words with spaces
    preprocessed_text = " ".join(filtered_words)
    
    # Remove backslashes and plus characters that might interfere with extraction
    preprocessed_text = re.sub(r"[\\+]", "", preprocessed_text)
    
    return preprocessed_text


# Sample article text for testing purposes
article_text = """
Enter text for testing here
"""
