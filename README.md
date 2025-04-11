
# Truth Scope

Truth Scope is a news credibility analysis tool that evaluates the reliability of news articles and headlines by comparing them with trusted sources.

---

## Project Overview

Truth Scope helps users determine the credibility of news articles by:

- Extracting key information from an input article or headline  
- Finding related articles from various sources  
- Analyzing semantic similarity between the original and related content  
- Applying source credibility weighting based on domain reputation  
- Generating a comprehensive credibility score and assessment report  

The tool uses a combination of natural language processing, web scraping, and similarity algorithms to provide an objective credibility evaluation.

---

## Key Features

- Multi-source Verification: Compares content against multiple credible sources  
- Source Credibility Weighting: Assigns different weights to sources based on reliability  
- Semantic Analysis: Understands the meaning of text beyond keyword matching  
- Graceful Degradation: Falls back to simpler methods when advanced NLP isn't available  
- Comprehensive Reporting: Detailed JSON output with credibility assessment  

---

## Architecture

Truth Scope consists of several interconnected modules:

- `main.py` – Coordinates the overall workflow and generates the final report  
- `articleScraper.py` – Flask service for extracting content from web pages  
- `collector.py` – Finds related articles using Google Custom Search API  
- `extractor.py` – Extracts key phrases from text using YAKE and KeyBERT  
- `scorer.py` – Calculates similarity scores and overall credibility assessment  

---

## Installation

### Prerequisites

- Python 3.8+  
- Google Custom Search API credentials  
- Internet connection for API requests  

### Setup

1. Clone the repository

2. Set up a virtual environment

On Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

On macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Get Google API Credentials

- Create a [Programmable Search Engine](https://programmablesearchengine.google.com/)
- Get your **Search Engine ID (cx)**
- Generate an **API Key** from [Google Cloud Console](https://console.cloud.google.com/)

5. Create a `.env` file in the project root

```env
GOOGLE_API_KEY=your_api_key_here
SEARCH_ENGINE_ID=your_search_engine_id_here
```

6. Create an empty `link.txt` file in the project root

```bash
touch link.txt
```

---

## Usage

1. Add a news URL or headline to `link.txt` (single line)

Example:

```
https://www.example.com/news/article-about-science
```

2. Run the main script

```bash
python main.py
```

3. View the results

The output will be saved in `credibility_report.json`.

---

## Credibility Levels

| Level       | Description                                      | Score Range |
|-------------|--------------------------------------------------|-------------|
| High        | Strong support from multiple reliable sources    | ≥ 12.0      |
| Moderate    | Good support from credible sources               | ≥ 8.0       |
| Fair        | Some support from reliable sources               | ≥ 5.0       |
| Low         | Limited support from reliable sources            | ≥ 2.0       |
| Very Low    | Minimal or no support from reliable sources      | < 2.0       |

---

## Source Weighting

| Source Type                    | Weight     |
|--------------------------------|------------|
| Government / Academic          | 8.0–10.0   |
| Fact-Checking Organizations    | 5.3–6.0    |
| Major International News       | 3.2–5.0    |
| Major National News            | 2.0–3.5    |
| Regional News                  | 1.6–1.8    |
| Less Reliable Sources          | 0.5–0.9    |

---

## Technical Details

- Similarity Metrics: Semantic similarity (transformers), Jaccard, overlap coefficient  
- Keyword Extraction: YAKE (statistical) + KeyBERT (semantic)  
- Web Scraping: BeautifulSoup4  
- API Integration: Google Custom Search API  
- Output Format: Structured JSON report  

---

## Output Format

The tool generates a JSON report saved in `credibility_report.json` with the following structure:

```json
{
  "input": {
    "text": "original input text",
    "type": "url or headline"
  },
  "credibility": {
    "headline": "analyzed headline",
    "keywords": ["keyword1", "keyword2", "..."],
    "total_score": 8.5,
    "credibility_level": "moderate",
    "interpretation": "Information has good support from credible sources",
    "sources_analyzed": 12
  },
  "sources": [
    {
      "url": "https://example.com/article1",
      "title": "Article title",
      "raw_similarity": 0.85,
      "source_weight": 4.2,
      "weighted_score": 3.57,
      "similarity_method": "semantic"
    }
  ],
  "weights_used": {
    "example.com": 4.2
  }
}
```

---

## Dependencies

- `Flask`  
- `beautifulsoup4`  
- `requests`  
- `keybert`  
- `transformers`  
- `sentence-transformers`  
- `scikit-learn`  
- `torch`  
- `yake`  
- `python-dotenv`  
- `nltk`  
- `numpy`  

Install all dependencies via:

```bash
pip install -r requirements.txt
```

---

## Limitations

- Requires a valid API key with sufficient quota  
- May produce lower scores for new/breaking news with limited coverage  
- Relies on third-party APIs which may change or become unavailable  
- Speed depends on network connectivity and API response time  
```

---

Let me know if you'd like this saved as a file or need a markdown preview!
