# Truth Scope

# News Article Scraper API

This Flask API scrapes the head and body text from a given news article URL.

## Usage

1.  **Clone the repository**
2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    ```
3.  **Activate the virtual environment:**
    * On Windows:
        ```bash
        venv\Scripts\activate
        ```
    * On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```
4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
5.  **Run the Flask app:**
    ```bash
    python articleScraper.py
    ```
6.  **Send a POST request:**
    * Use an API testing tool like Thunder Client or Postman.
    * Send a POST request to `http://127.0.0.1:5000/scrape`.
    * Include a JSON body with the URL to scrape:

        ```json
        {
            "url": "[https://www.example.com/news-article](https://www.example.com/news-article)"
        }
        ```

    * The API will return a JSON response containing the scraped head and body text.

## Dependencies

* Flask
* BeautifulSoup4
* Requests

## Error Handling

* The API returns a 400 status code if no URL is provided.
* The API returns an error message if there are issues accessing the URL or parsing the HTML.
