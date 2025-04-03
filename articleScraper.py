from flask import Flask, request, jsonify
from bs4 import BeautifulSoup
import requests

app = Flask(__name__)

def scrape_article(url):
    try:
        response = requests.get(url)
        response.raise_for_status() #exception for bad status codes like 4## or 5##
        soup = BeautifulSoup(response.content,'html.parser')
        head_text = ''.join([text for text in soup.head.stripped_strings]) if soup.head else ""
        body_text = ''.join([text for text in soup.body.stripped_strings]) if soup.body else ""
        return {"head":head_text, "body":body_text}
    except requests.exceptions.RequestException as e:
        return {"error": f"Request error type {e}"}
    except Exception as e:
        return {"error": f"Code is cooked, {e} occured"}
    
@app.route('/scrape', methods = ['POST'])
def scrape():
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({"error": "No URL given"}), 400
    url = data['url']
    result = scrape_article(url)
    return jsonify(result)
if __name__== '__main__':
    app.run(debug=True)