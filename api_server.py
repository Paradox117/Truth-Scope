from flask import Flask, request, jsonify
from main import main as run_analysis

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    input_data = data.get('input', {})
    text = input_data.get('text', '')
    type_ = input_data.get('type', 'url')
    # Optionally, write text to link.txt if needed by main.py
    with open('link.txt', 'w', encoding='utf-8') as f:
        f.write(text.strip())
    # Run analysis (main() returns the report dict)
    report = run_analysis()
    return jsonify(report)

if __name__ == '__main__':
    app.run(port=3000)