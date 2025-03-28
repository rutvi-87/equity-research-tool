from flask import Flask, request, jsonify, render_template
import main

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # Assumes index.html is in a templates/ directory

@app.route('/process_urls', methods=['POST'])
def process_urls():
    data = request.json
    urls = [data.get('url1'), data.get('url2'), data.get('url3')]
    response = main.process_urls(urls)
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
