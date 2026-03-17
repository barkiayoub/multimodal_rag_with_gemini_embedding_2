import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from rag import query_rag
from pathlib import Path

app = Flask(__name__, template_folder='template')

# Serve images from the 'images' and 'pages_cache' directories
@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory('images', filename)

@app.route('/pages_cache/<path:filename>')
def serve_cache(filename):
    return send_from_directory('pages_cache', filename)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/query', methods=['POST'])
def query():
    data = request.json
    question = data.get('question')
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    answer, sources = query_rag(question)
    
    # Normalize image paths for web URLs
    for s in sources:
        raw_path = s.get('image_path', '')
        # Convert backslashes to forward slashes
        web_path = raw_path.replace('\\', '/')
        # Ensure it starts with /
        if not web_path.startswith('/'):
            web_path = '/' + web_path
        s['image_path'] = web_path
        print(f"  DEBUG: Page {s['page_number']}: {raw_path} -> {web_path}")
    
    return jsonify({
        'answer': answer,
        'sources': sources
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False)
