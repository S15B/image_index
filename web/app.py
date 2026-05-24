import os
import sys
import time
import pickle
import numpy as np
from flask import Flask, render_template, request, send_from_directory

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'index'))

from query_processor import search
from dedup import deduplicate


app = Flask(__name__)

DATA_DIR = os.path.join(BASE_DIR, 'data')
INDEX_DIR = os.path.join(DATA_DIR, 'index')
IMAGES_DIR = os.path.join(DATA_DIR, 'extracted_images')
DOCUMENTS_PATH = os.path.join(DATA_DIR, 'unified_documents.json')


indices = {}

for field in ['context', 'description', 'full']:
    for comp in ['none', 'gamma', 'delta']:
        filename = f"index_{field}_{comp}.pkl"
        filepath = os.path.join(INDEX_DIR, filename)
        with open(filepath, 'rb') as f:
            indices[(field, comp)] = pickle.load(f)

tokenizer_path = os.path.join(INDEX_DIR, 'tokenizer.pkl')
with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)

import json
with open(DOCUMENTS_PATH, 'r', encoding='utf-8') as f:
    docs_list = json.load(f)
docs = {item['id']: item for item in docs_list}


@app.route('/')
def index():
    """Главная страница с формой поиска."""
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def do_search():
    """Обработка поискового запроса."""
    query_text = request.form.get('query', '').strip()
    field = request.form.get('field', 'full')        # all -> full
    compression = request.form.get('compression', 'none')

    if field not in ('context', 'description', 'full'):
        field = 'full'
    if compression not in ('none', 'gamma', 'delta'):
        compression = 'none'

    if not query_text:
        return render_template('results.html',
                               query=query_text,
                               field=field,
                               compression=compression,
                               results=[],
                               elapsed=0)

    index = indices[(field, compression)]

    start_time = time.perf_counter()
    doc_ids_array = search(index, query_text, tokenizer)
    elapsed = (time.perf_counter() - start_time) * 1000

    doc_ids = doc_ids_array.tolist() if isinstance(doc_ids_array, np.ndarray) else list(doc_ids_array)

    results = []
    for doc_id in doc_ids:
        doc = docs.get(doc_id)
        if doc is None:
            continue
        image_rel_path = doc['image_path']
        image_url = f"/images/{image_rel_path}"
        results.append({
            'id': doc['id'],
            'image_url': image_url,
            'context': doc.get('context', ''),
            'description_ru': doc.get('description_ru', ''),
            'source_url': doc.get('source_url', ''),
            'phash': doc.get('phash'),
        })

    results = deduplicate(results, by_phash=True, by_url=True)

    return render_template('results.html',
                           query=query_text,
                           field=field,
                           compression=compression,
                           results=results,
                           elapsed=elapsed)

@app.route('/images/<path:filename>')
def serve_image(filename):
    """Отдача статических изображений."""
    return send_from_directory(IMAGES_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)
