from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
tokenizer = AutoTokenizer.from_pretrained(
    "NlpHUST/ner-vietnamese-electra-base")
model = AutoModelForTokenClassification.from_pretrained(
    "NlpHUST/ner-vietnamese-electra-base")
nlp = pipeline("ner", model=model, tokenizer=tokenizer)
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


def post_process(text):
    text = text.replace(' ##', '')
    return text


def predict(text):
    ner_results = nlp(text)
    merged_entities = []
    current_entity = None
    for entity in ner_results:
        if entity['entity'].startswith('B-'):
            if current_entity:
                merged_entities.append(current_entity)
            current_entity = entity
        elif entity['entity'].startswith('I-'):
            if current_entity and current_entity['entity'][2:] == entity['entity'][2:]:
                current_entity['word'] += ' ' + entity['word']
                current_entity['end'] = entity['end']
            else:
                if current_entity:
                    merged_entities.append(current_entity)
                current_entity = entity
        else:
            if current_entity:
                merged_entities.append(current_entity)
            current_entity = None

    if current_entity:
        merged_entities.append(current_entity)

    keys = ['PERSON', 'LOCATION']
    results = {}
    for key in keys:
        results[key] = []
    for entity in merged_entities:
        entity_type = entity['entity'].split('-')[1]
        if entity_type in keys:
            results[entity_type].append(post_process(entity['word']))
    return results


@app.route('/ner', methods=['POST'])
def extract_entities():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file:
        keys = ['PERSON', 'LOCATION']
        results = {}
        for key in keys:
            results[key] = []
        text = file.read().decode("utf-8")
        lines = text.split('\n')
        for line in lines:
            arr = line.split('.')
            for a in arr:
                result_pred = predict(a)
                for key in keys:
                    values = result_pred[key]
                    if len(values) > 0:
                        for value in values:
                            results[key].append(value)
        return jsonify(results), 200
if __name__ == '__main__':
    app.run(debug=True)
