from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import pypandoc
# from pypandoc.pandoc_download import download_pandoc
# download_pandoc()
from tqdm import tqdm
tokenizer = AutoTokenizer.from_pretrained(
    "NlpHUST/ner-vietnamese-electra-base")
model = AutoModelForTokenClassification.from_pretrained(
    "NlpHUST/ner-vietnamese-electra-base")
nlp = pipeline("ner", model=model, tokenizer=tokenizer)
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


def post_process(text):
    text = text.replace(' ##', '').strip()
    return text


def remove_special_chars(text):
    return ''.join(char for char in text if char.isalnum() or char.isspace())


def predict(text):
    ner_results = nlp(text)
    ner_results = [i for i in ner_results if i['score'] > 0.6]
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
            word = post_process(entity['word'])
            word = remove_special_chars(word)
            if len(word.split()) > 1:
                results[entity_type].append(word)
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

        if 'txt' in file.filename:
            text = file.read().decode("utf-8")
        elif 'docx' in file.filename or 'doc' in file.filename:
            file.save('temp_folder/temp.docx')
            output = pypandoc.convert_file(
                'temp_folder/temp.docx', 'plain', outputfile="temp_folder/temp.txt")
            f = open('temp_folder/temp.txt', mode='r', encoding='utf-8')
            text = '\n'.join(f.readlines())
        lines = text.split('\n')
        for line in tqdm(lines[:1000]):
            result_pred = predict(line)
            for key in keys:
                values = result_pred[key]
                if len(values) > 0:
                    for value in values:
                        results[key].append(value)
        for key in keys:
            results[key] = list(set(results[key]))
        return jsonify(results), 200


if __name__ == '__main__':
    app.run(debug=True)
