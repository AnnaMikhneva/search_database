from fastapi import FastAPI
import time
from flask import Flask, render_template, request, redirect, session, jsonify
from flaskext.mysql import MySQL
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertModel
from typing import List, Tuple
from pydantic import BaseModel
from enum import Enum
import mysql.connector
import joblib
# app = FastAPI()

app = Flask(__name__)
app.secret_key = 'your_secret_key'
# MySQL configuration
app.config['MYSQL_DATABASE_USER'] = 'anna'
app.config['MYSQL_DATABASE_PASSWORD'] = 'LessWeak$$$_32'
app.config['MYSQL_DATABASE_DB'] = 'library'
app.config['MYSQL_DATABASE_HOST'] = 'localhost'

# mysql = MySQL()
# mysql.init_app(app)

with open('bert_matrix.pkl', 'rb') as f:
    bert_matrix = pickle.load(f)

with open('tfidf_matrix.pkl', 'rb') as f:
    tfidf_matrix = pickle.load(f)


class IndexType(str, Enum):
    bert = "bert"
    tfidf = "tf-idf"


class SearchRequest(BaseModel):
    user_id: int  # Добавляем user_id в модель запроса
    query: str
    index_type: IndexType
    top_k: int


class SearchResponse(BaseModel):
    results: List[Tuple[str, float]]


class CorpusInfo(BaseModel):
    name: str
    token_count: int


class AvailableMethodsResponse(BaseModel):
    methods: List[str]

vectorizer = TfidfVectorizer()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
obr_df = pd.read_csv('obr_df.csv')
annotations_obr = obr_df['abstract'].tolist()
new_df = pd.read_csv('new_df.csv')
final = new_df['key'].tolist()


def index_corpus(annotations_obr, model_filename='tfidf.pkl'):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(annotations_obr)
    joblib.dump(vectorizer, model_filename)
    return tfidf_matrix, vectorizer


# Индексируем корпус
tfidf_matrix, vectorizer = index_corpus(annotations_obr)


def calculate_similarity(query_embedding, document_embeddings):
    similarity_scores = cosine_similarity(query_embedding.reshape(1, -1), document_embeddings)
    return similarity_scores.flatten()


def search(query: str, index_type: str, top_k: int) -> List[Tuple[str, float]]:
    if index_type == 'bert':
        query_inputs = tokenizer(query, return_tensors='pt')
        query_outputs = model(**query_inputs)
        query_embedding = query_outputs.pooler_output.detach().numpy()
        similarity_scores = calculate_similarity(query_embedding, bert_matrix)
    elif index_type == 'tf-idf':
        query_vector = vectorizer.transform([query])
        similarity_scores = calculate_similarity(query_vector, tfidf_matrix).flatten()
    else:
        raise ValueError("Invalid index type. Please choose 'bert' or 'tf-idf'.")
    top_indices = similarity_scores.argsort()[-top_k:][::-1]
    return [(annotations_obr[i], similarity_scores[i]) for i in top_indices]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        full_name = request.form['full_name']
        age = request.form['age']
        education_field = request.form['education_field']
        conn = mysql.connector.connect(
            user=app.config['MYSQL_DATABASE_USER'],
            password=app.config['MYSQL_DATABASE_PASSWORD'],
            host=app.config['MYSQL_DATABASE_HOST'],
            database=app.config['MYSQL_DATABASE_DB']
        )
        cursor = conn.cursor()
        cursor.execute("INSERT INTO User (full_name, age, education_field) VALUES (%s, %s, %s)",
                       (full_name, age, education_field))
        conn.commit()
        user_id = cursor.lastrowid
        session['user_id'] = user_id
        cursor.close()
        conn.close()
        return redirect(f'/search?user_id={user_id}')# Redirect to search page after registration
    return render_template('register.html')


@app.route('/search', methods=['GET', 'POST'])
def perform_search():
    user_id = session.get('user_id')
    if request.method == 'POST':
        query = request.form.get('query')
        index_type = request.form.get('index_type')
        top_k = int(request.form.get('top_k', 5))

        conn = mysql.connector.connect(
            user=app.config['MYSQL_DATABASE_USER'],
            password=app.config['MYSQL_DATABASE_PASSWORD'],
            host=app.config['MYSQL_DATABASE_HOST'],
            database=app.config['MYSQL_DATABASE_DB']
        )
        cursor = conn.cursor()
        cursor.execute("INSERT INTO Query (user_id, query_text) VALUES (%s, %s)", (user_id, query))
        conn.commit()
        cursor.close()
        conn.close()

        start_time = time.time()
        results = search(query, index_type, top_k)
        execution_time = time.time() - start_time
        print(f"Execution Time: {execution_time:.4f} seconds")
        response = SearchResponse(results=results)
        return render_template('search_page.html', results=response.results, user_id=user_id, query=query)

    return render_template('search.html')  # Render search ```python


@app.route('/search_page', methods=['GET'])
def search_page():
    return render_template('search_page.html')  # Render search results page


@app.route('/api/available_methods', methods=['GET'])
def available_methods():
    methods = [method.value for method in IndexType]
    response = AvailableMethodsResponse(methods=methods)
    return jsonify(response.dict())


@app.route('/api/corpus_info', methods=['GET'])
def corpus_info():
    data = {
        "title": "Corpus of linguistic abstracts",
        "description": "Corpus consists of abstracts retrieved from linguistic articles, journals and etc",
        "size": 5000
    }
    return jsonify(data)


@app.route('/api/search', methods=['POST'])
def api_search():
    data = request.get_json()
    search_request = SearchRequest(**data)
    results = search(search_request.query, search_request.index_type, search_request.top_k)
    response = SearchResponse(results=results)
    return jsonify(response.dict())


@app.route('/api/search_with_relevance', methods=['POST'])
def api_search_with_relevance():
    data = request.get_json()
    search_request = SearchRequest(**data)
    results = search(search_request.query, search_request.index_type, search_request.top_k)
    response = SearchResponse(results=results)  # Assuming relevance scores are included in the results
    return jsonify(response.dict())


if __name__ == '__main__':
    app.run(debug=True)
