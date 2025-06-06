import requests
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Load and prepare the data
docs_url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
response = requests.get(docs_url)
documents_raw = response.json()

# Flatten and normalize documents
documents = []
for course in documents_raw:
    course_name = course['course']
    for doc in course['documents']:
        doc['course'] = course_name
        documents.append(doc)

# Create DataFrame
df = pd.DataFrame(documents, columns=['course', 'section', 'question', 'text'])

# Step 2: Define search engine class
class TextSearch:
    def __init__(self, text_fields):
        self.text_fields = text_fields
        self.vectorizers = {}
        self.matrices = {}

    def fit(self, records, vectorizer_params={}):
        self.df = pd.DataFrame(records)
        for field in self.text_fields:
            tfidf = TfidfVectorizer(**vectorizer_params)
            matrix = tfidf.fit_transform(self.df[field])
            self.vectorizers[field] = tfidf
            self.matrices[field] = matrix

    def search(self, query, n_results=5, boost=None, filters=None):
        if boost is None:
            boost = {}
        if filters is None:
            filters = {}

        score = np.zeros(len(self.df))

        for field in self.text_fields:
            weight = boost.get(field, 1.0)
            q_vec = self.vectorizers[field].transform([query])
            similarity = cosine_similarity(self.matrices[field], q_vec).flatten()
            score += weight * similarity

        for field, value in filters.items():
            mask = (self.df[field] == value).astype(int)
            score *= mask

        top_idx = np.argsort(-score)[:n_results]
        return self.df.iloc[top_idx].to_dict(orient='records')

# Step 3: Instantiate and use the search engine
index = TextSearch(text_fields=['section', 'question', 'text'])
index.fit(documents, vectorizer_params={'stop_words': 'english', 'min_df': 3})

results = index.search(
    query='I just signed up. Is it too late to join the course?',
    n_results=5,
    boost={'question': 3.0},
    filters={'course': 'data-engineering-zoomcamp'}
)

# Print top results
for i, res in enumerate(results, 1):
    print(f"--- Result {i} ---")
    print(f"Section: {res['section']}")
    print(f"Question: {res['question']}")
    print(f"Answer: {res['text']}\n")