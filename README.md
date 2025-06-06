# Simple Text Search Engines: TF-IDF & BERT Embeddings

This repo contains two separate text search engines to find relevant documents or sentences in a dataset:

- **TF-IDF Search:** A classic, fast keyword-based search engine using `CountVectorizer` and TF-IDF weighting to rank documents based on keyword importance.
- **BERT Search:** A semantic search engine using BERT embeddings from Hugging Face transformers to find contextually relevant results based on sentence meaning.

### TF-IDF Search
- Uses `CountVectorizer` to tokenize the dataset text and build a term-frequency matrix.
- Converts it into a TF-IDF matrix using `TfidfTransformer` to weigh terms by their importance across the dataset.
- Computes cosine similarity between the TF-IDF vector of a query and all documents.
- Returns top matching documents ranked by similarity.
- **TF-IDF Search** is simple and fast. It looks for keyword overlap weighted by term importance. But it doesn’t understand word order or context, so it can miss meaning behind phrases. Useful for small datasets that don't require the raw power that transformer architectures provide.

### BERT Search
- Uses Hugging Face’s `transformers` library with the `bert-base-uncased` model.
- Tokenizes input texts and generates contextual token embeddings.
- Creates sentence embeddings by averaging token embeddings for each sentence.
- Supports batch processing of texts for efficiency.
- Computes cosine similarity between the query embedding and document embeddings to find semantic matches.
- **BERT Search** uses deep learning to understand the meaning of sentences. It captures word order and context, so it can find relevant texts even when the exact keywords don’t match. It is however much slower, and sometimes overkill for smaller datasets.