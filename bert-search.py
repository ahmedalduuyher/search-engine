import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity


class BERTSearch:
    def __init__(self, model_name='bert-base-uncased', batch_size=8):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.batch_size = batch_size

    def _make_batches(self, seq):
        # Splits the input sequence into batches of the given size
        for i in range(0, len(seq), self.batch_size):
            yield seq[i:i + self.batch_size]

    def compute_embeddings(self, texts):
        # Computes BERT-based embeddings for a list of input texts
        embeddings = []

        for batch in tqdm(self._make_batches(texts)):
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors='pt').to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state.mean(dim=1)
                embeddings.append(batch_embeddings.cpu().numpy())

        return np.vstack(embeddings)

    def fit(self, df, text_column='text'):
        # Stores the dataset and computes embeddings for the given text column
        self.df = df.reset_index(drop=True)
        self.texts = df[text_column].tolist()
        self.embeddings = self.compute_embeddings(self.texts)

    def search(self, query, n_results=10, return_scores=True):
        # Returns the top n most similar entries to the given query
        query_emb = self.compute_embeddings([query])
        scores = cosine_similarity(self.embeddings, query_emb).flatten()
        idx = np.argsort(-scores)[:n_results]

        result = self.df.iloc[idx].copy()
        if return_scores:
            result['similarity'] = scores[idx]

        return result


# Example usage
texts = [
    "Yes, we will keep all the materials after the course finishes.",
    "You can follow the course at your own pace after it finishes"
]
df = pd.DataFrame({'text': texts})

search_engine = BERTSearch()
search_engine.fit(df, 'text')

query = "Can I still access the materials later?"
results = search_engine.search(query)

print("Top matches:")
print(results)
