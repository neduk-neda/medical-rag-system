from sentence_transformers import SentenceTransformer
import numpy as np

# Dataset
docs = [
    "Diabetes: High blood sugar over a long period can damage organs.",
    "Hypertension: High blood pressure increases risk of heart disease.",
    "Asthma: Chronic disease affecting airways and breathing.",
    "Obesity increases risk of diabetes and heart disease.",
    "Smoking is a major risk factor for lung cancer."
]

# Model
model = SentenceTransformer('all-MiniLM-L6-v2')
doc_embeddings = model.encode(docs)

# Search function
def search_topk(query, k=3):
    query_embedding = model.encode([query])
    scores = np.dot(doc_embeddings, query_embedding.T).flatten()
    top_k_indices = scores.argsort()[-k:][::-1]
    return [docs[i] for i in top_k_indices]

# Chat function
def ask(query):
    contexts = search_topk(query)
    context_text = "\n".join(contexts)

    return f"""
Question: {query}

Relevant medical knowledge:
{context_text}

Explanation:
This answer is generated using semantic retrieval from medical knowledge base.
"""

# Test
if name == "__main__":
    print(ask("high blood pressure"))