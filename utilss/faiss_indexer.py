import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load your SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Example model

def create_faiss_index(documents):
    doc_embeddings = model.encode(documents)
    index = faiss.IndexFlatL2(doc_embeddings.shape[1])  # Using L2 distance
    index.add(np.array(doc_embeddings, dtype=np.float32))
    return index

def retrieve_documents(query, index, k=1):
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding, dtype=np.float32), k)  # k nearest neighbors
    return I  # Indices of the retrieved documents
