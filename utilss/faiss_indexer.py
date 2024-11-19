import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load your SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")  # Example model


def create_faiss_index(documents):
    # Encode the documents
    doc_embeddings = model.encode(
        documents, convert_to_tensor=False
    )  # Ensures numpy array
    doc_embeddings = np.array(
        doc_embeddings, dtype=np.float32
    )  # Convert to float32 for FAISS compatibility

    # Check embedding shape and type
    print("Document embeddings shape:", doc_embeddings.shape)
    print("Document embeddings dtype:", doc_embeddings.dtype)

    # Initialize the FAISS index using L2 (Euclidean) distance
    index = faiss.IndexFlatL2(doc_embeddings.shape[1])

    # Add embeddings to the index
    index.add(doc_embeddings)
    return index


def retrieve_documents(query, index, k=1):
    # Encode the query
    query_embedding = model.encode([query], convert_to_tensor=False)
    query_embedding = np.array(
        query_embedding, dtype=np.float32
    )  # Ensure float32 for FAISS compatibility

    # Check query embedding shape and type
    print("Query embedding shape:", query_embedding.shape)
    print("Query embedding dtype:", query_embedding.dtype)

    # Perform the search in the index
    D, I = index.search(query_embedding, k)  # D: distances, I: indices
    print("Distances:", D)
    print("Indices:", I)

    return I  # Return indices of the retrieved documents
