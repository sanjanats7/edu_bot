import subprocess
from utilss.faiss_indexer import retrieve_documents

def query_llama(prompt):
    result = subprocess.run(['ollama', 'chat', 'llama3', '-q', prompt],shell=True, capture_output=True, text=True)
    if result.stderr:
        print("Error:", result.stderr)
    return result.stdout

def retrieve_and_answer(query, index, documents):
    # Retrieve relevant document chunk
    retrieved_indices = retrieve_documents(query, index)
    closest_doc = documents[retrieved_indices[0][0]]  # Get the closest document

    # Pass the retrieved document chunk to LLaMA
    full_query = f"Based on the following document: {closest_doc}, answer the question: {query}"
    answer = query_llama(full_query)
    return answer
