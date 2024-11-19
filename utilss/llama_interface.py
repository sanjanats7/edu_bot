import subprocess
from utilss.faiss_indexer import retrieve_documents


def query_llama(prompt):
    try:
        # Remove `shell=True` or pass command as a single string if using `shell=True`
        result = subprocess.run(
            ["ollama", "run", "llama3.1", prompt], capture_output=True, text=True
        )
        if result.stderr:
            print("Error:", result.stderr)
            return None  # Return None if there is an error
        return result.stdout
    except Exception as e:
        print("An exception occurred:", e)
        return None


def retrieve_and_answer(query, index, documents):
    # Retrieve relevant document chunk
    retrieved_indices = retrieve_documents(query, index)
    if not retrieved_indices:
        print("No documents retrieved.")
        return None

    print("Retrieved indices:", retrieved_indices)  # Debugging output

    # Get the closest document
    closest_doc = documents[
        retrieved_indices[0][0]
    ]  # Ensure `retrieved_indices` format is correct

    # Formulate the query
    full_query = (
        f"Based on the following document: {closest_doc}, answer the question: {query}"
    )
    print("Full query sent to LLaMA:", full_query)  # Debugging output

    # Query LLaMA and return the answer
    answer = query_llama(full_query)
    if answer is None:
        print("Failed to get a response from LLaMA.")
    return answer
