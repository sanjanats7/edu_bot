import streamlit as st
from utilss.document_processor import extract_text_from_pdf
from utilss.faiss_indexer import create_faiss_index, retrieve_documents
from utilss.llama_interface import query_llama, retrieve_and_answer

# File uploader
uploaded_files = st.file_uploader("Upload your study materials", accept_multiple_files=True)

# Initialize index and documents list
index = None
documents = []

if uploaded_files:
    for file in uploaded_files:
        if file.type == "application/pdf":
            text = extract_text_from_pdf(file)
            documents.append(text)
            st.write(f"File uploaded: {file.name}")

    # Create FAISS index
    index = create_faiss_index(documents)

    st.success("Documents processed and indexed.")

# Input for user query
if index:
    user_query = st.text_input("Ask a question about your study materials:")
    
    if st.button("Submit"):
        # Retrieve relevant documents and generate answer
        print("inside app")
        
        answer = retrieve_and_answer(user_query, index, documents)
        print(answer)
        st.write("Answer:", answer)
