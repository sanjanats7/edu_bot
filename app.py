from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
import ollama
import streamlit as st
import time
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

embeddings = SpacyEmbeddings(model_name="en_core_web_sm")


def pdf_read(pdf_doc):
    text = ""
    for pdf in pdf_doc:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks


def vector_store(text_chunks):

    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_db")


def get_conversational_chain(retrieval_chain, user_question):
    # Run the retrieval chain to get contextually relevant information
    retrieved_context = retrieval_chain.run(user_question)

    # Set up the Ollama Llama 3.1 interface with the retrieved context
    response_stream = ollama.chat(
        model="llama3.1",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant providing answers from a document context.",
            },
            {"role": "user", "content": f"Context: {retrieved_context}"},
            {"role": "user", "content": user_question},
        ],
        stream=True,
    )

    # Create an empty container to hold the response
    response_container = st.empty()
    response_text = ""

    # Iterate over the response stream, updating the response text incrementally
    for chunk in response_stream:
        # Append the new chunk of text to the response
        response_text += chunk["message"]["content"]
        response_container.write("Reply: " + response_text)

        # Optional: add a small delay to simulate typing speed
        time.sleep(0.05)

    # Final write in case there’s any delay in receiving chunks
    response_container.write("Reply: " + response_text)


def user_input(user_question):
    # Load FAISS index locally and set up the retriever
    new_db = FAISS.load_local(
        "faiss_db", embeddings, allow_dangerous_deserialization=True
    )
    retriever = new_db.as_retriever()

    # Create a retrieval tool with the locally loaded retriever
    retrieval_chain = create_retriever_tool(
        retriever,
        "pdf_extractor",
        "This tool is to give answer to queries from the pdf",
    )

    # Invoke the conversational chain using the locally configured model
    get_conversational_chain(retrieval_chain, user_question)


def main():
    st.set_page_config("Chat PDF")
    st.header("RAG based Chat with PDF")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_doc = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button",
            accept_multiple_files=True,
        )
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = pdf_read(pdf_doc)
                text_chunks = get_chunks(raw_text)
                vector_store(text_chunks)
                st.success("Done")


if __name__ == "__main__":
    main()
