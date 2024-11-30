from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
import ollama
import streamlit as st
import time
import os
from PIL import Image
import pytesseract

# Configure Tesseract path if required
# pytesseract.pytesseract.tesseract_cmd = r"Tesseract-OCR Path (if needed)"

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

embeddings = SpacyEmbeddings(model_name="en_core_web_sm")


def pdf_read(pdf_doc):
    text = ""
    for pdf in pdf_doc:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

import tempfile
from pdf2image import convert_from_path
from PIL import Image
import pytesseract


def perform_ocr(files):
    """
    Perform OCR on uploaded files (images or PDFs).
    """
    text = ""
    for file in files:
        if file.name.endswith(".pdf"):
            # Save the PDF to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                temp_pdf.write(file.read())  # Write the contents of the UploadedFile
                temp_pdf_path = temp_pdf.name

            # Convert PDF pages to images using pdf2image
            try:
                images = convert_from_path(temp_pdf_path, dpi=300)
                for image in images:
                    text += pytesseract.image_to_string(image)
            except Exception as e:
                st.error(f"Failed to process PDF file: {e}")
        else:
            # For image files
            try:
                image = Image.open(file)
                text += pytesseract.image_to_string(image)
            except Exception as e:
                st.error(f"Failed to process image file: {e}")
    return text


def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks


def vector_store(text_chunks):
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_db")


def get_conversational_chain(retrieval_chain, user_question):
    retrieved_context = retrieval_chain.run(user_question)

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

    response_container = st.empty()
    response_text = ""

    for chunk in response_stream:
        response_text += chunk["message"]["content"]
        response_container.write("Reply: " + response_text)
        time.sleep(0.05)

    response_container.write("Reply: " + response_text)


def user_input(user_question):
    new_db = FAISS.load_local(
        "faiss_db", embeddings, allow_dangerous_deserialization=True
    )
    retriever = new_db.as_retriever()

    retrieval_chain = create_retriever_tool(
        retriever,
        "pdf_extractor",
        "This tool is to give answer to queries from the pdf",
    )

    get_conversational_chain(retrieval_chain, user_question)


def main():
    st.set_page_config("Chat PDF")
    st.header("RAG based Chat with PDF")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        is_handwritten = st.checkbox("Is this a handwritten document?")
        pdf_doc = st.file_uploader(
            "Upload your PDF/Image Files and Click on the Submit & Process Button",
            accept_multiple_files=True,
        )
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                if is_handwritten:
                    st.info("Performing OCR on handwritten notes...")
                    raw_text = perform_ocr(pdf_doc)
                else:
                    st.info("Reading text from PDFs...")
                    raw_text = pdf_read(pdf_doc)

                text_chunks = get_chunks(raw_text)
                vector_store(text_chunks)
                st.success("Processing Completed!")


if __name__ == "__main__":
    main()
