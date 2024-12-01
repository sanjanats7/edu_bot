import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
import ollama
import time
import os
from PIL import Image
import pytesseract
import tempfile
from pdf2image import convert_from_path
import re
from database import register_user, login_user, save_query, get_user_history, clear_all_history


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

embeddings = SpacyEmbeddings(model_name="en_core_web_sm")

is_elaborate = False


def pdf_read(pdf_doc):
    text = ""
    for pdf in pdf_doc:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def perform_ocr(files):
    text = ""
    for file in files:
        if file.name.endswith(".pdf"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                temp_pdf.write(file.read())
                temp_pdf_path = temp_pdf.name

            try:
                images = convert_from_path(temp_pdf_path, dpi=300)
                for image in images:
                    text += pytesseract.image_to_string(image)
            except Exception as e:
                st.error(f"Failed to process PDF file: {e}")
        else:
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


def get_conversational_chain(retrieval_chain, user_question, top_k, top_p, temperature):
    retrieved_context = retrieval_chain.run(user_question)

    pre_prompt = (
        "You are a helpful assistant providing answers from a document context. "
    )
    global is_elaborate
    if is_elaborate:
        pre_prompt += "Give a detailed answer in points."
    else:
        pre_prompt += "Give a concise answer in points"

    response_stream = ollama.chat(
        model="llama3.1",
        messages=[
            {
                "role": "system",
                "content": pre_prompt,
            },
            {"role": "user", "content": f"Context: {retrieved_context}"},
            {"role": "user", "content": user_question},
        ],
        stream=True,
        options={
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "num_predict": -1,
        },
    )

    response_container = st.empty()
    response_text = ""

    for chunk in response_stream:
        response_text += chunk["message"]["content"]
        response_container.write("Reply: " + response_text)
        time.sleep(0.05)

    response_container.write("Reply: " + response_text)
    return response_text


def user_input(user_question, top_k, top_p, temperature):
    new_db = FAISS.load_local(
        "faiss_db", embeddings, allow_dangerous_deserialization=True
    )
    retriever = new_db.as_retriever()

    retrieval_chain = create_retriever_tool(
        retriever,
        "pdf_extractor",
        "This tool is to give answer to queries from the pdf",
    )

    return get_conversational_chain(retrieval_chain, user_question, top_k, top_p, temperature)


def extract_questions(question_bank_files):
    question_bank_text = pdf_read(question_bank_files)
    start_match = re.search(r"\b1\b", question_bank_text)
    if not start_match:
        return [] 
    text_from_first_1 = question_bank_text[start_match.start() :]
    question_pattern = re.compile(r"(\d+\s.*?)(?=\n\d+\s|\n1 \s|$)", re.DOTALL)
    matches = question_pattern.findall(text_from_first_1)
    return [q.strip() for q in matches]

def clear_text_input(key):
    st.session_state[key] = ""

def main():
    st.set_page_config("Chat PDF", layout="wide")
    st.header("RAG based Chat with PDF")

    creativity = {}
    creativity[0] = (0, 1, 0)
    creativity[1] = (50, 0.5, 0.5) # default value for creativity
    creativity[2] = (100, 0, 1)
    
    menu = ["Login", "Register", "Chat"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Register":
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Register"):
            result = register_user(username, password)
            st.success(result)

    elif choice == "Login":
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if login_user(username, password):
                st.session_state["username"] = username
                st.success(f"Welcome {username}!")
            else:
                st.error("Invalid credentials.")

    
    elif choice == "Chat":
        if "username" in st.session_state:
            
            with st.sidebar:
                st.subheader(f"Welcome, {st.session_state['username']}!")
                st.title("Settings:")
                
                is_handwritten = st.checkbox("Handwritten notes?")
                
                global is_elaborate
                is_elaborate = st.checkbox("Detailed answer?")
                
                pdf_doc = st.file_uploader(
                    "Upload your PDF/Image Files and Click on the Submit & Process Button",
                    accept_multiple_files=True,
                )
                
                question_bank = st.file_uploader(
                    "Upload Question Bank (PDF/Text)", accept_multiple_files=False
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
            
            if question_bank:
                questions = extract_questions([question_bank])
                st.session_state["questions"] = questions
                st.success("Questions extracted successfully!")
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
            cret = st.sidebar.slider("creativity values", min_value = 0, max_value = 2, value = 1)
            top_k = creativity[cret][0]
            top_p = creativity[cret][1]
            temperature = creativity[cret][2]                       

            col1, col2 = st.columns([4, 1])
            
            with col1:
                # Display chat history
                st.subheader("Chat History")
            with col2:
                if st.button("Clear history"):
                    if "user_input" in st.session_state:
                        clear_text_input("user_input")
                    clear_all_history(st.session_state["username"])
                    st.rerun()
                
            chat_history = get_user_history(st.session_state["username"])
                                
            if chat_history:
                for entry in chat_history:
                    c = st.container(border=True)
                    c.write(f"*Q:* {entry['question']}")
                    c.write(f"*A:* {entry['answer']}")
            else:
                st.info("No chat history available.")
                
            st.subheader("Ask Questions")
            questions = st.session_state.get("questions", [])
            if questions:
                selected_question = st.selectbox("Select a question:", [""] + questions)
                if selected_question:
                    response = user_input(selected_question, 50, 0.5, 0.7)
                    save_query(st.session_state["username"], selected_question, response)

            user_question = st.text_input("Ask your own question", key="user_input")
            if user_question:
                response = user_input(user_question, 50, 0.5, 0.7)
                save_query(st.session_state["username"], user_question, response)
                st.write(f"*Q:* {user_question}")
                st.write(f"*A:* {response}")
            
    else:
        st.error("Please log in to use the chat application.")             


if __name__ == "__main__":
    main()
