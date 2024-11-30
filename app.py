from flask import Flask, request, render_template, redirect, url_for, flash
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
import ollama
import os
from PIL import Image
import pytesseract
import tempfile
from pdf2image import convert_from_path

app = Flask(__name__)
app.secret_key = "secretkey"
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configure Tesseract path if needed
# pytesseract.pytesseract.tesseract_cmd = r"Tesseract-OCR Path (if needed)"

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
embeddings = SpacyEmbeddings(model_name="en_core_web_sm")


def pdf_read(pdf_path):
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def perform_ocr(file_path, is_pdf):
    text = ""
    if is_pdf:
        # Convert PDF pages to images using pdf2image
        try:
            images = convert_from_path(file_path, dpi=300)
            for image in images:
                text += pytesseract.image_to_string(image)
        except Exception as e:
            raise RuntimeError(f"Failed to process PDF file: {e}")
    else:
        # For image files
        try:
            image = Image.open(file_path)
            text += pytesseract.image_to_string(image)
        except Exception as e:
            raise RuntimeError(f"Failed to process image file: {e}")
    return text


def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text)


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

    response_text = ""
    for chunk in response_stream:
        response_text += chunk["message"]["content"]
    return response_text


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        is_handwritten = request.form.get("is_handwritten") == "on"
        uploaded_files = request.files.getlist("files")

        if not uploaded_files:
            flash("No files uploaded!", "danger")
            return redirect(url_for("index"))

        raw_text = ""
        for file in uploaded_files:
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)

            try:
                if is_handwritten or file.filename.endswith(".pdf"):
                    raw_text += perform_ocr(file_path, file.filename.endswith(".pdf"))
                else:
                    raw_text += pdf_read(file_path)
            except Exception as e:
                flash(f"Error processing file {file.filename}: {e}", "danger")
                return redirect(url_for("index"))

        text_chunks = get_chunks(raw_text)
        vector_store(text_chunks)
        flash("Processing completed!", "success")
        return redirect(url_for("chat"))

    return render_template("index.html")


@app.route("/chat", methods=["GET", "POST"])
def chat():
    if request.method == "POST":
        user_question = request.form.get("user_question")
        if not user_question:
            flash("Please enter a question!", "danger")
            return redirect(url_for("chat"))

        new_db = FAISS.load_local(
            "faiss_db", embeddings, allow_dangerous_deserialization=True
        )
        retriever = new_db.as_retriever()

        retrieval_chain = create_retriever_tool(
            retriever,
            "pdf_extractor",
            "This tool is to give answer to queries from the pdf",
        )

        response = get_conversational_chain(retrieval_chain, user_question)
        return render_template("result.html", question=user_question, response=response)

    return render_template("chat.html")


if __name__ == "__main__":
    app.run(debug=True)
