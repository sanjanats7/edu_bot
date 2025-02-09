# EduBot



## Installation
To get started with EduBot, follow these steps:

1. **Clone the Repository**:
```bash
git clone https://github.com/sanjanats7/edu_bot.git
cd edu_bot
```

2. **Set Up a Virtual Environment**(Optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

4. **Set Up Ollama with Llama 3.1**:
EduBot requires Ollama to run the Llama 3.1 model in the background. Follow these steps to set it up:
- **Install Ollama**:
    - Visit the [Ollama GitHub repository](https://github.com/ollama/ollama) and follow the installtion instructions for your operating system.
- **Download the Llama 3.1 Model**:
```bash
ollama pull llama3.1
```
- **Run Ollama in the Background**:
Start the Ollama server to serve the Llama 3.1 model:
```bash
ollama serve
```
Keep this service running in the backgorund.


5. **Run the Streamlit App**
```bash
streamlit run app.py
```

6. **Access the App**
Open your web browser and navigate to `http://localhost:8501` to interact with EduBot.

