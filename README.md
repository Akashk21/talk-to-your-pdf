# 🧠 Talk to Your PDF (Demo Vide: [https://drive.google.com/file/d/1zgZxRMMzoO-O1uRH5IPPSa6g5A44Giwt/view?usp=sharing](url))

A lightweight, privacy-friendly Streamlit app that lets you **chat with any PDF document** using local LLMs powered by **Ollama**, **FAISS** vector search, and **HuggingFace embeddings**.

> Upload a PDF. Ask questions. Get accurate, grounded answers — all locally.

---

## 📸 App Preview

<img src="https://github.com/user-attachments/assets/d91e6ef2-1d48-4f0d-a1da-f116f7ed1320" alt="App Screenshot" width="700" />

---

## 🚀 Features

- 📂 Upload one or more PDF files
- 🔍 Intelligent chunking and semantic indexing using FAISS
- 🧠 Local LLMs (e.g. `mistral:7b-instruct`, `gemma:2b-instruct`) via Ollama
- 🔗 HuggingFace `all-MiniLM-L6-v2` for embeddings
- 💬 Natural language Q&A over PDF content
- ✅ Fast, accurate, and completely offline

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/)
- [LangChain](https://www.langchain.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Ollama](https://ollama.com/)
- [HuggingFace Embeddings](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [PyPDF2](https://pypi.org/project/PyPDF2/)

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/talk-to-your-pdf.git
cd talk-to-your-pdf

```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Pull a Model with Ollama
```bash
ollama pull mistral:7b-instruct
```
You can replace mistral:7b-instruct with other supported models like gemma:2b-instruct.

### 4. Run the App
```bash
streamlit run app.py
```

### 📁 Project Structure
```bash
talk-to-your-pdf/
├── app.py
├── faiss_index/
├── requirements.txt
└── README.md
```

### 🙌 Acknowledgements

🤗 HuggingFace for embedding models
🔗 LangChain for prompt orchestration
🐙 Ollama for running LLMs locally
🎈 Streamlit for the interactive UI

### ⭐️ Star this repo if it helped you!

It motivates me to keep building more cool things like this! 😊
