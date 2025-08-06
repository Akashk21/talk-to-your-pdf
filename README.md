# 🧠 Talk to Your PDF

A lightweight, privacy-friendly Streamlit app that lets you **chat with any PDF document** using local LLMs powered by **Ollama**, **FAISS** vector search, and **HuggingFace embeddings**.

> Upload a PDF. Ask questions. Get accurate, grounded answers — all locally.

---

## 📸 App Preview

<!-- Replace below base64 src with your actual image if needed -->
<img src="[data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAA...](https://drive.google.com/file/d/1rXcoBeCP9N-eMOoE0yxa_1r4-WzipRHy/view?usp=drive_link)" alt="App Screenshot" width="700"/>

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
