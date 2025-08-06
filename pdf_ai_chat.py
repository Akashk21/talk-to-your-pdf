import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama

# PDF Text Extraction
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Text Chunking
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Embedding + Vector Store
def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# QA Chain using Ollama
def get_conversational_chain():
    prompt_template = """
    Answer the question based on the context below. Keep the answer concise.
    If you don't know the answer, just say you don't know.

    Context: {context}

    Question: {question}

    Answer:"""

    llm = Ollama(
        # model="mistral:7b-instruct", 
        model="gemma:2b-instruct",
        temperature=0.3,
        top_p=0.9,
        num_ctx=2048
    )


    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    return chain

# Input User Handling
def user_input(user_question):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )
        st.success("✅ Answer Generated:")
        st.markdown(f"**🧠 {response['output_text']}**")
    except Exception as e:
        st.error(f"❌ Error processing your question: {str(e)}")

# ---- Main App ----
def main():

    st.set_page_config(
        page_title="📚 Pdf-Pro",
        layout="centered",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
        <style>
        /* ---------- GLOBAL BACKGROUND ---------- */
        .main {
            background-color: #0e1117;
            color: #ffffff;
        }

        /* ---------- SIDEBAR ---------- */
        section[data-testid="stSidebar"] {
            background-color: #161b22;
        }

        /* ---------- HEADINGS ---------- */
        h1, h2, h3, h4, h5, h6 {
            color: #58a6ff !important;
        }

        /* ---------- FILE UPLOAD & BUTTON ---------- */
        .stFileUploader label {
            color: #c9d1d9 !important;
        }

        .stButton > button {
            background-color: #238636;
            color: white;
            font-weight: bold;
            border-radius: 10px;
            padding: 0.6rem 1.2rem;
            transition: 0.3s ease;
        }

        .stButton > button:hover {
            background-color: #2ea043;
        }

        /* ---------- TEXT INPUT AREA ---------- */
        textarea, .stTextInput input {
            background-color: #1c2128;
            color: #ffffff !important;
            border: 1px solid #30363d;
            border-radius: 10px;
            font-size: 16px;
            padding: 0.75rem;
        }

        /* ---------- PLACEHOLDER TEXT ---------- */
        ::placeholder {
            color: #8b949e !important;
            opacity: 1;
        }

        /* ---------- SUCCESS / ERROR / SPINNER ---------- */
        .stAlert {
            border-radius: 8px;
        }

        /* ---------- MARKDOWN RESPONSE ---------- */
        .markdown-text-container {
            color: #d1d5da;
            font-size: 1rem;
        }

        /* ---------- SCROLLBAR ---------- */
        ::-webkit-scrollbar {
            width: 10px;
        }

        ::-webkit-scrollbar-thumb {
            background-color: #30363d;
            border-radius: 5px;
        }

        ::-webkit-scrollbar-track {
            background-color: #161b22;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("📚 Talk to Your PDFs with AI")
    st.markdown("Turn any PDF into a smart assistant. Ask anything — get ground truth answers instantly.")


    with st.container():
        st.subheader("❓ Ask a Question")
        user_question = st.text_area("Type your question below:", height=100, placeholder="E.g. What is the summary of the second chapter?")
        if user_question:
            with st.spinner("💬 Generating answer..."):
                user_input(user_question)

    
    with st.sidebar:
        st.header("📂 Upload PDF")
        pdf_docs = st.file_uploader("Upload one or more PDF files", accept_multiple_files=True, type=["pdf"])
        
        if st.button("🚀 Submit & Process PDFs"):
            if not pdf_docs:
                st.warning("⚠️ Please upload at least one PDF.")
            else:
                with st.spinner("🔍 Reading & indexing your PDF..."):
                    try:
                        raw_text = get_pdf_text(pdf_docs)
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("✅ PDF successfully processed and indexed!")
                    except Exception as e:
                        st.error(f"❌ Failed to process PDFs: {str(e)}")

if __name__ == "__main__":
    main()
