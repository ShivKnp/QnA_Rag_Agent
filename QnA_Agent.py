import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings

# ---------------- CONFIG ----------------
st.set_page_config(page_title="PDF Q&A (RAG)", layout="wide")
st.header("📄 Chat with your PDF (Optimized RAG 🚀)")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("Missing GOOGLE_API_KEY")
    st.stop()

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("Upload Document")
    file = st.file_uploader("Upload a PDF", type="pdf")

# ---------------- FUNCTIONS ----------------

def extract_text(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text


def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,     # reduced chunks
        chunk_overlap=100
    )
    return text_splitter.split_text(text)


@st.cache_resource
def load_embeddings():
    # FREE local embeddings (no API cost 🚀)
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )


def create_or_load_vectorstore(chunks, embeddings):
    if os.path.exists("faiss_index"):
        return FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        vector_store = FAISS.from_texts(chunks, embeddings)
        vector_store.save_local("faiss_index")
        return vector_store


@st.cache_resource
def load_llm(api_key):
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0
    )

# ---------------- MAIN LOGIC ----------------

if file is not None:

    # Extract text
    text = extract_text(file)

    # Split text
    chunks = split_text(text)

    # Load embeddings
    embeddings = load_embeddings()

    # Create or load FAISS
    vector_store = create_or_load_vectorstore(chunks, embeddings)

    # Store in session to avoid rerun recompute
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = vector_store

    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})

    # Load LLM
    llm = load_llm(GOOGLE_API_KEY)

    # QA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    # User Input
    user_question = st.text_input("Ask a question from your PDF:")

    if user_question:
        with st.spinner("Thinking... 🤔"):
            response = qa_chain({"query": user_question})

        st.subheader("Answer:")
        st.write(response["result"])

        # Sources
        with st.expander("📚 Source Chunks"):
            for doc in response["source_documents"]:
                st.write(doc.page_content[:500])
                st.write("---")
