
import os
import streamlit as st
import hashlib # creates a hash of the file to prevent reprocessing the same file when uploaded multiple times
import time
from uuid import uuid4
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS # faiss vector database for similarity search
from langchain_classic.chains import RetrievalQA # RAG chain that combines retrieval and LLM for question answering

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Gemini PDF Q&A", layout="wide")
st.header("Chat with your PDF (Gemini✨)")

# Load .env from the project directory so Streamlit can read secrets consistently.
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "qna-rag-agent")

if not GOOGLE_API_KEY:
    st.error("Missing GOOGLE_API_KEY. Add it in .env as GOOGLE_API_KEY=your_key")
    st.stop()


def setup_langsmith():
    """Enable LangSmith tracing when API key is available."""
    if not LANGSMITH_API_KEY:
        return False

    os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_PROJECT"] = LANGSMITH_PROJECT
    return True


LANGSMITH_ENABLED = setup_langsmith()

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("Upload Document")
    file = st.file_uploader("Upload a PDF", type="pdf")
    if LANGSMITH_ENABLED:
        st.caption(f"LangSmith tracing: ON ({LANGSMITH_PROJECT})")
    else:
        st.caption("LangSmith tracing: OFF (set LANGSMITH_API_KEY)")

# ---------------- HELPERS ----------------

def get_file_hash(file):
    return hashlib.md5(file.getvalue()).hexdigest()


def extract_text(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text


def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=100
    )
    return splitter.split_text(text)


@st.cache_resource
def load_embeddings(api_key):
    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=api_key
    )


def create_vectorstore_with_retry(chunks, embeddings):  # creates faiss vector store with retry logic to handle rate limits from the API when creating embeddings for a large number of chunks
    """Handles rate limit by batching"""
    batch_size = 20
    all_docs = [] # store all FAISS indexes created for each batch

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]

        for attempt in range(3):
            try:
                vs = FAISS.from_texts(batch, embeddings)
                all_docs.append(vs)
                break
            except Exception as e:
                if "429" in str(e):
                    time.sleep(10)  # wait for quota reset
                else:
                    raise e

    # Merge all FAISS indexes
    base = all_docs[0] # taking the first FAISS index as the base
    for vs in all_docs[1:]:
        base.merge_from(vs) # mergeing other batch indexes into the base index to create one final vector store.

    return base


def load_or_create_faiss(file_hash, chunks, embeddings):
    path = f"faiss_{file_hash}"

    if os.path.exists(path): # Checks if FAISS index already exists on disk for this file hash
        return FAISS.load_local(
            path,
            embeddings,
            allow_dangerous_deserialization=True
        )

    vector_store = create_vectorstore_with_retry(chunks, embeddings)
    vector_store.save_local(path)
    return vector_store


@st.cache_resource # catching llm to avoid reinitialization 
def load_llm(api_key):
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0
    )

# ---------------- MAIN ----------------

if file is not None:

    file_hash = get_file_hash(file)

    # Avoid recomputation
    if "current_file" not in st.session_state or st.session_state.current_file != file_hash: # prevent reprocessing same file

        st.session_state.current_file = file_hash

        with st.spinner("Processing PDF"):
            text = extract_text(file)
            chunks = split_text(text)
            embeddings = load_embeddings(GOOGLE_API_KEY)

            vector_store = load_or_create_faiss(file_hash,chunks,embeddings)
            st.session_state.vector_store = vector_store

    retriever = st.session_state.vector_store.as_retriever(   # Converts vector store into retriever that returns top 10 relevant chunks
        search_kwargs={"k": 10}
    )

    llm = load_llm(GOOGLE_API_KEY)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    query = st.text_input("Ask something about your PDF:")

    if query:
        with st.spinner("Thinking..."):
            response = qa_chain.invoke(
                {"query": query},
                config={
                    "run_name": "pdf_qna",
                    "tags": ["streamlit", "rag", "pdf"],
                    "metadata": {
                        "file_hash": file_hash,
                        "query_id": str(uuid4())
                    }
                }
            )

        st.subheader("Answer:")
        st.write(response["result"])

        with st.expander("📚 Sources"):
            for doc in response["source_documents"]:
                st.write(doc.page_content[:500])
                st.write("---")
