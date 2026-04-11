import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Load API key securely
GOOGLE_API_KEY = "AIzaSyA8Lr5QvJ-Mh_ljhC9KbRAqUAr-_GA0zSU"

st.header("📄 Chat with your PDF (Gemini Powered)")

# Sidebar
with st.sidebar:
    st.title("Upload Document")
    file = st.file_uploader("Upload a PDF", type="pdf")

if file is not None:
    # Extract text
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = text_splitter.split_text(text)

    # Embeddings (Gemini)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )

    # Vector Store
    vector_store = FAISS.from_texts(chunks, embeddings)

    # Convert to retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # LLM (Gemini Flash)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0
    )

    # RetrievalQA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    # User Input
    user_question = st.text_input("Type your question here")

    if user_question:
        response = qa_chain({"query": user_question})

        st.subheader("Answer:")
        st.write(response["result"])

        # showing sources
        with st.expander("Source Chunks"):
            for doc in response["source_documents"]:
                st.write(doc.page_content[:500])
                st.write("---")
