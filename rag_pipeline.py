from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import streamlit as st


# 📄 Load PDF
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()


# ✂️ Split text
def split_text(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=150
    )
    return splitter.split_documents(documents)


# 🧠 Create vector store (no auto-load here)
def create_vector_store(chunks):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Save index (for performance)
    vectorstore.save_local("faiss_index")

    return vectorstore


# 🤖 Create QA chain
def create_qa_chain(vectorstore):

    llm = ChatOpenAI(
        openai_api_key=st.secrets["OPENAI_API_KEY"],  # ✅ FIXED
        openai_api_base="https://api.groq.com/openai/v1",
        model="llama-3.1-8b-instant",
        temperature=0
    )

    prompt_template = """You are a helpful assistant.

Use ONLY the context below to answer the question.
Do NOT use external knowledge.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}

Answer:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain