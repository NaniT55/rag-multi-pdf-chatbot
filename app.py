import streamlit as st
import os
import shutil

from rag_pipeline import load_pdf, split_text, create_vector_store, create_qa_chain

st.set_page_config(page_title="RAG Chatbot", layout="wide")

# 🎨 Simple clean UI
st.title("🤖 Multi-PDF RAG Chatbot")
st.caption("Upload PDFs and ask questions")

# 🧠 Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "uploaded_names" not in st.session_state:
    st.session_state.uploaded_names = []

# 📂 Sidebar
with st.sidebar:
    st.header("Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDFs", type="pdf", accept_multiple_files=True
    )

    current_files = [file.name for file in uploaded_files] if uploaded_files else []

    # ✅ Process only when files change
    if uploaded_files and current_files != st.session_state.uploaded_names:

        with st.spinner("Processing PDFs..."):

            # Clear old index
            if os.path.exists("chroma_db"):
                shutil.rmtree("chroma_db")

            all_docs = []

            for file in uploaded_files:
                path = f"temp_{file.name}"

                with open(path, "wb") as f:
                    f.write(file.read())

                docs = load_pdf(path)

                for doc in docs:
                    doc.metadata["source"] = file.name

                all_docs.extend(docs)

            chunks = split_text(all_docs)
            vectorstore = create_vector_store(chunks)

            st.session_state.qa_chain = create_qa_chain(vectorstore)
            st.session_state.uploaded_names = current_files

        st.success("✅ PDFs processed!")

    if st.button("Clear Chat"):
        st.session_state.messages = []

# 💬 Chat display
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

# 📥 Input
prompt = st.chat_input("Ask a question...")

if prompt and st.session_state.qa_chain:

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        result = st.session_state.qa_chain({"query": prompt})
        answer = result["result"]

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()