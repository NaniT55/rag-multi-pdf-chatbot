import streamlit as st
import os
import shutil

from rag_pipeline import load_pdf, split_text, create_vector_store, create_qa_chain

st.set_page_config(page_title="RAG Chatbot", layout="wide")

# 🎨 CSS
st.markdown("""
<style>
body { background-color: #0F172A; color: #E5E7EB; }

.chat-container { max-height: 70vh; overflow-y: auto; padding: 10px; }

.user-msg {
    background-color: #2563EB;
    color: #FFFFFF;
    padding: 12px;
    border-radius: 12px;
    margin-bottom: 10px;
    text-align: right;
    max-width: 75%;
    margin-left: auto;
}

.bot-msg {
    background-color: #1F2937;
    color: #E5E7EB;
    padding: 12px;
    border-radius: 12px;
    margin-bottom: 10px;
    text-align: left;
    max-width: 75%;
}

.source-box {
    background-color: #111827;
    color: #9CA3AF;
    padding: 10px;
    border-left: 3px solid #2563EB;
    border-radius: 8px;
    margin-top: 8px;
    font-size: 0.9em;
}
</style>
""", unsafe_allow_html=True)

# 🧠 Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "uploaded_names" not in st.session_state:
    st.session_state.uploaded_names = []

# 📂 Sidebar
with st.sidebar:
    st.title("⚙️ Settings")

    uploaded_files = st.file_uploader(
        "Upload PDFs", type="pdf", accept_multiple_files=True
    )

    # 🔥 Detect file changes
    current_files = [file.name for file in uploaded_files] if uploaded_files else []

    if uploaded_files and current_files != st.session_state.uploaded_names:

        with st.spinner("Processing PDFs..."):

            # 🧹 Clear old index
            if os.path.exists("faiss_index"):
                shutil.rmtree("faiss_index")

            all_docs = []

            for uploaded_file in uploaded_files:
                file_path = f"temp_{uploaded_file.name}"

                with open(file_path, "wb") as f:
                    f.write(uploaded_file.read())

                docs = load_pdf(file_path)

                # add source metadata
                for doc in docs:
                    doc.metadata["source"] = uploaded_file.name

                all_docs.extend(docs)

            chunks = split_text(all_docs)
            vectorstore = create_vector_store(chunks)

            st.session_state.qa_chain = create_qa_chain(vectorstore)

            # 🔥 save uploaded file names
            st.session_state.uploaded_names = current_files

        st.success("✅ PDFs processed successfully!")

    # 🧹 Clear chat
    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []

    # 🔄 Reload documents
    if st.button("🔄 Reload Documents"):
        st.session_state.qa_chain = None
        st.session_state.messages = []
        st.session_state.uploaded_names = []

    # 🗑️ Clear saved embeddings
    if st.button("🗑️ Clear Saved Index"):
        if os.path.exists("faiss_index"):
            shutil.rmtree("faiss_index")

        st.session_state.qa_chain = None
        st.session_state.messages = []
        st.session_state.uploaded_names = []
        st.success("Index cleared!")

# 🧠 Helpers
def is_valid_answer(ans):
    ans = ans.lower()
    return not any(x in ans for x in ["i don't know", "not available", "no information"])

def filter_sources(sources, query):
    query_words = set(query.lower().split())
    filtered = []

    for doc in sources:
        content = doc.page_content.lower()
        match_count = sum(1 for word in query_words if word in content)

        if match_count >= 2:
            filtered.append((match_count, doc))

    filtered.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in filtered[:2]]

# 🧾 Header
st.title("🤖 Multi-PDF RAG Chatbot")
st.caption("Chat across multiple documents 📄")

# 💬 Chat UI
chat_container = st.container()

with chat_container:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="user-msg">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-msg">🤖 {msg["content"]}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# 📥 Input
prompt = st.chat_input("Ask something about your documents...")

if prompt and st.session_state.qa_chain:

    # store user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        result = st.session_state.qa_chain({"query": prompt})
        answer = result["result"].strip()
        sources = result.get("source_documents", [])

    response = answer

    filtered_sources = filter_sources(sources, prompt)

    # show sources only if valid answer
    if is_valid_answer(answer) and filtered_sources:
        response += "\n\n📄 **Sources:**\n"

        for doc in filtered_sources:
            page = doc.metadata.get("page", "N/A")
            source = doc.metadata.get("source", "Unknown file")
            content = doc.page_content[:150] + "..."

            response += f"""
<div class="source-box">
<b>{source} | Page {page}</b><br>
{content}
</div>
"""

    # store assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})

    st.rerun()