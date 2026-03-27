import streamlit as st
import os
import tempfile

from rag_pipeline import load_pdf, split_text, create_vector_store, create_qa_chain

st.set_page_config(page_title="RAG Chatbot", layout="wide")

# 🎨 CSS
st.markdown("""
<style>
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

    current_files = [f.name for f in uploaded_files] if uploaded_files else []

    if uploaded_files and current_files != st.session_state.uploaded_names:
        with st.spinner("Processing PDFs..."):
            all_docs = []
            temp_paths = []

            for uploaded_file in uploaded_files:
                # ✅ Use tempfile for safe cross-platform temp handling
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf"
                ) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                    temp_paths.append(tmp_path)

                docs = load_pdf(tmp_path)

                for doc in docs:
                    doc.metadata["source"] = uploaded_file.name

                all_docs.extend(docs)

            chunks = split_text(all_docs)
            vectorstore = create_vector_store(chunks)
            st.session_state.qa_chain = create_qa_chain(vectorstore)
            st.session_state.uploaded_names = current_files

            # ✅ Clean up temp files after processing
            for path in temp_paths:
                try:
                    os.remove(path)
                except OSError:
                    pass

        st.success("✅ PDFs processed successfully!")

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    if st.button("🔄 Reload Documents"):
        st.session_state.qa_chain = None
        st.session_state.messages = []
        st.session_state.uploaded_names = []
        st.rerun()

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


def render_sources(sources):
    """Render source boxes safely using st.markdown per source."""
    st.markdown("📄 **Sources:**")
    for doc in sources:
        page = doc.metadata.get("page", "N/A")
        source = doc.metadata.get("source", "Unknown file")
        content = doc.page_content[:150] + "..."
        st.markdown(
            f'<div class="source-box"><b>{source} | Page {page}</b><br>{content}</div>',
            unsafe_allow_html=True
        )


# 🧾 Header
st.title("🤖 Multi-PDF RAG Chatbot")
st.caption("Chat across multiple documents 📄")

# 💬 Chat history — ✅ use st.chat_message instead of raw HTML bubbles
for msg in st.session_state.messages:
    role = msg["role"]
    with st.chat_message(role):
        st.markdown(msg["content"])
        # Re-render sources if stored
        if role == "assistant" and msg.get("sources"):
            render_sources(msg["sources"])

# ⚠️ Guide user if no PDFs uploaded yet
if not st.session_state.qa_chain:
    st.info("👈 Upload one or more PDFs in the sidebar to get started.")

# 📥 Input
prompt = st.chat_input("Ask something about your documents...")

if prompt:
    if not st.session_state.qa_chain:
        st.warning("Please upload and process at least one PDF first.")
    else:
        # Show user message immediately
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.qa_chain.invoke({"query": prompt})  # ✅ .invoke() not __call__
                answer = result["result"].strip()
                sources = result.get("source_documents", [])

            st.markdown(answer)

            filtered_sources = filter_sources(sources, prompt)
            stored_sources = []

            if is_valid_answer(answer) and filtered_sources:
                render_sources(filtered_sources)
                stored_sources = filtered_sources

        # Store message + sources for re-render on next run
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": stored_sources
        })