"""
app.py - Streamlit frontend for the Enterprise AI Knowledge Assistant.
Run with:  streamlit run app.py
"""
import streamlit as st

from rag import build_vector_store, answer_question


# ---------- Page setup ----------
st.set_page_config(
    page_title="Enterprise AI Knowledge Assistant",
    page_icon="🏢",
    layout="wide",
)
st.title("🏢 Enterprise AI Knowledge Assistant")
st.caption("Upload your company documents and ask questions across all of them.")


# ---------- Session state ----------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "messages" not in st.session_state:
    st.session_state.messages = []


def show_sources(sources):
    with st.expander("📚 Sources"):
        for s in sources:
            st.markdown(f"**{s['name']}**")
            st.caption(s["text"] + "...")


# ---------- Sidebar: upload ----------
with st.sidebar:
    st.header("📁 Company Documents")

    files = st.file_uploader(
        "Upload PDF, DOCX or TXT files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

    if st.button("Process Documents", type="primary", use_container_width=True):
        if not files:
            st.warning("Please upload at least one document.")
        else:
            with st.spinner("Reading and indexing documents..."):
                try:
                    payload = [(f.name, f.getvalue()) for f in files]
                    st.session_state.vector_store, n_chunks = build_vector_store(payload)
                    st.session_state.messages = []
                    st.success(f"Indexed {len(files)} document(s) • {n_chunks} chunks")
                except Exception as e:
                    st.error(f"Could not process documents: {e}")

    if st.session_state.vector_store is not None:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


# ---------- Main chat area ----------
if st.session_state.vector_store is None:
    st.info("👈 Upload your documents in the sidebar and click **Process Documents** to start.")
else:
    # Show previous messages
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("sources"):
                show_sources(m["sources"])

    # New question
    question = st.chat_input("Ask a question about your documents...")
    if question:
        history = list(st.session_state.messages)
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                try:
                    answer, sources = answer_question(
                        question, st.session_state.vector_store, history
                    )
                except Exception as e:
                    answer, sources = f"Error: {e}", []
            st.markdown(answer)
            if sources:
                show_sources(sources)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )