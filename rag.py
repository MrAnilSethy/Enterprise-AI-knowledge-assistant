"""
rag.py - RAG backend for the Enterprise AI Knowledge Assistant.
Handles: loading documents, chunking, embeddings, ChromaDB search, and answering.
"""
from dotenv import load_dotenv
load_dotenv()

import os
import tempfile

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------- Config ----------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-120b"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 8        # chunks sent to the LLM
FETCH_K = 24     # candidates considered before MMR picks the best TOP_K

SYSTEM_PROMPT = """You are an Enterprise Knowledge Assistant.
Answer the user's question using the context below, which was retrieved from the uploaded company documents.

Rules:
1. Base your answer only on the context. Do not use outside knowledge.
2. If the context contains the answer, or even part of it, give it.
   For partial matches, answer with what is available and say what is missing.
3. Reason over the context: summarize, combine facts from several chunks, and interpret numbers.
   The answer does not need to match the question's wording exactly.
4. Only if the context has nothing related to the question, say:
   "I could not find this information in the documents."
5. Keep the answer clear and simple, and mention the document name (and page) you used.

Context:
{context}
"""


# ---------- Models ----------
_embeddings = None
_llm = None


def get_embeddings():
    """Hugging Face embeddings by default. Set EMBEDDING_PROVIDER=mistral in .env to use Mistral."""
    global _embeddings
    if _embeddings is None:
        if os.getenv("EMBEDDING_PROVIDER", "huggingface").lower() == "mistral":
            from langchain_mistralai import MistralAIEmbeddings
            _embeddings = MistralAIEmbeddings(model="mistral-embed")
        else:
            from langchain_huggingface import HuggingFaceEmbeddings
            _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


def get_llm():
    global _llm
    if _llm is None:
        if not os.getenv("GROQ_API_KEY"):
            raise RuntimeError("GROQ_API_KEY not found. Add it to your .env file.")
        from langchain_groq import ChatGroq
        _llm = ChatGroq(model=LLM_MODEL, temperature=0)
    return _llm


# ---------- Ingestion ----------
def load_file(file_name: str, file_bytes: bytes):
    """Load one uploaded file (pdf / docx / txt) into LangChain documents."""
    suffix = os.path.splitext(file_name)[1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        path = tmp.name

    try:
        if suffix == ".pdf":
            docs = PyPDFLoader(path).load()
        elif suffix == ".docx":
            docs = Docx2txtLoader(path).load()
        else:
            docs = TextLoader(path, encoding="utf-8").load()
    finally:
        os.remove(path)

    # Keep the real file name as the source (not the temp path)
    for d in docs:
        d.metadata["source"] = file_name
    return docs


def build_vector_store(files):
    """
    files: list of (file_name, file_bytes)
    Returns: (vector_store, total_chunks)
    """
    all_docs = []
    for name, data in files:
        all_docs.extend(load_file(name, data))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(all_docs)

    from langchain_chroma import Chroma
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
    )
    return vector_store, len(chunks)


# ---------- Question Answering ----------
def format_source(doc) -> str:
    name = doc.metadata.get("source", "Unknown")
    if "page" in doc.metadata:
        return f"{name} (page {doc.metadata['page'] + 1})"
    return name


def rewrite_question(question: str, chat_history) -> str:
    """Turn a follow-up like 'what about that?' into a standalone search query."""
    if not chat_history:
        return question

    history_text = "\n".join(
        f"{m['role']}: {m['content'][:300]}" for m in chat_history[-4:]
    )
    prompt = (
        "Rewrite the last question as a single standalone search query, "
        "using the chat history to resolve words like 'it', 'that', 'they'. "
        "If it is already standalone, return it unchanged. "
        "Return ONLY the query.\n\n"
        f"Chat history:\n{history_text}\n\nLast question: {question}\n\nQuery:"
    )
    try:
        rewritten = get_llm().invoke(prompt).content.strip()
        return rewritten or question
    except Exception:
        return question


def answer_question(question: str, vector_store, chat_history=None):
    """
    Retrieve relevant chunks from ALL uploaded documents and generate an answer.
    Returns: (answer_text, list_of_sources)
    """
    # 1. Make the query standalone (fixes vague follow-up questions)
    search_query = rewrite_question(question, chat_history)

    # 2. MMR retrieval: relevant AND diverse chunks across documents
    results = vector_store.max_marginal_relevance_search(
        search_query, k=TOP_K, fetch_k=FETCH_K
    )

    context = "\n\n".join(
        f"[Source: {format_source(d)}]\n{d.page_content}" for d in results
    )

    messages = [("system", SYSTEM_PROMPT.format(context=context))]

    # Short chat memory (last 4 messages)
    for m in (chat_history or [])[-4:]:
        messages.append((m["role"], m["content"]))

    messages.append(("user", question))

    answer = get_llm().invoke(messages).content

    sources = [
        {"name": format_source(d), "text": d.page_content[:250]} for d in results
    ]
    return answer, sources