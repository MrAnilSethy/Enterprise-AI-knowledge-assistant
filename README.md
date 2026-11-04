# 🏢 Enterprise AI Knowledge Assistant (Advanced RAG)

Upload company documents (PDF, DOCX, TXT) and ask questions across all of them.
Answers are generated only from the documents and include source references.

**Stack:** Python · LangChain · Hugging Face Embeddings · ChromaDB · Groq · Streamlit

## Project Structure
```
├── app.py            # Streamlit frontend
├── rag.py            # RAG backend (load, chunk, embed, search, answer)
├── requirements.txt
├── .env.example
└── README.md
```

## How it works
1. Upload multiple documents in the sidebar
2. Documents are split into chunks (recursive splitter, 1000 / 200 overlap)
3. Chunks are embedded (all-MiniLM-L6-v2) and stored in ChromaDB
4. Your question retrieves the top-5 matching chunks from all documents
5. Groq LLM answers using only that context and shows the sources

## Run locally
```bash
pip install -r requirements.txt
cp .env.example .env        # add your GROQ_API_KEY
streamlit run app.py
```