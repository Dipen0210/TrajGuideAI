# RAG System

## Document Sources
- Place curated documents (traffic regulations, safety guides, weather notes, etc.) under `agent/rag/processed_docs/`.
- Supported formats: `.txt`, `.md`, `.csv`, `.pdf`. PDFs are parsed with `PyPDF2`.

## Text Splitting
- `build_vectorstore.py` loads all documents, converts them into `Document` objects, and splits them with `RecursiveCharacterTextSplitter` (`chunk_size=1000`, `chunk_overlap=200`).
- This strategy balances context richness and embedding efficiency.

## Embeddings
- Uses `HuggingFaceEmbeddings` with `sentence-transformers/all-MiniLM-L6-v2`, a lightweight yet performant bi-encoder suitable for local inference.

## Vector Store
- Chroma is instantiated with `persist_directory="agent/rag/chroma_db"`.
- `Chroma.from_documents(...)` writes both the embeddings and metadata to disk for reuse across sessions.

## Retrieval + Generation
- `rag_chain.py` loads the persisted store, exposes a retriever with `k=3`, and constructs a `RetrievalQA` chain.
- `rag_query(question)` returns:
  - `answer`: Llama 3-generated response using retrieved chunks.
  - `sources`: metadata (including file names) for transparency.
- Streamlit or backend callers can surface these sources so users know which references were consulted.
