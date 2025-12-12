"""
Utility script to build and persist a Chroma vector store for RAG.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from PyPDF2 import PdfReader


PROCESSED_DOCS_DIR = Path(__file__).resolve().parent / "processed_docs"
KNOWLEDGE_BASE_DIR = Path(__file__).resolve().parent / "knowledge_base"
PERSIST_DIR = Path(__file__).resolve().parent / "chroma_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _load_text_from_file(path: Path) -> str:
    """
    Read supported document types into raw text.
    """
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".csv"}:
        return path.read_text(encoding="utf-8")

    if suffix == ".pdf":
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)

    raise ValueError(f"Unsupported document type: {path}")


def _gather_documents() -> List[Document]:
    """
    Walk the processed docs directory and create LangChain Documents.
    """
    if not PROCESSED_DOCS_DIR.exists():
        raise FileNotFoundError(f"Processed docs directory not found: {PROCESSED_DOCS_DIR}")

    documents: List[Document] = []
    search_dirs = [PROCESSED_DOCS_DIR, KNOWLEDGE_BASE_DIR]
    for directory in search_dirs:
        if not directory.exists():
            print(f"Warning: Directory not found: {directory}")
            continue

        for path in sorted(directory.glob("**/*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".txt", ".md", ".pdf", ".csv"}:
                continue
            text = _load_text_from_file(path)
            if not text.strip():
                continue
            documents.append(Document(page_content=text, metadata={"source": str(path.relative_to(directory))}))

    if not documents:
        raise ValueError("No valid documents found for vector store construction.")

    return documents


def build_vectorstore() -> None:
    """
    Build the Chroma vector store from processed documents and persist it locally.
    """
    documents = _gather_documents()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    split_docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=str(PERSIST_DIR),
    )


if __name__ == "__main__":
    build_vectorstore()
