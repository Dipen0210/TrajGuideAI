"""
Retrieval-Augmented Generation chain using Chroma and Llama 3.
"""

from __future__ import annotations

import logging
import socket
from pathlib import Path
from typing import Dict, List

from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from agent.llm.llama3_client import load_llama3


PERSIST_DIR = Path(__file__).resolve().parent / "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LOGGER = logging.getLogger(__name__)


def _server_resolves(host: str = "huggingface.co") -> bool:
    try:
        socket.gethostbyname(host)
        return True
    except socket.gaierror:
        return False

_QA_CHAIN = None
try:
    if not _server_resolves():
        raise RuntimeError("Cannot resolve huggingface.co; skipping RAG initialization.")

    _EMBEDDINGS = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    _VECTORSTORE = Chroma(
        persist_directory=str(PERSIST_DIR),
        embedding_function=_EMBEDDINGS,
    )
    _RETRIEVER = _VECTORSTORE.as_retriever(search_kwargs={"k": 3})
    _QA_CHAIN = RetrievalQA.from_chain_type(
        llm=load_llama3(),
        chain_type="stuff",
        retriever=_RETRIEVER,
        return_source_documents=True,
    )
except Exception as exc:  # pylint: disable=broad-except
    LOGGER.warning("Failed to initialize RAG chain: %s", exc)
    _QA_CHAIN = None


def rag_query(question: str) -> Dict[str, List[dict] | str]:
    """
    Runs retrieval + Llama 3 generation and returns the answer and source metadata.
    """
    if _QA_CHAIN is None:
        LOGGER.info("RAG chain unavailable; returning fallback response.")
        return {
            "answer": "Context retrieval is temporarily unavailable. Please retry later.",
            "sources": [],
        }
    result = _QA_CHAIN(question)
    return {
        "answer": result["result"],
        "sources": [doc.metadata for doc in result.get("source_documents", [])],
    }
