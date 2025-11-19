"""
Context retrieval tool backed by the RAG pipeline.
"""

from __future__ import annotations

from typing import Dict

from langchain.tools import tool

from agent.rag.rag_chain import rag_query


@tool("context_query")
def context_query(query: str) -> Dict:
    """
    Run a retrieval-augmented generation query over the processed knowledge base.
    """

    result = rag_query(query)
    return {
        "answer": result["answer"],
        "sources": result["sources"],
    }
