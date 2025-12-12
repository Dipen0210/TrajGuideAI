"""
Tool for consulting traffic safety rules from the likelihood knowledge base.
"""
from __future__ import annotations
from typing import Dict
from langchain.tools import tool
from agent.rag.rag_chain import rag_query

@tool("consult_safety_rules")
def consult_safety_rules(query: str) -> Dict[str, str | list]:
    """
    Retrieves specific traffic safety rules or limits from the knowledge base.
    Use this when you need to know "safe following distance", "speed limits",
    or "braking thresholds".
    
    Args:
        query: The specific question about safety rules (e.g. "What is the safe headway at 25m/s?")
    """
    # We can reuse the existing rag_query since it now points to the
    # expanded knowledge base including traffic_rules.txt
    return rag_query(query)
