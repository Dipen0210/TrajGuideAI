"""
Tool for consulting traffic safety rules.

NOTE: RAG system bypassed due to ChromaDB/gRPC mutex lock issues on macOS.
Using static rules embedded directly instead.
"""
from __future__ import annotations
from typing import Dict
from langchain.tools import tool


# Static safety rules to avoid ChromaDB/gRPC mutex lock issues
STATIC_SAFETY_RULES = """
## Traffic Safety Rules and Thresholds

### Following Distance (Headway)
- SAFE: Minimum 2-3 second gap (at 25 m/s, this is 50-75 meters)
- WARNING: 1-2 second gap (at 25 m/s, this is 25-50 meters)
- CRITICAL: Less than 1 second gap (at 25 m/s, less than 25 meters)

### Speed Limits
- Urban areas: 30-50 km/h (8-14 m/s)
- Highway: 100-130 km/h (28-36 m/s)
- Residential: 20-30 km/h (6-8 m/s)

### Safe Braking Thresholds
- Normal braking: -2 to -3 m/s²
- Hard braking: -3 to -5 m/s²
- Emergency braking: > -5 m/s²

### Lane Change Safety
- Safe lane change requires: 
  - Speed difference < 10 km/h from target lane traffic
  - Gap of at least 3 seconds in target lane
  - Clear indicator for 3+ seconds before maneuver

### Driver Style Benchmarks:
- Aggressive: Std acceleration > 1.5 m/s², hard braking < -4 m/s², headway < 1.5s
- Defensive: Std acceleration < 0.5 m/s², gentle braking > -2 m/s², headway > 2.5s
- Distracted: Variable metrics, delayed reactions, inconsistent headway
- Normal: Std acceleration 0.5-1.5 m/s², moderate braking, headway 1.5-2.5s

### Collision Risk Assessment
- Time-to-collision (TTC) < 2s: CRITICAL
- TTC 2-4s: WARNING
- TTC > 4s: SAFE
"""


@tool("consult_safety_rules")
def consult_safety_rules(query: str) -> Dict[str, str | list]:
    """
    Retrieves specific traffic safety rules or limits from the knowledge base.
    Use this when you need to know "safe following distance", "speed limits",
    or "braking thresholds".
    
    Args:
        query: The specific question about safety rules (e.g. "What is the safe headway at 25m/s?")
    """
    # Return static rules to avoid ChromaDB/gRPC mutex lock issues
    return {
        "answer": STATIC_SAFETY_RULES,
        "sources": [{"type": "static_rules", "description": "Embedded traffic safety rules"}]
    }

