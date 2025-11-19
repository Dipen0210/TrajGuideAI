"""
LangChain tool that evaluates trajectory risk via the Llama 3 client.
"""

from __future__ import annotations

import json
from typing import Dict, Optional

from langchain.tools import tool

from agent.llm.llama3_client import load_llama3


_LLAMA3 = load_llama3()


def _metadata_summary(metadata: Optional[Dict]) -> str:
    if not metadata:
        return "No supplemental telemetry provided."
    summary_parts = []
    for key, value in metadata.items():
        summary_parts.append(f"{key}={value}")
    return ", ".join(summary_parts)


@tool("trajectory_risk_assessment")
def trajectory_risk_assessment(prediction: Dict, metadata: Optional[Dict] = None) -> Dict:
    """
    Estimate motion risk, causal factors, and mitigation advice.
    """

    prompt = f"""
You are Llama 3 performing a vehicle trajectory risk assessment.
Analyze the predicted coordinates (Local_X={prediction.get("predicted_local_x")},
Local_Y={prediction.get("predicted_local_y")}) together with any telemetry.

Telemetry: {_metadata_summary(metadata)}

Respond strictly as JSON with the following keys:
- risk_score: float between 0 and 1
- risk_factors: short text listing primary factors
- recommendation: actionable suggestion for improving safety
"""
    raw_response = _LLAMA3.generate_text(prompt.strip())
    try:
        structured = json.loads(raw_response)
    except json.JSONDecodeError:
        structured = {
            "risk_score": None,
            "risk_factors": "LLM response could not be parsed as JSON.",
            "recommendation": raw_response,
        }
    return structured
