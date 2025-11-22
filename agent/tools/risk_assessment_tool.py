"""
LangChain tool that evaluates trajectory risk via the Llama 3 client.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, Optional

from pydantic import BaseModel

from langchain.tools import tool

from agent.llm.llama3_client import load_llama3


_LLAMA3 = load_llama3()
LOGGER = logging.getLogger(__name__)


def _metadata_summary(metadata: Optional[Dict]) -> str:
    if not metadata:
        return "No supplemental telemetry provided."
    summary_parts = []
    for key, value in metadata.items():
        summary_parts.append(f"{key}={value}")
    return ", ".join(summary_parts)


class RiskAssessmentInput(BaseModel):
    prediction: Dict
    metadata: Optional[Dict] = None


def _run_risk_assessment(prediction: Dict, metadata: Optional[Dict] = None) -> Dict:
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
    try:
        raw_response = _LLAMA3.generate_text(prompt.strip())
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.warning("Risk assessment generation failed: %s", exc)
        return {
            "risk_score": 0.0,
            "risk_factors": "LLM unavailable. Check configuration or connectivity.",
            "recommendation": "Retry once the trajectory reasoning service is reachable.",
        }
    try:
        structured = json.loads(raw_response)
    except json.JSONDecodeError:
        structured = {
            "risk_score": 0.0,
            "risk_factors": "LLM response could not be parsed as JSON.",
            "recommendation": raw_response,
        }
    return structured


@tool("trajectory_risk_assessment", args_schema=RiskAssessmentInput)
def trajectory_risk_assessment_tool(input_data: RiskAssessmentInput) -> Dict:
    """
    LangChain tool wrapper that expects {"prediction": {...}, "metadata": {...}}.
    """

    if not input_data.prediction:
        raise ValueError("trajectory_risk_assessment requires a prediction object.")
    return _run_risk_assessment(input_data.prediction, input_data.metadata)


def trajectory_risk_assessment(prediction: Dict, metadata: Optional[Dict] = None) -> Dict:
    """
    Convenience helper for backend callers needing JSON risk analysis.
    """

    return _run_risk_assessment(prediction, metadata)
