"""
LangChain tool that asks Llama 3 to explain predicted trajectories.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from pydantic import BaseModel

from langchain.tools import tool

from agent.llm.llama3_client import load_llama3


_LLAMA3 = load_llama3()
LOGGER = logging.getLogger(__name__)


def _format_metadata(metadata: Optional[Dict]) -> str:
    if not metadata:
        return "No additional vehicle telemetry was provided."

    parts = []
    speed = metadata.get("v_Vel")
    acceleration = metadata.get("v_Acc")
    space_headway = metadata.get("Space_Headway")
    dis_cen = metadata.get("dis_cen")
    lane_distances = {
        "left": metadata.get("dis_l"),
        "right": metadata.get("dis_r"),
        "front": metadata.get("dis_f"),
    }
    indicators = {
        "left_signal": metadata.get("i_l"),
        "right_signal": metadata.get("i_r"),
        "front_indicator": metadata.get("i_f"),
    }

    if speed is not None:
        parts.append(f"Speed: {speed}")
    if acceleration is not None:
        parts.append(f"Acceleration: {acceleration}")
    if space_headway is not None:
        parts.append(f"Space headway: {space_headway}")
    if dis_cen is not None:
        parts.append(f"Lane center offset: {dis_cen}")

    lane_info = ", ".join(f"{k}={v}" for k, v in lane_distances.items() if v is not None)
    if lane_info:
        parts.append(f"Relative distances -> {lane_info}")

    indicator_info = ", ".join(f"{k}={v}" for k, v in indicators.items() if v is not None)
    if indicator_info:
        parts.append(f"Indicators -> {indicator_info}")

    if not parts:
        return "Telemetry provided but contained no recognized fields."

    return " | ".join(parts)


class ExplanationInput(BaseModel):
    prediction: Dict
    metadata: Optional[Dict] = None


def _generate_explanation(prediction: Dict, metadata: Optional[Dict] = None) -> str:
    meta_summary = _format_metadata(metadata)
    prompt = f"""
You are Llama 3, an automotive trajectory analyst.
Given the predicted next coordinates and current telemetry, explain what the vehicle is doing,
why the LSTM produced this forecast, and how surrounding context influences it.

Predicted Local_X: {prediction.get("predicted_local_x")}
Predicted Local_Y: {prediction.get("predicted_local_y")}
Telemetry summary: {meta_summary}

Respond with a concise multi-sentence explanation focused on behavior, dynamics, and context.
"""
    try:
        return _LLAMA3.generate_text(prompt.strip())
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.warning("Explanation generation failed: %s", exc)
        return (
            "Unable to contact the Llama 3 explanation service at the moment. "
            "Please verify the LLM configuration or internet connectivity."
        )


@tool("explain_trajectory", args_schema=ExplanationInput)
def explain_trajectory_tool(input_data: ExplanationInput) -> str:
    """
    LangChain tool wrapper that expects {"prediction": {...}, "metadata": {...}}.
    """

    if not input_data.prediction:
        raise ValueError("explain_trajectory requires a prediction object.")
    return _generate_explanation(input_data.prediction, input_data.metadata)


def explain_trajectory(prediction: Dict, metadata: Optional[Dict] = None) -> str:
    """
    Convenience function for backend callers to reuse the same explanation logic.
    """

    return _generate_explanation(prediction, metadata)
