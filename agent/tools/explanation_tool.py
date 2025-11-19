"""
LangChain tool that asks Llama 3 to explain predicted trajectories.
"""

from __future__ import annotations

from typing import Dict, Optional

from langchain.tools import tool

from agent.llm.llama3_client import load_llama3


_LLAMA3 = load_llama3()


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


@tool("explain_trajectory")
def explain_trajectory(prediction: Dict, metadata: Optional[Dict] = None) -> str:
    """
    Generate a natural language explanation of the predicted motion.
    """

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
    return _LLAMA3.generate_text(prompt.strip())
