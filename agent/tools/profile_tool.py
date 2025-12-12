"""
Tool for profiling driver behavior statistics.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, List
from langchain.tools import tool

@tool("analyze_driving_profile")
def analyze_driving_profile(velocity_series: List[float], acceleration_series: List[float]) -> Dict[str, float | str]:
    """
    Analyzes a sequence of vehicle states to profile the driving style.
    Computes statistical metrics (aggression, stability) and classifies the behavior.
    
    Args:
        velocity_series: List of velocity values (m/s).
        acceleration_series: List of acceleration values (m/s^2).
    """
    if not velocity_series or not acceleration_series:
        return {"error": "Insufficient data for profiling."}

    # Convert to numpy for stats
    vel = np.array(velocity_series)
    acc = np.array(acceleration_series)

    # Compute Metrics
    mean_speed = float(np.mean(vel))
    std_acc = float(np.std(acc))
    max_dec = float(np.min(acc))  # Negative value for braking
    jerk = float(np.mean(np.diff(acc))) if len(acc) > 1 else 0.0

    # Classification Logic (Simplified based on knowledge base)
    style = "Normal"
    if std_acc > 1.5 or max_dec < -4.0:
        style = "Aggressive"
    elif std_acc < 0.5 and max_dec > -2.0:
        style = "Defensive"
    
    return {
        "style_classification": style,
        "metrics": {
            "mean_velocity": round(mean_speed, 2),
            "std_acceleration": round(std_acc, 2),
            "max_braking": round(max_dec, 2),
            "avg_jerk": round(jerk, 2)
        }
    }
