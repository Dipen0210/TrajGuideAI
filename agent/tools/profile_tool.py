"""
Tool for profiling driver behavior statistics.

Enhanced with:
- Lane change detection
- Headway analysis
- Distracted driver classification
- Golden Driver comparison
"""
from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Any
from langchain.tools import tool


# Golden Driver benchmarks (ideal thresholds)
GOLDEN_DRIVER = {
    "std_acceleration": (0.3, 0.5),  # (min, max) ideal range
    "max_braking": -2.5,  # Should be greater than this (less harsh)
    "avg_jerk": 0.5,  # Should be less than this
    "headway_seconds": 2.5,  # Should be greater than this
    "speed_variance": 2.0,  # Should be less than this
}


@tool("analyze_driving_profile")
def analyze_driving_profile(
    velocity_series: List[float], 
    acceleration_series: List[float],
    headway_series: Optional[List[float]] = None,
    lane_change_count: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Analyzes a sequence of vehicle states to profile the driving style.
    Computes statistical metrics and classifies the behavior as:
    - Aggressive: High acceleration variance, hard braking, tailgating
    - Defensive: Smooth driving, large safety margins
    - Distracted: Erratic patterns, delayed responses, inconsistent behavior
    - Normal: Average driving within acceptable bounds
    
    Args:
        velocity_series: List of velocity values (m/s).
        acceleration_series: List of acceleration values (m/s^2).
        headway_series: Optional list of headway/distance values (m).
        lane_change_count: Optional count of lane changes in the session.
    
    Returns:
        Dictionary with style classification, detailed metrics, and comparison to Golden Driver.
    """
    if not velocity_series or not acceleration_series:
        return {"error": "Insufficient data for profiling."}

    # Convert to numpy for stats
    vel = np.array(velocity_series)
    acc = np.array(acceleration_series)

    # Compute Core Metrics
    mean_speed = float(np.mean(vel))
    std_speed = float(np.std(vel))
    std_acc = float(np.std(acc))
    max_dec = float(np.min(acc))  # Negative value for braking
    max_acc = float(np.max(acc))
    
    # Jerk: rate of change of acceleration
    jerk = np.diff(acc) if len(acc) > 1 else np.array([0.0])
    avg_jerk = float(np.mean(np.abs(jerk)))
    max_jerk = float(np.max(np.abs(jerk))) if len(jerk) > 0 else 0.0
    
    # Headway analysis (if provided)
    headway_metrics = {}
    if headway_series and len(headway_series) > 0:
        hw = np.array(headway_series)
        headway_metrics = {
            "min_headway": round(float(np.min(hw)), 2),
            "mean_headway": round(float(np.mean(hw)), 2),
            "headway_variance": round(float(np.std(hw)), 2),
        }
        # Calculate time headway (headway / velocity) approximately
        avg_time_headway = float(np.mean(hw)) / mean_speed if mean_speed > 0 else 0
        headway_metrics["avg_time_headway_seconds"] = round(avg_time_headway, 2)
    
    # Classification Logic with enhanced Distracted detection
    style = "Normal"
    confidence = 70  # Default confidence
    risk_factors = []
    
    # Check for Aggressive indicators
    aggressive_score = 0
    if std_acc > 1.5:
        aggressive_score += 2
        risk_factors.append("High acceleration variance")
    if max_dec < -4.0:
        aggressive_score += 2
        risk_factors.append("Frequent hard braking")
    if headway_metrics.get("avg_time_headway_seconds", 3.0) < 1.5:
        aggressive_score += 2
        risk_factors.append("Tailgating behavior")
    if lane_change_count and lane_change_count > 2:
        aggressive_score += 1
        risk_factors.append("Frequent lane changes")
    
    # Check for Distracted indicators
    distracted_score = 0
    if std_speed > 5.0:  # High speed variance without traffic cause
        distracted_score += 2
        risk_factors.append("Inconsistent speed maintenance")
    if headway_metrics.get("headway_variance", 0) > 10:
        distracted_score += 2
        risk_factors.append("Erratic following distance")
    if avg_jerk > 1.5:  # Jerky, unpredictable movements
        distracted_score += 1
        risk_factors.append("Jerky acceleration patterns")
    
    # Check for Defensive indicators
    defensive_score = 0
    if std_acc < 0.5 and max_dec > -2.0:
        defensive_score += 2
    if headway_metrics.get("avg_time_headway_seconds", 0) > 2.5:
        defensive_score += 2
    if 0.3 <= std_acc <= 0.5:  # Golden Driver range
        defensive_score += 1
    
    # Determine classification based on scores
    if aggressive_score >= 3:
        style = "Aggressive"
        confidence = min(95, 60 + aggressive_score * 10)
    elif distracted_score >= 3:
        style = "Distracted"
        confidence = min(90, 55 + distracted_score * 10)
    elif defensive_score >= 3:
        style = "Defensive"
        confidence = min(90, 60 + defensive_score * 10)
    else:
        style = "Normal"
        confidence = 70
    
    # Golden Driver Comparison
    golden_comparison = {
        "acceleration_smoothness": "PASS" if 0.3 <= std_acc <= 0.5 else ("CLOSE" if std_acc < 0.7 else "FAIL"),
        "braking_gentleness": "PASS" if max_dec > GOLDEN_DRIVER["max_braking"] else "FAIL",
        "jerk_comfort": "PASS" if avg_jerk < GOLDEN_DRIVER["avg_jerk"] else "FAIL",
        "speed_consistency": "PASS" if std_speed < GOLDEN_DRIVER["speed_variance"] else "FAIL",
    }
    if headway_metrics.get("avg_time_headway_seconds"):
        golden_comparison["safe_following"] = (
            "PASS" if headway_metrics["avg_time_headway_seconds"] > GOLDEN_DRIVER["headway_seconds"] 
            else "FAIL"
        )
    
    golden_score = sum(1 for v in golden_comparison.values() if v == "PASS")
    golden_total = len(golden_comparison)
    
    return {
        "style_classification": style,
        "confidence_percent": confidence,
        "metrics": {
            "mean_velocity": round(mean_speed, 2),
            "velocity_std": round(std_speed, 2),
            "std_acceleration": round(std_acc, 2),
            "max_braking": round(max_dec, 2),
            "max_acceleration": round(max_acc, 2),
            "avg_jerk": round(avg_jerk, 2),
            "max_jerk": round(max_jerk, 2),
            **headway_metrics,
        },
        "risk_factors": risk_factors,
        "golden_driver_comparison": golden_comparison,
        "golden_driver_score": f"{golden_score}/{golden_total}",
        "lane_changes_analyzed": lane_change_count,
    }
