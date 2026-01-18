"""
Pydantic schemas for the FastAPI backend.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field


SequenceType = List[Union[List[float], Dict[str, float]]]


class PredictRequest(BaseModel):
    sequence: SequenceType = Field(..., description="Time-series data points for inference.")
class PredictResponse(BaseModel):
    trajectory: List[Dict[str, float]]


class AgentRequest(BaseModel):
    query: str


class AgentResponse(BaseModel):
    response: str


# =============================================================================
# Safety Auditor Schemas
# =============================================================================

class SafetyAuditRequest(BaseModel):
    """Request for Safety Auditor agent."""
    sequence: SequenceType = Field(
        ..., 
        description="Vehicle state sequence with keys: Local_X, Local_Y, v_Vel, v_Acc, Space_Headway, etc."
    )
    predicted_trajectory: Optional[List[Dict[str, float]]] = Field(
        None,
        description="Precomputed LSTM trajectory prediction (full state including velocity, acceleration, etc.).",
    )


class SafetyAuditResponse(BaseModel):
    """Response from Safety Auditor agent."""
    status: str = Field(..., description="Safety status: SAFE, WARNING, or CRITICAL")
    violations: List[str] = Field(default=[], description="List of detected violations")
    report: str = Field(..., description="Full LLM-generated safety report")


# =============================================================================
# Driver Profiler Schemas
# =============================================================================

class DriverProfileRequest(BaseModel):
    """Request for Driver Profiler agent."""
    sequence: Optional[SequenceType] = Field(
        None,
        description="Full vehicle state sequence. If provided, metrics are extracted automatically."
    )
    velocity_series: Optional[List[float]] = Field(
        None,
        description="List of velocity values (m/s). Required if sequence not provided."
    )
    acceleration_series: Optional[List[float]] = Field(
        None,
        description="List of acceleration values (m/s²). Required if sequence not provided."
    )
    lane_changes: Optional[int] = Field(
        None,
        description="Number of lane changes detected in the session."
    )
    headway_series: Optional[List[float]] = Field(
        None,
        description="List of headway/following distance values (m)."
    )
    predicted_trajectory: Optional[List[Dict[str, float]]] = Field(
        None,
        description="Precomputed LSTM trajectory prediction (full state including velocity, acceleration, etc.).",
    )


class DriverProfileResponse(BaseModel):
    """Response from Driver Profiler agent."""
    classification: str = Field(
        ..., 
        description="Driver style: Aggressive, Defensive, Distracted, or Normal"
    )
    confidence: int = Field(..., description="Classification confidence (0-100)")
    recommendations: List[str] = Field(
        default=[], 
        description="List of improvement suggestions"
    )
    report: str = Field(..., description="Full LLM-generated profile report")
