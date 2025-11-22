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
    predicted_local_x: float
    predicted_local_y: float


class ExplainRequest(BaseModel):
    prediction: Dict[str, float]
    metadata: Optional[Dict[str, float]] = None


class ExplainResponse(BaseModel):
    explanation: str


class RiskRequest(BaseModel):
    prediction: Dict[str, float]
    metadata: Optional[Dict[str, float]] = None


class RiskResponse(BaseModel):
    risk_score: float
    risk_factors: str
    recommendation: str

