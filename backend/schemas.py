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
