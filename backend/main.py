
"""
FastAPI backend for the Agentic Vehicle Trajectory Prediction System.

Endpoints:
- /predict: Run LSTM trajectory prediction
- /agent/run: General-purpose agent query
- /agent/safety-audit: Dedicated Safety Auditor agent
- /agent/driver-profile: Dedicated Driver Profiler agent
"""

from __future__ import annotations

import traceback
from typing import Dict, List, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from agent.tools.prediction_tool import _sequence_to_dataframe
from model.inference import TrajectoryInference

from .schemas import (
    PredictRequest,
    PredictResponse,
    AgentRequest,
    AgentResponse,
    SafetyAuditRequest,
    SafetyAuditResponse,
    DriverProfileRequest,
    DriverProfileResponse,
)
from agent.agent_pipeline import (
    run_agent,
    audit_trajectory,
    profile_driver,
    profile_driver_from_sequence,
)


app = FastAPI(
    title="Agentic Vehicle Trajectory Prediction System",
    description="AI-powered vehicle trajectory prediction with safety auditing and driver profiling",
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

INFERENCE = TrajectoryInference()

@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "message": "Agentic Vehicle Trajectory Prediction API is running.",
        "version": "2.0.0",
        "agents": ["safety-auditor", "driver-profiler"],
    }


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest) -> PredictResponse:
    """Run LSTM-based trajectory prediction."""
    try:
        df = _sequence_to_dataframe(request.sequence)
        prediction = INFERENCE.predict(df, steps=3)
        return PredictResponse(**prediction)
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/agent/run", response_model=AgentResponse)
def agent_endpoint(request: AgentRequest) -> AgentResponse:
    """Run a general-purpose query through the unified agent."""
    try:
        response = run_agent(request.query)
        return AgentResponse(response=response)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# =============================================================================
# Dedicated Agent Endpoints
# =============================================================================

@app.post("/agent/safety-audit", response_model=SafetyAuditResponse)
def safety_audit_endpoint(request: SafetyAuditRequest) -> SafetyAuditResponse:
    """
    🏆 Safety Auditor Agent
    
    Analyzes a vehicle trajectory sequence to detect safety violations.
    Uses LSTM prediction + RAG-based safety rules with Chain of Thought reasoning.
    
    Returns:
        - status: "SAFE", "WARNING", or "CRITICAL"
        - violations: List of detected violations
        - report: Full LLM-generated safety report
    """
    try:
        result = audit_trajectory(
            request.sequence,
            predicted_trajectory=request.predicted_trajectory,
        )
        return SafetyAuditResponse(
            status=result.get("status", "UNKNOWN"),
            violations=result.get("violations", []),
            report=result.get("report", ""),
        )
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/agent/driver-profile", response_model=DriverProfileResponse)
def driver_profile_endpoint(request: DriverProfileRequest) -> DriverProfileResponse:
    """
    🏎️ Driver Style Profiler Agent
    
    Analyzes driving behavior to classify the driver style.
    Compares against "Golden Driver" benchmarks from knowledge base.
    
    Returns:
        - classification: "Aggressive", "Defensive", "Distracted", or "Normal"
        - confidence: Classification confidence (0-100)
        - recommendations: List of improvement suggestions
        - report: Full LLM-generated profile report
    """
    try:
        # Check if full sequence provided or just metrics
        if request.sequence:
            result = profile_driver_from_sequence(
                request.sequence,
                predicted_trajectory=request.predicted_trajectory,
            )
        else:
            result = profile_driver(
                velocity_series=request.velocity_series or [],
                acceleration_series=request.acceleration_series or [],
                lane_changes=request.lane_changes,
                headway_series=request.headway_series,
                predicted_trajectory=request.predicted_trajectory,
            )
        
        return DriverProfileResponse(
            classification=result.get("classification", "Unknown"),
            confidence=result.get("confidence", 0),
            recommendations=result.get("recommendations", []),
            report=result.get("report", ""),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
