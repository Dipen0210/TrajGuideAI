"""
FastAPI backend for the Agentic Vehicle Trajectory Prediction System.
"""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from agent.agent_pipeline import AGENT, run_agent
from agent.tools.explanation_tool import explain_trajectory
from agent.tools.prediction_tool import _sequence_to_dataframe, _INFERENCE
from agent.tools.risk_assessment_tool import trajectory_risk_assessment
from model.inference import TrajectoryInference

from .schemas import (
    AgentQueryRequest,
    AgentQueryResponse,
    ExplainRequest,
    ExplainResponse,
    PredictRequest,
    PredictResponse,
    RiskRequest,
    RiskResponse,
)


app = FastAPI(title="Agentic Vehicle Trajectory Prediction System")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

INFERENCE = TrajectoryInference()


@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "Agentic Vehicle Trajectory Prediction API is running."}


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest) -> PredictResponse:
    try:
        df = _sequence_to_dataframe(request.sequence)
        prediction = INFERENCE.predict(df)
        return PredictResponse(**prediction)
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/explain", response_model=ExplainResponse)
def explain_endpoint(request: ExplainRequest) -> ExplainResponse:
    try:
        explanation = explain_trajectory(request.prediction, request.metadata)
        return ExplainResponse(explanation=explanation)
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/risk", response_model=RiskResponse)
def risk_endpoint(request: RiskRequest) -> RiskResponse:
    try:
        risk = trajectory_risk_assessment(request.prediction, request.metadata)
        return RiskResponse(**risk)
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/agent/query", response_model=AgentQueryResponse)
def agent_query_endpoint(request: AgentQueryRequest) -> AgentQueryResponse:
    try:
        response = run_agent(request.query)
        return AgentQueryResponse(response=response)
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(status_code=400, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
