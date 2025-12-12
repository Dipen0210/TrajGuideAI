"""
FastAPI backend for the Agentic Vehicle Trajectory Prediction System.
"""

from __future__ import annotations

from typing import Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from agent.tools.prediction_tool import _sequence_to_dataframe
from model.inference import TrajectoryInference

from .schemas import (
    PredictRequest,
    PredictResponse,
    AgentRequest,
    AgentResponse,
)
from agent.agent_pipeline import run_agent


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
        prediction = INFERENCE.predict(df, steps=3)
        return PredictResponse(**prediction)
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(status_code=400, detail=str(exc)) from exc






@app.post("/agent/run", response_model=AgentResponse)
def agent_endpoint(request: AgentRequest) -> AgentResponse:
    try:
        response = run_agent(request.query)
        return AgentResponse(response=response)
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(status_code=400, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
