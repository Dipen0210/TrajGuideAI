# Architecture

## Textual Block Diagram
```
Raw CSVs ──▶ Preprocessor ──▶ dataset.npz + scalers.pkl ──▶ LSTM Training ──▶ checkpoint.pth
                                                │
                                                ▼
                                          TrajectoryInference
                                                │
                        ┌───────────────LangChain Tools──────────────┐
                        │        │               │                   │
                        ▼        ▼               ▼                   ▼
                predict_trajectory  explain_trajectory  trajectory_risk_assessment  context_query (RAG)
                        │        │               │                   │
                        └───────────────Agent Pipeline (ReAct)───────┘
                                                │
                           ┌───────────────FastAPI Backend───────────────┐
                           │            │                │               │
                           ▼            ▼                ▼               ▼
                       /predict     /explain          /risk       /agent/query
                           │_______________________________________________│
                                                │
                                                ▼
                                       Streamlit Frontend
```

## Data & ML Flow
1. **Data ingestion** merges all CSVs under `data/raw/`, enforces feature schema, handles missing values, normalizes with Min-Max, and exports tensors plus scaler objects.
2. **Model training** uses `TrajectoryDataset` and `TrajectoryLSTM` with configurable hyperparameters, GPU acceleration, and best-checkpoint saving.
3. **Inference** loads `dataset.npz`, `scalers.pkl`, and `trajectory_lstm_best.pth` for consistent preprocessing + prediction.

## Agentic Flow
1. LangChain tools wrap inference, explanations, risk assessments, and RAG retrieval.
2. `agent_pipeline.py` constructs a Zero-Shot ReAct agent with conversation memory.
3. The agent interprets user intent, selects tools, parses outputs, and synthesizes final responses.

## Backend + UI
1. **FastAPI** hosts `/predict`, `/explain`, `/risk`, `/agent/query`, handling JSON schemas defined in `api/schemas.py`.
2. **Streamlit** consumes the API, enabling CSV uploads, manual JSON input, charts, risk gauges, and conversational agent interactions.
3. RAG sources and explanations are surfaced alongside predictions to provide transparency.
