# Agentic Vehicle Trajectory Prediction System

## 🎯 Summary
A graduate-level AI stack that marries sequence modeling, retrieval-augmented reasoning, and interactive tooling. The system ingests thousands of vehicle telemetry CSVs, trains an LSTM for next-point trajectory prediction, layers LangChain + Llama 3 for reasoning, augments responses with RAG knowledge, and exposes everything through a FastAPI backend plus Streamlit dashboard.

## 🚀 Features
- Deep learning trajectory model (PyTorch LSTM with configurable windows and horizons)
- Agentic layer powered by LangChain’s ReAct agent and Llama 3 reasoning
- Retrieval-Augmented Generation (RAG) for safety rules, weather, and contextual policies
- FastAPI backend with production-ready endpoints
- Streamlit UI for visualization, analytics, and conversational agent access
- Modular Python architecture with clean separation of data, model, agent, API, and app layers
- Industrial-grade explanations, quantified risk scores, and an agent that can route between tools

## 📂 Project Structure
```
vehicle-trajectory-agent/
├── backend/
│   ├── __init__.py
│   ├── main.py
│   └── schemas.py
├── agent/
│   ├── agent_pipeline.py
│   ├── llm/
│   │   ├── llama3_client.py
│   │   └── llama3_config.yaml
│   ├── rag/
│   │   ├── build_vectorstore.py
│   │   ├── rag_chain.py
│   │   └── processed_docs/
│   └── tools/
│       ├── context_query_tool.py
│       ├── explanation_tool.py
│       ├── prediction_tool.py
│       └── risk_assessment_tool.py
├── api/
│   ├── main.py
│   └── schemas.py
├── frontend/
│   ├── app.py
│   └── components/
│       └── plot_utils.py
├── data/
│   ├── processed/
│   └── raw/
├── docs/
│   ├── architecture.md
│   ├── overview.md
│   ├── data_preprocessing.md
│   ├── model_training.md
│   ├── agent_system.md
│   ├── rag_system.md
│   ├── api_docs.md
│   ├── frontend_guide.md
│   └── future_work.md
├── model/
│   ├── dataset.py
│   ├── inference.py
│   ├── lstm_model.py
│   ├── train.py
│   └── utils.py
├── notebooks/
├── tests/
├── requirements.txt
└── LICENSE
```

## 📊 Dataset Description
- 9,400 CSV files containing per-frame telemetry: `Local_X`, `Local_Y`, `v_Vel`, `v_Acc`, `Space_Headway`, `dis_cen`, `i_l`, `i_r`, `i_f`, `dis_l`, `dis_r`, `dis_f`.
- Files are concatenated into one normalized DataFrame, bad/missing values are sanitized, and Min-Max scaling is applied.
- Sliding-window generator produces tensors of shape `(window_size, num_features)` with prediction targets `(Local_X, Local_Y)` `prediction_horizon` steps ahead.

## 🧠 ML Model (LSTM)
- Two-layer LSTM (hidden size configurable, default 64) processes each window and regresses next `(Local_X, Local_Y)`.
- Training uses MSE loss and Adam optimizer (1e-3 lr) with GPU acceleration when available.
- Train/validation split (80/20) tracks the best checkpoint saved under `model/checkpoints/`.
- CLI arguments expose batch size, epochs, window size, dropout, etc.

## 🤖 Agentic AI (LangChain + Llama 3)
- Tools:
  - `predict_trajectory`: wraps inference pipeline.
  - `explain_trajectory`: prompts Llama 3 for context-aware explanations.
  - `trajectory_risk_assessment`: prompts Llama 3 for risk scoring + mitigation.
  - `context_query`: bridges into the RAG subsystem for knowledge retrieval.
- Agent pipeline uses LangChain’s Zero-Shot ReAct agent with conversation memory and parsing-resilient reasoning.

## 🧩 RAG System
- Documents (`agent/rag/processed_docs/`) are chunked with RecursiveCharacterTextSplitter.
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`.
- Vector store: persistent Chroma instance at `agent/rag/chroma_db/`.
- `rag_query()` fetches top-k sources and routes them through Llama 3 for grounded answers.

## 🔌 FastAPI Backend
- `/predict`: run inference on uploaded sequences.
- `/explain`: generate natural-language explanation for a prediction/metadata pair.
- `/risk`: obtain risk score, factors, and recommendations.
- `/agent/query`: converse with the agent (which can invoke prediction/explanation/risk/context tools).

## 🖥 Streamlit Frontend
- Upload CSV / paste JSON → preview sequence.
- Trigger predictions, view charts (`plot_trajectory`) and JSON results.
- Explanation view with chat bubbles, risk meter gauge, and interactive agent chat that shows retrieved sources.
- Responsive layout with sidebar controls and multi-column sections.

## ⚙️ Installation
1. **Python**: 3.10+ recommended.
2. **Create environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Configure Llama 3 client**: fill `agent/llm/llama3_config.yaml` with `api_base`, `api_key`, `model`.
4. **Prepare data**: place raw CSVs under `data/raw/` and run the preprocessing script via `TrajectoryPreprocessor`.

## ▶️ Run Instructions
1. **Build vector store**
   ```bash
   python agent/rag/build_vectorstore.py
   ```
2. **Train LSTM**
   ```bash
   python model/train.py --epochs 20 --batch_size 64
   ```
3. **Run API**
   ```bash
   uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
   ```
4. **Launch Streamlit**
   ```bash
   streamlit run frontend/app.py
   ```

## 📚 Documentation
- Detailed markdown guides live under `/docs`. See:
  - `docs/overview.md`
  - `docs/architecture.md`
  - `docs/data_preprocessing.md`
  - `docs/model_training.md`
  - `docs/agent_system.md`
  - `docs/rag_system.md`
  - `docs/api_docs.md`
  - `docs/frontend_guide.md`
  - `docs/future_work.md`

## 🌟 Future Work
- Multi-agent prediction and collaborative planning across multiple vehicles.
- Upgrade to Transformer-based sequence models or diffusion trajectory predictors.
- Integration with CARLA or other simulators for closed-loop testing.
- Multi-modal sensor fusion (LiDAR, camera, V2X) to enrich context.
- Edge deployment on vehicle-grade hardware with pruning/quantization.
