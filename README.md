# Agentic Vehicle Trajectory Prediction System

## 🎯 Summary
A graduate-level AI stack that marries sequence modeling, retrieval-augmented reasoning, and interactive tooling into a **Dual-Agent Architecture**. The system ingests vehicle telemetry, trains an LSTM for next-point trajectory prediction, and deploys two specialized AI agents powered by Llama 3 and RAG:
1.  **Autonomous Safety Auditor**: Monitors real-time trajectories for safety violations and adherence to traffic rules.
2.  **Driver Style Profiler**: Analyzes long-term driving patterns to classify styles (e.g., Aggressive, Defensive, Distracted) and provide coaching.

All components are exposed through a FastAPI backend and a verified Streamlit dashboard.

## 🚀 Features
-   **Dual-Agent System**:
    -   **Safety Auditor**: Determines "SAFE", "WARNING", or "CRITICAL" status with 5-step Chain-of-Thought reasoning.
    -   **Driver Profiler**: Computes metrics (lane changes, headway, jerk) to classify behavior relative to "Golden Driver" benchmarks.
-   **Deep Learning Core**: PyTorch LSTM model with configurable windows for precise sequence forecasting.
-   **RAG Knowledge Base**: Vectorized traffic rules and driver style guidelines for grounded agent reasoning.
-   **Interactive Dashboard**: Streamlit UI with real-time status badges, violation lists, and confidence visualizations.
-   **Production API**: Dedicated endpoints (`/agent/safety-audit`, `/agent/driver-profile`) returning structured JSON reports.

## 📊 Dataset Description
-   **Telemetry**: 9,400+ CSV files containing `Local_X`, `Local_Y`, `v_Vel`, `v_Acc`, `Space_Headway`, and more.
-   **Preprocessing**: Automated sanitization, missing value handling, and Min-Max scaling.
-   **Sliding Windows**: Generators produce `(window_size, num_features)` tensors for LSTM training.

## 🧠 ML Model (LSTM)
-   **Architecture**: Two-layer LSTM (default hidden size 64) -> Linear Regression for `(x, y)`.
-   **Training**: MSE loss, Adam optimizer, customizable epochs/batch size via CLI.
-   **Inference**: Sub-millisecond latency for real-time trajectory prediction.

## 🤖 Dual-Agent Architecture
Powered by **Llama 3** (via HuggingFace) and **LangChain**:

### 1. Safety Auditor Agent (`/agent/safety-audit`)
-   **Goal**: Prevent accidents by detecting unsafe maneuvers in real-time.
-   **Tools**: `predict_trajectory`, `consult_safety_rules`.
-   **Workflow**: Observation -> Prediction -> Rule Retrieval -> Violation Check -> Report.

### 2. Driver Profiler Agent (`/agent/driver-profile`)
-   **Goal**: Assess long-term driving habits for insurance or coaching.
-   **Tools**: `analyze_driving_profile` (calculates volatility, lane changes, time-to-collision).
-   **Workflow**: Metric Extraction -> Benchmark Comparison -> Classification -> Recommendation.

## 🔌 FastAPI Backend
-   **`POST /agent/safety-audit`**: Submit trajectory -> Get status & violation report.
-   **`POST /agent/driver-profile`**: Submit session data -> Get classification & coaching tips.
-   **`POST /predict`**: Raw LSTM trajectory inference.

## 🖥 Streamlit Frontend
-   **Agent Control Center**: Specialized tabs for running Safety Audits and Driver Profiling.
-   **Visualizations**:
    -   Trajectory plots.
    -   Status badges (✅ SAFE, 🚨 CRITICAL).
    -   Driver style confidence bars.
    -   Expandable detailed LLM reports.

## ⚙️ Installation
1.  **Python**: 3.10+ recommended.
2.  **Create environment**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```
3.  **Configure API Keys**:
    -   Create a `.env` file or export variables:
        ```bash
        export HF_API_KEY="your_huggingface_token"
        export HF_MODEL_ID="meta-llama/Llama-3.3-70B-Instruct"
        ```
        
## ▶️ Run Instructions
1.  **Start Backend API**:
    ```bash
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
    ```
2.  **Launch Dashboard**:
    ```bash
    streamlit run frontend/app.py
    ```
3.  **Verify Agents**:
    -   Upload a sample CSV in the UI.
    -   Click **"Run Safety Audit"** or **"Analyze Driver Style"**.

## 🌟 Future Work
-   **Multi-Vehicle Coordination**: Agents negotiating right-of-way.
-   **Vision Integration**: Adding camera input to the Safety Auditor.
-   **Reinforcement Learning**: Using Driver Profiler feedback to train self-driving policies.
