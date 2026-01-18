# 🚗 TrajAI - Agentic Vehicle Trajectory Prediction System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-orange.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.1+-purple.svg)
![Llama](https://img.shields.io/badge/Llama_3.3-70B-yellow.svg)

**AI-powered vehicle trajectory prediction with real-time safety analysis and driver profiling**

[Features](#-features) • [Architecture](#-architecture) • [Installation](#️-installation) • [Usage](#-usage) • [API Reference](#-api-reference)

</div>

---

## 🎯 Overview

TrajAI is a graduate-level AI system that combines deep learning sequence modeling with LLM-powered reasoning agents. The system ingests vehicle telemetry, predicts future trajectories using an LSTM neural network, and deploys two specialized AI agents for safety and behavior analysis.

### Dual-Agent Architecture

| Agent | Purpose | Output |
|-------|---------|--------|
| **🛡️ Safety Auditor** | Real-time trajectory monitoring | SAFE / WARNING / CRITICAL status |
| **🏎️ Driver Profiler** | Behavioral pattern analysis | Driver classification + recommendations |

---

## ✨ Features

### 🧠 Deep Learning Core
- **LSTM Model**: Two-layer LSTM architecture for precise sequence forecasting
- **12-Feature Prediction**: Predicts position, velocity, acceleration, and lane metrics
- **Sub-millisecond Inference**: Optimized for real-time applications

### 🤖 AI Agents (Powered by Llama 3.3 70B)
- **Chain-of-Thought Reasoning**: 5-step structured analysis workflow
- **Safety Rule Knowledge Base**: Embedded traffic rules and thresholds
- **Natural Language Reports**: Detailed explanations with actionable insights

### 🖥️ Modern Dashboard
- **Dark Theme UI**: Sleek, modern interface with gradient accents
- **Real-time Visualization**: Interactive trajectory plots
- **Status Badges**: Visual indicators for safety status
- **Confidence Metrics**: Progress bars and classification displays

### 🔌 Production API
- **RESTful Endpoints**: Clean JSON request/response format
- **Structured Outputs**: Consistent schema for all agent responses
- **Error Handling**: Graceful degradation with helpful messages

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    TrajAI System Architecture                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────────┐ │
│  │   Streamlit  │────▶│   FastAPI    │────▶│  LSTM Model      │ │
│  │   Frontend   │     │   Backend    │     │  (Trajectory)    │ │
│  └──────────────┘     └──────┬───────┘     └──────────────────┘ │
│                              │                                   │
│                              ▼                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    AI Agent Layer                          │  │
│  │  ┌─────────────────┐       ┌─────────────────────────┐    │  │
│  │  │  Safety Auditor │       │    Driver Profiler      │    │  │
│  │  │  - Rule Check   │       │  - Behavior Analysis    │    │  │
│  │  │  - Violation    │       │  - Style Classification │    │  │
│  │  │    Detection    │       │  - Recommendations      │    │  │
│  │  └────────┬────────┘       └────────────┬────────────┘    │  │
│  │           │                             │                  │  │
│  │           ▼                             ▼                  │  │
│  │  ┌──────────────────────────────────────────────────────┐ │  │
│  │  │           Llama 3.3 70B via Cerebras                 │ │  │
│  │  │           (HuggingFace Inference Providers)          │ │  │
│  │  └──────────────────────────────────────────────────────┘ │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Dataset

The system is trained on vehicle telemetry data with the following features:

| Feature | Description |
|---------|-------------|
| `Local_X`, `Local_Y` | Vehicle position coordinates |
| `v_Vel` | Velocity (m/s) |
| `v_Acc` | Acceleration (m/s²) |
| `Space_Headway` | Distance to lead vehicle |
| `dis_cen` | Distance from lane center |
| `i_l`, `i_r`, `i_f` | Lane change indicators |
| `dis_l`, `dis_r`, `dis_f` | Distance to adjacent vehicles |

---

## ⚙️ Installation

### Prerequisites
- Python 3.10+
- pip or conda

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/TrajAI.git
   cd TrajAI
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   
   Create a `.env` file in the project root:
   ```env
   # HuggingFace Configuration
   HF_TOKEN=your_huggingface_token
   HF_MODEL_ID=meta-llama/Llama-3.3-70B-Instruct
   LLAMA3_PROVIDER=cerebras
   ```

---

## ▶️ Usage

### Start the Backend API

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### Launch the Dashboard

```bash
streamlit run frontend/app.py --server.port 8501
```

### Access the Application

- **Dashboard**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs

---

## 📡 API Reference

### Predict Trajectory

```http
POST /predict
```

**Request:**
```json
{
  "sequence": [
    {"Local_X": 5.0, "Local_Y": 100.0, "v_Vel": 25.0, ...},
    {"Local_X": 5.1, "Local_Y": 125.0, "v_Vel": 25.5, ...}
  ]
}
```

**Response:**
```json
{
  "trajectory": [
    {"predicted_local_x": 5.2, "predicted_local_y": 150.0},
    {"predicted_local_x": 5.3, "predicted_local_y": 175.0}
  ]
}
```

### Safety Audit

```http
POST /agent/safety-audit
```

**Request:**
```json
{
  "sequence": [...],
  "predicted_trajectory": [...]
}
```

**Response:**
```json
{
  "status": "WARNING",
  "report": "Detailed safety analysis...",
  "violations": ["tailgating", "speeding"],
  "chain_steps": [...]
}
```

### Driver Profile

```http
POST /agent/driver-profile
```

**Response:**
```json
{
  "classification": "Aggressive",
  "confidence": 85,
  "report": "Driver behavior analysis...",
  "recommendations": ["Maintain larger following distance", ...]
}
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **ML Model** | PyTorch LSTM |
| **LLM** | Llama 3.3 70B Instruct |
| **Inference Provider** | Cerebras (via HuggingFace) |
| **Agent Framework** | LangChain |
| **Backend** | FastAPI |
| **Frontend** | Streamlit |
| **Embeddings** | HuggingFace Embeddings |

---

## 🌟 Future Work

- **Multi-Vehicle Coordination**: Agents negotiating right-of-way scenarios
- **Vision Integration**: Camera input for enhanced safety analysis
- **Reinforcement Learning**: Using profiler feedback to train driving policies
- **Edge Deployment**: Optimized models for in-vehicle inference

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <p>Built with ❤️ using PyTorch, LangChain, and Llama 3</p>
</div>
