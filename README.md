# 🚗 TrajAI - Agentic Vehicle Trajectory Prediction System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![React](https://img.shields.io/badge/React-18+-61DAFB.svg)
![Vite](https://img.shields.io/badge/Vite-5.0+-646CFF.svg)
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
- **4-Feature Prediction**: Predicts position, velocity, acceleration, and lane metrics
- **Sub-millisecond Inference**: Optimized for real-time applications

### 🤖 AI Agents (Powered by Llama 3.3 70B)
- **Chain-of-Thought Reasoning**: 5-step structured analysis workflow
- **Safety Rule Knowledge Base**: Embedded traffic rules and thresholds
- **Natural Language Reports**: Detailed explanations with actionable insights

### 🖥️ Modern React Dashboard
- **Glassmorphism UI**: Frosted glass effects with backdrop blur
- **Dark Theme**: Premium dark interface with gradient accents
- **Real-time Charts**: Interactive trajectory visualization with Chart.js
- **Responsive Design**: Works on desktop and mobile

### 🔌 Production API
- **RESTful Endpoints**: Clean JSON request/response format
- **Structured Outputs**: Consistent schema for all agent responses
- **CORS Enabled**: Ready for frontend integration

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    TrajAI System Architecture                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────────┐ │
│  │    React     │────▶│   FastAPI    │────▶│  LSTM Model      │ │
│  │   Frontend   │     │   Backend    │     │  (Trajectory)    │ │
│  │  (Vite)      │     │              │     │                  │ │
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
│  │  │           Llama 3.3 70B via HuggingFace              │ │  │
│  │  │           (Cerebras / SambaNova / Together)          │ │  │
│  │  └──────────────────────────────────────────────────────┘ │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
TrajAI/
├── frontend/              # React + Vite frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── common/        # Header, Sidebar
│   │   │   ├── data/          # FileUpload, DataPreview
│   │   │   ├── visualization/ # TrajectoryChart, PredictionPanel
│   │   │   └── analysis/      # SafetyAuditor, DriverProfiler
│   │   ├── services/          # API client
│   │   └── styles/            # Global CSS
│   └── package.json
├── backend/               # FastAPI backend
│   ├── main.py
│   └── schemas.py
├── model/                 # LSTM model
│   ├── lstm_model.py
│   ├── inference.py
│   └── train.py
├── agent/                 # AI agents
│   ├── safety_auditor_agent.py
│   ├── driver_profiler_agent.py
│   └── llm/               # LLM client
└── data/                  # Training data & samples
    ├── raw/
    └── samples/
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.10+
- Node.js 18+
- npm or yarn

### Backend Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Dipen0210/TrajGuideAI.git
   cd TrajGuideAI
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   
   Create a `.env` file:
   ```env
   HF_TOKEN=your_huggingface_token
   HF_MODEL_ID=meta-llama/Llama-3.3-70B-Instruct
   LLAMA3_PROVIDER=cerebras
   ```

### Frontend Setup

```bash
cd frontend
npm install
```

---

## ▶️ Usage

### Start Backend API

```bash
source .venv/bin/activate
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### Start React Frontend

```bash
cd frontend
npm run dev -- --port 3000
```

### Access the Application

- **Frontend**: http://localhost:3000
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
    {"Local_X": 5.0, "Local_Y": 100.0, "v_Vel": 25.0, ...}
  ]
}
```

### Safety Audit

```http
POST /agent/safety-audit
```

**Response:**
```json
{
  "status": "SAFE",
  "report": "Analysis details...",
  "violations": []
}
```

### Driver Profile

```http
POST /agent/driver-profile
```

**Response:**
```json
{
  "classification": "Normal",
  "confidence": 85,
  "recommendations": ["..."]
}
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | React 18, Vite, Chart.js |
| **Backend** | FastAPI, Python 3.10+ |
| **ML Model** | PyTorch LSTM |
| **LLM** | Llama 3.3 70B Instruct |
| **Inference** | HuggingFace (Cerebras/SambaNova) |
| **Agent Framework** | LangChain |

---

## 📜 License

This project is licensed under the MIT License.

---

<div align="center">
  <p>Built with ❤️ using PyTorch, LangChain, React, and Llama 3</p>
</div>
