# Overview

## Problem Statement
Autonomous vehicles, ADAS platforms, and smart infrastructure rely on accurate short-term trajectory predictions to anticipate driver behavior, avoid collisions, and coordinate maneuvers. Classical kinematic models struggle when real-world traffic introduces noisy inputs, lane changes, and abrupt speed shifts. The project tackles this gap by building a modular system that can ingest thousands of telemetry recordings, learn motion patterns, and reason about the safety implications in natural language.

## Motivation
- **Safety**: Anticipating the motion of adjacent vehicles is critical for collision avoidance, emergency braking, and adaptive cruise control.
- **Explainability**: Human operators and regulators demand clear narratives about why a system issued a warning or decided on a maneuver.
- **Operational intelligence**: Combining predictive modeling with RAG knowledge allows the system to surface relevant policies, weather considerations, and best practices.

## High-Level Summary
1. **Data Layer** – 9,400 CSVs are merged, cleaned, normalized, and transformed into sliding windows.
2. **ML Layer** – A PyTorch LSTM consumes the windows and predicts the next `(Local_X, Local_Y)` coordinates.
3. **Inference + Tools** – A unified inference module plus LangChain tools (prediction, explanation, risk, RAG).
4. **Agentic Reasoning** – A Llama 3-powered LangChain agent orchestrates the tools via a ReAct loop.
5. **RAG** – Chroma vector store indexed with safety rules, best practices, and contextual notes.
6. **APIs + UI** – FastAPI exposes the backend, while Streamlit delivers interactive visualizations and agent chat.

The end result is a production-style system that blends deep learning accuracy with agent-driven interpretability and contextual awareness.
