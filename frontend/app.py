"""
Streamlit frontend for the Agentic Vehicle Trajectory Prediction System.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
for path in {APP_DIR, ROOT_DIR}:
    if str(path) not in sys.path:
        sys.path.append(str(path))

from components.plot_utils import plot_risk_meter, plot_trajectory  # noqa: E402


st.set_page_config(page_title="Agentic Vehicle Trajectory Prediction", layout="wide")

STYLE_BLOCK = """
<style>
.explanation-card {
    background: linear-gradient(135deg, #10254d, #1c3b72);
    color: #f9fbff;
    border-radius: 18px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    line-height: 1.6;
    box-shadow: 0 20px 45px rgba(9, 17, 42, 0.25);
    border: 1px solid rgba(255, 255, 255, 0.08);
}
.explanation-card p {
    margin-bottom: 0.9rem;
}
.explanation-card p:last-child {
    margin-bottom: 0;
}
</style>
"""

st.markdown(STYLE_BLOCK, unsafe_allow_html=True)

DEFAULT_BASE_URL = "http://localhost:8000"


def ensure_state() -> None:
    defaults = {
        "sequence": [],
        "prediction": None,
        "explanation": None,
        "risk": None,
        "api_base": DEFAULT_BASE_URL,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


ensure_state()


def parse_csv_file(uploaded_file) -> Optional[List[Dict[str, float]]]:
    df = pd.read_csv(uploaded_file)
    if df.empty:
        raise ValueError("Uploaded CSV is empty.")
    return df.to_dict(orient="records")


def parse_json_sequence(json_text: str) -> List[Dict[str, float]]:
    data = json.loads(json_text)
    if not isinstance(data, list):
        raise ValueError("JSON sequence must be a list of records.")
    parsed = []
    for item in data:
        if isinstance(item, dict):
            parsed.append(item)
        elif isinstance(item, list):
            raise ValueError("JSON list-of-lists not supported in UI. Use list of dicts.")
        else:
            raise ValueError("Sequence entries must be dicts.")
    return parsed


def call_api(endpoint: str, payload: Dict) -> Dict:
    base_url = st.session_state.get("api_base", DEFAULT_BASE_URL).rstrip("/")
    url = f"{base_url}{endpoint}"
    response = requests.post(url, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()


with st.sidebar:
    st.title("Agentic Trajectory System")
    st.markdown("Graduate-level AI system combining deep learning with RAG-backed insights.")
    st.session_state["api_base"] = st.text_input(
        "API Base URL",
        value=st.session_state["api_base"],
    )
    st.markdown("---")
    predict_clicked = st.button("Predict Trajectory")
    explain_clicked = st.button("Explain Prediction")
    risk_clicked = st.button("Risk Assessment")


st.header("Trajectory Data Ingestion")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Upload CSV")
    uploaded = st.file_uploader("Upload trajectory CSV", type=["csv"])
    if uploaded is not None:
        try:
            sequence = parse_csv_file(uploaded)
            st.session_state["sequence"] = sequence
            st.success(f"Loaded {len(sequence)} records from CSV.")
        except Exception as err:  # pylint: disable=broad-except
            st.error(f"Failed to parse CSV: {err}")

with col2:
    st.subheader("Paste JSON Sequence")
    json_input = st.text_area("Enter JSON list of dicts", height=200, key="json_input")
    if st.button("Load JSON Sequence"):
        try:
            sequence = parse_json_sequence(json_input)
            st.session_state["sequence"] = sequence
            st.success(f"Loaded {len(sequence)} records from JSON.")
        except Exception as err:  # pylint: disable=broad-except
            st.error(f"Invalid JSON sequence: {err}")


if st.session_state["sequence"]:
    st.markdown("### Parsed Sequence Preview")
    st.dataframe(pd.DataFrame(st.session_state["sequence"]))
else:
    st.info("Upload a CSV or paste JSON to get started.")


sequence = st.session_state["sequence"]


def handle_prediction():
    if not sequence:
        st.warning("Please load a sequence before predicting.")
        return
    with st.spinner("Running trajectory prediction..."):
        prediction = call_api("/predict", {"sequence": sequence})
    st.session_state["prediction"] = prediction


def handle_explanation():
    prediction = st.session_state.get("prediction")
    if not prediction:
        st.warning("Run a prediction before requesting an explanation.")
        return
    metadata = sequence[-1] if sequence else None
    with st.spinner("Generating explanation..."):
        response = call_api("/explain", {"prediction": prediction, "metadata": metadata})
    st.session_state["explanation"] = response


def handle_risk():
    prediction = st.session_state.get("prediction")
    if not prediction:
        st.warning("Run a prediction before assessing risk.")
        return
    metadata = sequence[-1] if sequence else None
    with st.spinner("Running risk assessment..."):
        response = call_api("/risk", {"prediction": prediction, "metadata": metadata})
    st.session_state["risk"] = response


st.markdown("---")
st.header("Model Operations")

if predict_clicked:
    handle_prediction()

prediction = st.session_state.get("prediction")
if prediction:
    st.subheader("Latest Prediction")
    st.success(
        f"Predicted Local_X: {prediction['predicted_local_x']:.4f} | "
        f"Local_Y: {prediction['predicted_local_y']:.4f}"
    )
    col_plot, col_json = st.columns(2)
    with col_plot:
        st.pyplot(plot_trajectory(sequence, prediction))
    with col_json:
        st.json(prediction)

if explain_clicked:
    handle_explanation()

if st.session_state.get("explanation"):
    st.subheader("Explanation")
    explanation_text = st.session_state["explanation"]["explanation"]
    paragraphs = [para.strip() for para in explanation_text.split("\n") if para.strip()]
    formatted = "".join(f"<p>{para}</p>" for para in paragraphs) or "<p>No explanation available.</p>"
    st.markdown(f'<div class="explanation-card">{formatted}</div>', unsafe_allow_html=True)

if risk_clicked:
    handle_risk()

risk = st.session_state.get("risk")
if risk:
    st.subheader("Risk Assessment")
    st.write(f"**Risk Score:** {risk['risk_score']:.2f}")
    st.write(f"**Factors:** {risk['risk_factors']}")
    st.write(f"**Recommendation:** {risk['recommendation']}")
    st.pyplot(plot_risk_meter(risk["risk_score"]))
