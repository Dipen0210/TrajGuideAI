"""
Streamlit frontend for the Agentic Vehicle Trajectory Prediction System.
"""

from __future__ import annotations


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





DEFAULT_BASE_URL = "http://localhost:8000"


def ensure_state() -> None:
    defaults = {
        "sequence": [],
        "prediction": None,
        "prediction": None,
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



st.header("Trajectory Data Ingestion")
st.subheader("Upload CSV")
uploaded = st.file_uploader("Upload trajectory CSV", type=["csv"])
if uploaded is not None:
    try:
        sequence = parse_csv_file(uploaded)
        st.session_state["sequence"] = sequence
        st.success(f"Loaded {len(sequence)} records from CSV.")
    except Exception as err:  # pylint: disable=broad-except
        st.error(f"Failed to parse CSV: {err}")


if st.session_state["sequence"]:
    st.markdown("### Parsed Sequence Preview")
    st.dataframe(pd.DataFrame(st.session_state["sequence"]))
else:
    st.info("Upload a CSV to get started.")


sequence = st.session_state["sequence"]


def handle_prediction():
    if not sequence:
        st.warning("Please load a sequence before predicting.")
        return
    with st.spinner("Running trajectory prediction..."):
        prediction = call_api("/predict", {"sequence": sequence})
    st.session_state["prediction"] = prediction





st.markdown("---")
st.header("Model Operations")

if predict_clicked:
    handle_prediction()

prediction = st.session_state.get("prediction")
if prediction:
    if prediction and "trajectory" in prediction:
        traj = prediction["trajectory"]
        # Show stats for the first point (or last?)
        # Let's show the first predicted point as immediate next state
        next_pt = traj[0]
        st.subheader("Trajectory Forecast (Next 3 Steps)")
        st.success(
            f"Next Step -> X: {next_pt['predicted_local_x']:.4f} | "
            f"Y: {next_pt['predicted_local_y']:.4f}"
        )
        
        col_plot, col_table = st.columns(2)
        with col_plot:
            st.pyplot(plot_trajectory(sequence, prediction))
            
        with col_table:
             st.write("### Full Forecast Data")
             
             # Prepare Table Data: Last 3 history + 3 future
             history_subset = sequence[-3:] if len(sequence) >= 3 else sequence
             
             table_data = []
             for pt in history_subset:
                 table_data.append({
                     "Step": "History",
                     "Local_X": pt.get("Local_X"),
                     "Local_Y": pt.get("Local_Y")
                 })
                 
             for pt in traj:
                 table_data.append({
                     "Step": "Prediction",
                     "Local_X": pt.get("predicted_local_x"),
                     "Local_Y": pt.get("predicted_local_y")
                 })
                 
             st.dataframe(pd.DataFrame(table_data))


# --- Agent Operations ---

def handle_safety_audit():
    """Call the dedicated Safety Auditor endpoint."""
    sequence = st.session_state.get("sequence", [])
    if not sequence:
        st.warning("Please load a trajectory sequence first.")
        return
    
    with st.spinner("🏆 Safety Auditor Agent is analyzing trajectory..."):
        try:
            response = call_api("/agent/safety-audit", {"sequence": sequence})
            st.session_state["safety_audit"] = response
        except Exception as e:
            st.error(f"Safety audit failed: {e}")
            return


def handle_driver_profile():
    """Call the dedicated Driver Profiler endpoint."""
    sequence = st.session_state.get("sequence", [])
    if not sequence:
        st.warning("Please load a trajectory sequence first.")
        return
    
    with st.spinner("🏎️ Driver Profiler Agent is classifying behavior..."):
        try:
            response = call_api("/agent/driver-profile", {"sequence": sequence})
            st.session_state["driver_profile"] = response
        except Exception as e:
            st.error(f"Driver profiling failed: {e}")
            return


st.markdown("---")
st.header("🤖 Agent Analysis")

col_safe, col_prof = st.columns(2)

with col_safe:
    st.subheader("🏆 Safety Auditor")
    if st.button("Run Safety Audit", key="safety_btn"):
        handle_safety_audit()
    
    audit_res = st.session_state.get("safety_audit")
    if audit_res:
        # Display structured result
        status = audit_res.get("status", "UNKNOWN")
        violations = audit_res.get("violations", [])
        report = audit_res.get("report", "")
        
        # Status badge
        if status == "SAFE":
            st.success(f"✅ Status: **{status}**")
        elif status == "WARNING":
            st.warning(f"⚠️ Status: **{status}**")
        elif status == "CRITICAL":
            st.error(f"🚨 Status: **{status}**")
        else:
            st.info(f"Status: {status}")
        
        # Violations
        if violations:
            st.markdown("**Detected Violations:**")
            for v in violations:
                st.markdown(f"- ❌ {v}")
        
        # Full report in expander
        with st.expander("View Full Report"):
            st.markdown(report)

with col_prof:
    st.subheader("🏎️ Driver Profiler")
    if st.button("Analyze Driver Style", key="profile_btn"):
        handle_driver_profile()
    
    prof_res = st.session_state.get("driver_profile")
    if prof_res:
        # Display structured result
        classification = prof_res.get("classification", "Unknown")
        confidence = prof_res.get("confidence", 0)
        recommendations = prof_res.get("recommendations", [])
        report = prof_res.get("report", "")
        
        # Classification with color
        style_colors = {
            "Aggressive": "🔴",
            "Defensive": "🟢",
            "Distracted": "🟡",
            "Normal": "🔵",
        }
        emoji = style_colors.get(classification, "⚪")
        st.info(f"{emoji} Classification: **{classification}** ({confidence}% confidence)")
        
        # Progress bar for confidence
        st.progress(confidence / 100)
        
        # Recommendations
        if recommendations:
            st.markdown("**Recommendations:**")
            for rec in recommendations[:3]:  # Show top 3
                st.markdown(f"- 💡 {rec}")
        
        # Full report in expander
        with st.expander("View Full Report"):
            st.markdown(report)


# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <small>Agentic Vehicle Trajectory Prediction System v2.0 | 
        Powered by LSTM + Llama 3 + RAG</small>
    </div>
    """,
    unsafe_allow_html=True,
)

