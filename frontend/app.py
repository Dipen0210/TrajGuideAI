"""
Modern Streamlit frontend for the Agentic Vehicle Trajectory Prediction System.
Features a sleek, dark-themed UI with gradient accents and card-based layout.
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


# Page configuration
st.set_page_config(
    page_title="TrajAI - Vehicle Trajectory Prediction", 
    layout="wide",
    page_icon="🚗",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern dark theme
st.markdown("""
<style>
    /* Import Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    /* Card styling */
    .metric-card {
        background: linear-gradient(145deg, #1e1e2e 0%, #2d2d44 100%);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.2);
    }
    
    .card-title {
        color: #a0aec0;
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    
    .card-value {
        color: #f7fafc;
        font-size: 2rem;
        font-weight: 700;
    }
    
    /* Status badges */
    .status-safe {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 50px;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        box-shadow: 0 4px 15px rgba(72, 187, 120, 0.4);
    }
    
    .status-warning {
        background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 50px;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        box-shadow: 0 4px 15px rgba(237, 137, 54, 0.4);
    }
    
    .status-critical {
        background: linear-gradient(135deg, #f56565 0%, #e53e3e 100%);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 50px;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        box-shadow: 0 4px 15px rgba(245, 101, 101, 0.4);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #4a5568 0%, transparent 100%);
        padding: 0.75rem 1.25rem;
        border-radius: 8px;
        margin: 1.5rem 0 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    .section-header h2 {
        color: #f7fafc;
        font-size: 1.25rem;
        font-weight: 600;
        margin: 0;
    }
    
    /* Agent cards */
    .agent-card {
        background: linear-gradient(145deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid rgba(102, 126, 234, 0.2);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        height: 100%;
    }
    
    .agent-card-safety {
        border-top: 4px solid #48bb78;
    }
    
    .agent-card-profiler {
        border-top: 4px solid #667eea;
    }
    
    .agent-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    .agent-title {
        color: #f7fafc;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .agent-description {
        color: #a0aec0;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* File uploader */
    .stFileUploader > div > div {
        background: linear-gradient(145deg, #1e1e2e 0%, #2d2d44 100%);
        border: 2px dashed rgba(102, 126, 234, 0.5);
        border-radius: 16px;
        padding: 2rem;
    }
    
    /* DataFrames */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, #2d2d44 0%, #1e1e2e 100%);
        border-radius: 8px;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Recommendation cards */
    .recommendation-item {
        background: rgba(102, 126, 234, 0.1);
        border-left: 3px solid #667eea;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Metrics styling */
    .metric-container {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .stat-box {
        background: rgba(102, 126, 234, 0.1);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        flex: 1;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    .stat-label {
        color: #a0aec0;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stat-value {
        color: #667eea;
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 0.25rem;
    }
    
    /* Violation tags */
    .violation-tag {
        background: linear-gradient(135deg, #f56565 0%, #e53e3e 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        display: inline-block;
        margin: 0.25rem;
    }
    
    /* Classification badges */
    .classification-aggressive {
        background: linear-gradient(135deg, #f56565 0%, #c53030 100%);
    }
    
    .classification-defensive {
        background: linear-gradient(135deg, #48bb78 0%, #276749 100%);
    }
    
    .classification-distracted {
        background: linear-gradient(135deg, #ecc94b 0%, #b7791f 100%);
    }
    
    .classification-normal {
        background: linear-gradient(135deg, #4299e1 0%, #2b6cb0 100%);
    }
</style>
""", unsafe_allow_html=True)


DEFAULT_BASE_URL = "http://localhost:8000"


def ensure_state() -> None:
    defaults = {
        "sequence": [],
        "prediction": None,
        "api_base": DEFAULT_BASE_URL,
        "safety_audit": None,
        "driver_profile": None,
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
    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h1 style="color: #667eea; font-size: 2rem; margin: 0;">🚗 TrajAI</h1>
        <p style="color: #a0aec0; font-size: 0.9rem; margin-top: 0.5rem;">
            Agentic Vehicle Intelligence
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### ⚙️ Configuration")
    st.session_state["api_base"] = st.text_input(
        "API Endpoint",
        value=st.session_state["api_base"],
        help="Backend API URL"
    )
    
    st.markdown("---")
    
    st.markdown("### 📊 Quick Stats")
    if st.session_state["sequence"]:
        seq = st.session_state["sequence"]
        st.metric("Data Points", len(seq))
        if len(seq) > 0:
            avg_vel = sum(s.get("v_Vel", 0) for s in seq) / len(seq)
            st.metric("Avg Velocity", f"{avg_vel:.2f} m/s")
    else:
        st.info("No data loaded")
    
    st.markdown("---")
    
    # Reset button
    if st.button("🔄 Reset All", use_container_width=True, help="Clear all data and start fresh"):
        # Clear all session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    st.markdown("""
    <div style="text-align: center; color: #4a5568; font-size: 0.75rem; padding: 1rem 0;">
        Powered by LSTM + Llama 3<br/>
        via Cerebras Inference
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# MAIN CONTENT
# ============================================================================

# Header
st.markdown("""
<div class="main-header">
    <h1>🚗 Vehicle Trajectory Intelligence</h1>
    <p>AI-powered trajectory prediction with real-time safety analysis and driver profiling</p>
</div>
""", unsafe_allow_html=True)

# Data Upload Section
st.markdown("""
<div class="section-header">
    <h2>📁 Data Upload</h2>
</div>
""", unsafe_allow_html=True)

col_upload, col_preview = st.columns([1, 2])

with col_upload:
    uploaded = st.file_uploader(
        "Upload vehicle telemetry CSV",
        type=["csv"],
        help="Upload a CSV file containing vehicle trajectory data"
    )
    
    if uploaded is not None:
        try:
            sequence = parse_csv_file(uploaded)
            st.session_state["sequence"] = sequence
            st.success(f"✅ Loaded **{len(sequence)}** data points")
        except Exception as err:
            st.error(f"❌ Failed to parse: {err}")

with col_preview:
    if st.session_state["sequence"]:
        st.markdown("**Data Preview**")
        df = pd.DataFrame(st.session_state["sequence"])
        st.dataframe(df.head(10), use_container_width=True, height=300)
    else:
        st.markdown("""
        <div style="background: rgba(102, 126, 234, 0.1); 
                    border-radius: 12px; 
                    padding: 3rem; 
                    text-align: center;
                    border: 2px dashed rgba(102, 126, 234, 0.3);">
            <p style="color: #667eea; font-size: 1.5rem; margin: 0;">📤</p>
            <p style="color: #a0aec0; margin-top: 0.5rem;">
                Upload a CSV file to get started
            </p>
        </div>
        """, unsafe_allow_html=True)


# Prediction Section
if st.session_state["sequence"]:
    st.markdown("""
    <div class="section-header">
        <h2>🎯 Trajectory Prediction</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col_btn, col_result = st.columns([1, 3])
    
    with col_btn:
        if st.button("🔮 Predict Next Steps", use_container_width=True):
            with st.spinner("Running LSTM prediction..."):
                try:
                    prediction = call_api("/predict", {"sequence": st.session_state["sequence"]})
                    st.session_state["prediction"] = prediction
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
    
    with col_result:
        prediction = st.session_state.get("prediction")
        if prediction and "trajectory" in prediction:
            traj = prediction["trajectory"]
            next_pt = traj[0]
            
            st.markdown(f"""
            <div style="display: flex; gap: 1rem;">
                <div class="stat-box">
                    <div class="stat-label">Next X Position</div>
                    <div class="stat-value">{next_pt.get('predicted_local_x', 0):.2f}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Next Y Position</div>
                    <div class="stat-value">{next_pt.get('predicted_local_y', 0):.2f}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Prediction Steps</div>
                    <div class="stat-value">{len(traj)}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Show trajectory plot if prediction exists
    if st.session_state.get("prediction"):
        with st.expander("📈 View Trajectory Plot", expanded=True):
            col1, col2 = st.columns([2, 1])
            with col1:
                fig = plot_trajectory(st.session_state["sequence"], st.session_state["prediction"])
                st.pyplot(fig)
            with col2:
                st.markdown("**Forecast Data**")
                traj = st.session_state["prediction"]["trajectory"]
                table_data = []
                for i, pt in enumerate(traj):
                    table_data.append({
                        "Step": f"+{i+1}",
                        "X": f"{pt.get('predicted_local_x', 0):.3f}",
                        "Y": f"{pt.get('predicted_local_y', 0):.3f}"
                    })
                st.dataframe(pd.DataFrame(table_data), use_container_width=True)


# AI Agents Section
st.markdown("""
<div class="section-header">
    <h2>🤖 AI Agent Analysis</h2>
</div>
""", unsafe_allow_html=True)

col_safety, col_profiler = st.columns(2)

# Safety Auditor Card
with col_safety:
    st.markdown("""
    <div class="agent-card agent-card-safety">
        <div class="agent-icon">🛡️</div>
        <div class="agent-title">Safety Auditor</div>
        <div class="agent-description">
            Real-time trajectory analysis for safety violations using Chain-of-Thought reasoning
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🔍 Run Safety Audit", key="safety_btn", use_container_width=True):
        if not st.session_state["sequence"]:
            st.warning("Please upload data first")
        else:
            with st.spinner("🛡️ Analyzing trajectory safety..."):
                try:
                    prediction = call_api("/predict", {"sequence": st.session_state["sequence"]})
                    predicted_trajectory = prediction.get("trajectory", [])
                    response = call_api("/agent/safety-audit", {
                        "sequence": st.session_state["sequence"],
                        "predicted_trajectory": predicted_trajectory
                    })
                    st.session_state["safety_audit"] = response
                    st.session_state["prediction"] = prediction
                except Exception as e:
                    st.error(f"Audit failed: {e}")
    
    # Display safety audit results
    audit_res = st.session_state.get("safety_audit")
    if audit_res:
        status = audit_res.get("status", "UNKNOWN").upper()
        violations = audit_res.get("violations", [])
        report = audit_res.get("report", "")
        
        # Filter out generic/empty violations
        violations = [v for v in violations if v and len(v) > 2 and v.lower() not in ["none", "n/a", "safe"]]
        
        # Status badge with custom styling
        if status == "SAFE":
            st.markdown("""
            <div class="status-safe">✅ SAFE - No violations detected</div>
            """, unsafe_allow_html=True)
        elif status == "WARNING":
            st.markdown("""
            <div class="status-warning">⚠️ WARNING - Potential issues detected</div>
            """, unsafe_allow_html=True)
        elif status == "CRITICAL":
            st.markdown("""
            <div class="status-critical">🚨 CRITICAL - Immediate action required</div>
            """, unsafe_allow_html=True)
        else:
            st.info(f"Status: {status}")
        
        # Only show violations if status is not SAFE and there are actual violations
        if status != "SAFE" and violations:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Detected Violations:**")
            for v in violations:
                st.markdown(f'<span class="violation-tag">❌ {v}</span>', unsafe_allow_html=True)
        
        with st.expander("📋 View Full Report"):
            st.markdown(report)


# Driver Profiler Card
with col_profiler:
    st.markdown("""
    <div class="agent-card agent-card-profiler">
        <div class="agent-icon">🏎️</div>
        <div class="agent-title">Driver Profiler</div>
        <div class="agent-description">
            Behavioral analysis and style classification with personalized recommendations
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("📊 Analyze Driver Style", key="profile_btn", use_container_width=True):
        if not st.session_state["sequence"]:
            st.warning("Please upload data first")
        else:
            with st.spinner("🏎️ Profiling driving behavior..."):
                try:
                    prediction = call_api("/predict", {"sequence": st.session_state["sequence"]})
                    predicted_trajectory = prediction.get("trajectory", [])
                    response = call_api("/agent/driver-profile", {
                        "sequence": st.session_state["sequence"],
                        "predicted_trajectory": predicted_trajectory
                    })
                    st.session_state["driver_profile"] = response
                    st.session_state["prediction"] = prediction
                except Exception as e:
                    st.error(f"Profiling failed: {e}")
    
    # Display driver profile results
    prof_res = st.session_state.get("driver_profile")
    if prof_res:
        classification = prof_res.get("classification", "Unknown")
        confidence = prof_res.get("confidence", 0)
        recommendations = prof_res.get("recommendations", [])
        report = prof_res.get("report", "")
        
        # Classification badge
        class_styles = {
            "Aggressive": ("🔴", "classification-aggressive", "#f56565"),
            "Defensive": ("🟢", "classification-defensive", "#48bb78"),
            "Distracted": ("🟡", "classification-distracted", "#ecc94b"),
            "Normal": ("🔵", "classification-normal", "#4299e1"),
        }
        emoji, css_class, color = class_styles.get(classification, ("⚪", "", "#a0aec0"))
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {color}33 0%, {color}11 100%);
                    border: 2px solid {color};
                    border-radius: 12px;
                    padding: 1rem;
                    text-align: center;
                    margin: 1rem 0;">
            <span style="font-size: 2rem;">{emoji}</span>
            <h3 style="color: {color}; margin: 0.5rem 0;">{classification}</h3>
            <p style="color: #a0aec0; margin: 0;">{confidence}% Confidence</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Confidence bar
        st.progress(confidence / 100)
        
        # Recommendations
        if recommendations:
            st.markdown("**💡 Recommendations:**")
            for i, rec in enumerate(recommendations[:3]):
                st.markdown(f"""
                <div class="recommendation-item">
                    {i+1}. {rec}
                </div>
                """, unsafe_allow_html=True)
        
        with st.expander("📋 View Full Report"):
            st.markdown(report)


# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem 0; color: #4a5568;">
    <p style="font-size: 0.9rem; margin: 0;">
        <strong>TrajAI</strong> - Agentic Vehicle Trajectory Prediction System v2.0
    </p>
    <p style="font-size: 0.75rem; margin-top: 0.5rem;">
        Powered by LSTM Deep Learning • Llama 3.3 70B • Cerebras Inference
    </p>
</div>
""", unsafe_allow_html=True)
