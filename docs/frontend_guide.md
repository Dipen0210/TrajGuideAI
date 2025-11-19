# Frontend Guide

## Layout
1. **Sidebar**
   - Project description
   - API base URL field
   - Action buttons: Predict, Explain, Risk, Ask Agent
2. **Data Ingestion Section**
   - CSV uploader (expects canonical columns)
   - JSON text area for manual entry
   - Parsed sequence preview with `st.dataframe`
3. **Model Outputs**
   - Prediction summary + chart (`plot_trajectory`)
   - Explanation card styled as a chat bubble
   - Risk assessment block with textual insights + `plot_risk_meter`
4. **Agent Chat**
   - Text box for free-form questions
   - Conversation history with alternating blue/gray message bubbles
   - RAG sources listed beneath responses

## Interaction Flow
1. Upload or paste a sequence.
2. Click **Predict Trajectory** to fill `st.session_state["prediction"]`.
3. Use **Explain Prediction** and **Risk Assessment** to call respective APIs (metadata derived from the latest sequence item).
4. Ask multi-turn questions via **Ask Agent**; the agent will plan tool calls and reference the RAG corpus when applicable.

## Visual Components
- `plot_trajectory`: Matplotlib line plot of Local_X vs Local_Y with a highlighted predicted point.
- `plot_risk_meter`: Semi-circular gauge showing the risk score.
- Streamlit columns create side-by-side visualizations and JSON dumps for clarity.

## Error Handling
- Upload/JSON parsing errors raise `st.error()` messages.
- API failures surface network or validation issues in red callouts.
- Loading spinners wrap longer API calls to maintain responsiveness.
