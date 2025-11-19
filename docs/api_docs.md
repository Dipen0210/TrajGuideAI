# API Documentation

Base URL defaults to `http://localhost:8000`. All endpoints accept/return JSON.

## Root
- `GET /`
- Returns `{ "message": "Agentic Vehicle Trajectory Prediction API is running." }`

## POST /predict
**Request Body**
```json
{
  "sequence": [
    {"Local_X": 12.0, "Local_Y": 1.8, "v_Vel": 15.2, "...": "..."},
    ...
  ]
}
```
**Response**
```json
{
  "predicted_local_x": 12.56,
  "predicted_local_y": 2.01
}
```
Errors return HTTP 400 with detail messages if the sequence is malformed or too short.

## POST /explain
**Request**
```json
{
  "prediction": {"predicted_local_x": 12.56, "predicted_local_y": 2.01},
  "metadata": {"v_Vel": 15.2, "v_Acc": 0.4, "Space_Headway": 6.3}
}
```
**Response**
```json
{
  "explanation": "The vehicle is likely ..."
}
```

## POST /risk
**Request**
```json
{
  "prediction": {"predicted_local_x": 12.56, "predicted_local_y": 2.01},
  "metadata": {"dis_l": 2.1, "dis_r": 1.7, "i_l": 0, "i_r": 1}
}
```
**Response**
```json
{
  "risk_score": 0.64,
  "risk_factors": "Close right vehicle, indicator mismatch",
  "recommendation": "Increase spacing..."
}
```

## POST /agent/query
**Request**
```json
{
  "query": "Given this sequence, should I expect a risky maneuver?"
}
```
**Response**
```json
{
  "response": "Based on the last prediction..."
}
```
Behind the scenes the agent may call prediction/explanation/risk/context tools, and the Streamlit UI surfaces any RAG sources referenced.
