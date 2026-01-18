"""
Autonomous Safety Auditor Agent

This agent combines LSTM-based trajectory prediction with RAG-based safety rules
to detect traffic violations and safety hazards in real-time.

Workflow (Chain of Thought):
1. OBSERVATION: Analyze input vehicle state
2. PROVIDED PREDICTION: Use the supplied LSTM-predicted trajectory (do NOT re-predict)
3. RETRIEVAL: Call consult_safety_rules (RAG) to find relevant safety thresholds
4. REASONING: Compare supplied prediction against retrieved rules
5. OUTPUT: Safety confirmation OR violation warning with severity
"""

from __future__ import annotations

from typing import List, Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from agent.llm.llama3_client import load_llama3
from agent.tools.safety_rules_tool import consult_safety_rules


SAFETY_CHAIN_STEPS = [
    "OBSERVATION: Analyze the input vehicle state (position, velocity, acceleration, headway).",
    "PROVIDED PREDICTION: Analyze the LSTM-predicted trajectory, which now includes FUTURE velocity, acceleration, and lane position.",
    "RETRIEVAL: Call consult_safety_rules to fetch safety thresholds relevant to speed/headway.",
    "REASONING: Compare the provided prediction against retrieved rules (headway, speed limits, lane safety).",
    "OUTPUT: Summarize status [SAFE | WARNING | CRITICAL], primary violation, and a concise explanation.",
]

SAFETY_AUDITOR_PREFIX = f"""You are a rigorous Autonomous Driving Safety Auditor.
Your responsibility is to validate vehicle behavior against traffic laws and safety constraints.

You MUST follow this 5-step Chain of Thought for EVERY safety audit request (do not re-predict):

1. {SAFETY_CHAIN_STEPS[0]}
2. {SAFETY_CHAIN_STEPS[1]}
3. {SAFETY_CHAIN_STEPS[2]}
4. {SAFETY_CHAIN_STEPS[3]}
5. {SAFETY_CHAIN_STEPS[4]}

Inputs include:
- Raw vehicle sequence (state history).
- A precomputed predicted trajectory from the LSTM (fully populated with predicted dynamics).

IMPORTANT OUTPUT RULES:
- Populate the JSON fields precisely (status/report/violations).
- Do NOT restate a separate status line inside the report; the status is taken from the JSON field.
- Ensure the report narrative is consistent with the status value.
"""


class SafetyAuditSchema(BaseModel):
    status: str = Field(description="Safety status: SAFE, WARNING, or CRITICAL")
    report: str = Field(description="Detailed safety audit report")
    violations: List[str] = Field(default_factory=list, description="List of detected violations")


def _build_auditor_tools() -> list:
    """Return tools used by the Safety Auditor agent."""
    return [
        consult_safety_rules,
    ]


class SafetyAuditorAgent:
    """
    Dedicated agent for real-time safety auditing of vehicle trajectories.
    """
    
    def __init__(self):
        self.llm = load_llama3()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SAFETY_AUDITOR_PREFIX + """

IMPORTANT: You MUST respond with a valid JSON object containing exactly these fields:
{{
    "status": "SAFE" or "WARNING" or "CRITICAL",
    "report": "Your detailed analysis report here",
    "violations": ["list", "of", "violations"] or []
}}

Do NOT include any text before or after the JSON object. Only output valid JSON."""),
            (
                "human",
                "Sequence summary:\n{sequence_summary}\n\n"
                "Supplied LSTM prediction:\n{prediction_summary}\n\n"
                "Safety rules context:\n{rule_context}\n\n"
                "Provide the 5-step audit and final status as JSON.",
            ),
        ])
    
    def audit(
        self,
        vehicle_sequence: List[Dict[str, Any]],
        predicted_trajectory: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        """
        Perform a safety audit on a vehicle trajectory sequence.
        
        Args:
            vehicle_sequence: List of vehicle state dictionaries with keys:
                - Local_X, Local_Y: Position
                - v_Vel: Velocity
                - v_Acc: Acceleration
                - Space_Headway: Distance to lead vehicle
                - dis_cen, i_l, i_r, i_f, dis_l, dis_r, dis_f: Lane indicators
            predicted_trajectory: Precomputed LSTM predictions (list of {predicted_local_x, predicted_local_y})
        
        Returns:
            Dictionary containing:
                - status: "SAFE", "WARNING", or "CRITICAL"
                - report: Full safety audit report
                - violations: List of detected violations
        """
        if not predicted_trajectory:
            raise ValueError("predicted_trajectory is required for safety audit. Call /predict first and pass the result.")

        # Build rule context from retrieval tool (no LLM tool-calling).
        last_vel = vehicle_sequence[-1].get("v_Vel", 0) if vehicle_sequence else 0
        rule_resp = consult_safety_rules.invoke(
            {"query": f"safe following distance and speed limits for {last_vel:.2f} m/s"}
        )
        rule_context = rule_resp.get("answer") if isinstance(rule_resp, dict) else str(rule_resp)
        
        # Format prompt
        messages = self.prompt.format_messages(
            sequence_summary=self._format_sequence_summary(vehicle_sequence),
            prediction_summary=self._format_prediction_summary(predicted_trajectory),
            rule_context=rule_context,
        )
        
        # Invoke LLM
        response = self.llm.invoke(messages)
        
        # Parse response - handle both string and AIMessage responses
        if hasattr(response, 'content'):
            response_text = response.content
        else:
            response_text = str(response)
        
        # Try to extract JSON from the response
        result = self._parse_json_response(response_text)
        
        result["status"] = result.get("status", "SAFE").upper()
        result["chain_steps"] = SAFETY_CHAIN_STEPS
        
        # Extract violations heuristically if empty
        status = result.get("status", "SAFE").upper()
        if not result.get("violations"):
            result["violations"] = self._infer_violations(result.get("report", ""), status)
        
        return result
    
    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """Parse JSON from LLM response, with fallback handling."""
        import json
        import re
        
        # Try to find JSON in the response
        try:
            # First try to parse the whole response as JSON
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON object in the response
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Fallback: construct result from text analysis
        status = "SAFE"
        if "critical" in response_text.lower():
            status = "CRITICAL"
        elif "warning" in response_text.lower():
            status = "WARNING"
        
        return {
            "status": status,
            "report": response_text,
            "violations": self._infer_violations(response_text)
        }
    
    def audit_raw(self, query: str) -> str:
        """
        Run a raw natural language safety query through the agent.
        """
        result = self.agent.invoke({"input": query})
        return result["output"]
    
    def _format_sequence_summary(self, sequence: List[Dict[str, Any]]) -> str:
        """Format vehicle sequence into a readable summary."""
        if not sequence:
            return "No data provided"
        
        # Get key stats from the sequence
        velocities = [s.get("v_Vel", 0) for s in sequence]
        accelerations = [s.get("v_Acc", 0) for s in sequence]
        headways = [s.get("Space_Headway", 0) for s in sequence]
        
        summary = f"""
- Time Window: {len(sequence)} frames
- Velocity Range: {min(velocities):.2f} - {max(velocities):.2f} m/s
- Current Velocity: {velocities[-1]:.2f} m/s
- Acceleration Range: {min(accelerations):.2f} - {max(accelerations):.2f} m/s²
- Current Headway: {headways[-1]:.2f} m
- Min Headway: {min(headways):.2f} m

Raw sequence data:
{sequence}
"""
        return summary
    
    @staticmethod
    def _infer_violations(report: str, status: str = None) -> List[str]:
        """Heuristic to fill violations if model returned none."""
        import re
        
        # If status is SAFE, don't infer violations
        if status and status.upper() == "SAFE":
            return []
        
        output_lower = report.lower()
        
        # Look for explicit violation patterns, not just keywords
        violation_patterns = [
            (r'tailgating\s+(?:detected|violation|risk)', 'Tailgating'),
            (r'(?:is\s+)?speeding', 'Speeding'),
            (r'collision\s+risk\s+(?:is\s+)?high', 'High Collision Risk'),
            (r'unsafe\s+(?:behavior|maneuver|distance)', 'Unsafe Behavior'),
            (r'violation[:\s]+(\w+)', None),  # Extract specific violation
            (r'exceeds?\s+(?:speed\s+)?limit', 'Speed Limit Exceeded'),
            (r'insufficient\s+(?:headway|distance)', 'Insufficient Following Distance'),
        ]
        
        found = []
        for pattern, label in violation_patterns:
            match = re.search(pattern, output_lower)
            if match:
                if label:
                    found.append(label)
                elif match.group(1):
                    found.append(match.group(1).capitalize())
        
        # Avoid duplicates
        return list(set(found))
    
    def clear_memory(self):
        """No-op placeholder for backward compatibility."""
        return None

    @staticmethod
    def _format_prediction_summary(predicted: List[Dict[str, Any]] | None) -> str:
        """Format supplied predicted trajectory."""
        if not predicted:
            return "No prediction available."
        
        # Extract keys if available, otherwise default to 0
        # inference.py now returns keys matching feature columns (Local_X, v_Vel, etc)
        # But may also handle predicted_local_x for old compatibility if needed, though we moved to feature names.
        
        # We need to robustly handle whatever keys come back.
        # inference.py uses feature_columns names which are "Local_X", "v_Vel", etc.
        
        xs = [p.get("Local_X", p.get("predicted_local_x", 0)) for p in predicted]
        ys = [p.get("Local_Y", p.get("predicted_local_y", 0)) for p in predicted]
        vels = [p.get("v_Vel", 0) for p in predicted]
        accs = [p.get("v_Acc", 0) for p in predicted]
        
        return (
            f"- Horizon: {len(predicted)} steps\n"
            f"- Position: X({min(xs):.1f}->{max(xs):.1f}), Y({min(ys):.1f}->{max(ys):.1f})\n"
            f"- Predicted Velocity: {min(vels):.2f} -> {max(vels):.2f} m/s\n"
            f"- Predicted Accel: {min(accs):.2f} -> {max(accs):.2f} m/s²\n"
            f"- Raw Data: {predicted}"
        )


# Module-level singleton for convenience
_SAFETY_AUDITOR = None


def get_safety_auditor() -> SafetyAuditorAgent:
    """Get or create the Safety Auditor agent singleton."""
    global _SAFETY_AUDITOR
    if _SAFETY_AUDITOR is None:
        _SAFETY_AUDITOR = SafetyAuditorAgent()
    return _SAFETY_AUDITOR


def run_safety_audit(
    vehicle_sequence: List[Dict[str, Any]],
    predicted_trajectory: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """
    Convenience function to run a safety audit.
    
    Args:
        vehicle_sequence: List of vehicle state dictionaries
        predicted_trajectory: Precomputed LSTM predictions passed from UI/backend
    
    Returns:
        Safety audit result dictionary
    """
    return get_safety_auditor().audit(vehicle_sequence, predicted_trajectory)


if __name__ == "__main__":
    # Example usage
    sample_sequence = [
        {"Local_X": 10.0, "Local_Y": 100.0, "v_Vel": 25.0, "v_Acc": 0.5, 
         "Space_Headway": 15.0, "dis_cen": 0.0, "i_l": 0, "i_r": 0, "i_f": 1,
         "dis_l": 3.5, "dis_r": 3.5, "dis_f": 15.0},
        {"Local_X": 10.0, "Local_Y": 125.0, "v_Vel": 26.0, "v_Acc": 1.0,
         "Space_Headway": 12.0, "dis_cen": 0.0, "i_l": 0, "i_r": 0, "i_f": 1,
         "dis_l": 3.5, "dis_r": 3.5, "dis_f": 12.0},
        {"Local_X": 10.0, "Local_Y": 151.0, "v_Vel": 27.0, "v_Acc": 1.0,
         "Space_Headway": 10.0, "dis_cen": 0.0, "i_l": 0, "i_r": 0, "i_f": 1,
         "dis_l": 3.5, "dis_r": 3.5, "dis_f": 10.0},
    ]
    
    result = run_safety_audit(sample_sequence)
    print(f"Status: {result['status']}")
    print(f"Violations: {result['violations']}")
    print(f"\nFull Report:\n{result['report']}")
