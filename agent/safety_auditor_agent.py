"""
Autonomous Safety Auditor Agent

This agent combines LSTM-based trajectory prediction with RAG-based safety rules
to detect traffic violations and safety hazards in real-time.

Workflow (Chain of Thought):
1. OBSERVATION: Analyze input vehicle state
2. PREDICTION: Call predict_trajectory (LSTM) to forecast next 3 seconds
3. RETRIEVAL: Call consult_safety_rules (RAG) to find relevant safety thresholds
4. REASONING: Compare prediction against retrieved rules
5. OUTPUT: Safety confirmation OR violation warning with severity
"""

from __future__ import annotations

from typing import List, Dict, Any

from langchain.agents import initialize_agent
from langchain.memory import ConversationBufferMemory

from agent.llm.llama3_client import load_llama3
from agent.tools.prediction_tool import predict_trajectory
from agent.tools.safety_rules_tool import consult_safety_rules


SAFETY_AUDITOR_PREFIX = """You are an expert Autonomous Safety Auditor for vehicle trajectory analysis.
Your mission is to prevent accidents by detecting unsafe driving behaviors BEFORE they cause harm.

You MUST follow this strict 5-step Chain of Thought for EVERY safety audit request:

## Step 1: OBSERVATION
Analyze the input vehicle state sequence. Note key metrics:
- Current speed (v_Vel)
- Current headway (Space_Headway)
- Acceleration pattern (v_Acc)
- Lane position indicators (i_l, i_r, i_f)

## Step 2: PREDICTION
Use the `predict_trajectory` tool to forecast the vehicle's position for the next 3 seconds.
Extract the predicted trajectory coordinates and infer:
- Predicted speed trend
- Predicted gap to lead vehicle
- Predicted lateral movement

## Step 3: RETRIEVAL
Use the `consult_safety_rules` tool to query the knowledge base for relevant rules.
Good queries include:
- "What is the safe following distance at [X] m/s?"
- "What defines tailgating behavior?"
- "What is the collision risk threshold for TTC?"

## Step 4: REASONING
Compare PREDICTION results against RETRIEVED rules:
- Predicted gap vs Required safe distance
- Predicted TTC vs Collision warning threshold
- Predicted behavior vs Traffic rule definitions

## Step 5: OUTPUT
Provide a structured safety report:

```
SAFETY AUDIT REPORT
===================
Status: [✅ SAFE | ⚠️ WARNING | 🚨 CRITICAL VIOLATION]

Observation Summary:
- [Key metrics from input]

Prediction (Next 3s):
- [Predicted trajectory summary]

Rule Reference:
- [Retrieved safety rule]

Analysis:
- [Comparison of prediction vs rule]

Recommendation:
- [Action to take if violation detected]
```

IMPORTANT: You must ALWAYS use both tools before making a safety determination.
Never assume a situation is safe without checking the rules.
"""


def _build_safety_tools() -> list:
    """Return tools used by the Safety Auditor agent."""
    return [
        predict_trajectory,
        consult_safety_rules,
    ]


class SafetyAuditorAgent:
    """
    Dedicated agent for real-time safety auditing of vehicle trajectories.
    """
    
    def __init__(self):
        self.llm = load_llama3()
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True
        )
        self.tools = _build_safety_tools()
        
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent="structured-chat-zero-shot-react-description",
            verbose=True,
            memory=self.memory,
            handle_parsing_errors=True,
            agent_kwargs={
                "prefix": SAFETY_AUDITOR_PREFIX,
            },
        )
    
    def audit(self, vehicle_sequence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform a safety audit on a vehicle trajectory sequence.
        
        Args:
            vehicle_sequence: List of vehicle state dictionaries with keys:
                - Local_X, Local_Y: Position
                - v_Vel: Velocity
                - v_Acc: Acceleration
                - Space_Headway: Distance to lead vehicle
                - dis_cen, i_l, i_r, i_f, dis_l, dis_r, dis_f: Lane indicators
        
        Returns:
            Dictionary containing:
                - status: "SAFE", "WARNING", or "CRITICAL"
                - report: Full safety audit report
                - violations: List of detected violations
        """
        # Format the sequence for the agent
        query = f"""Perform a SAFETY AUDIT on the following vehicle trajectory sequence:

Vehicle States (last {len(vehicle_sequence)} timesteps):
{self._format_sequence_summary(vehicle_sequence)}

Follow the 5-step Chain of Thought to analyze this data and determine if there are any safety violations.
"""
        
        result = self.agent.invoke({"input": query})
        output = result["output"]
        
        # Parse the output to extract structured data
        return self._parse_audit_result(output)
    
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
    
    def _parse_audit_result(self, output: str) -> Dict[str, Any]:
        """Parse agent output into structured result."""
        status = "SAFE"
        violations = []
        
        output_lower = output.lower()
        
        if "critical" in output_lower or "🚨" in output:
            status = "CRITICAL"
        elif "warning" in output_lower or "⚠️" in output:
            status = "WARNING"
        elif "violation" in output_lower:
            status = "WARNING"
        
        # Extract violations mentioned
        violation_keywords = ["tailgating", "speeding", "collision", "unsafe", "violation"]
        for keyword in violation_keywords:
            if keyword in output_lower:
                violations.append(keyword)
        
        return {
            "status": status,
            "report": output,
            "violations": list(set(violations)),
        }
    
    def clear_memory(self):
        """Clear conversation history."""
        self.memory.clear()


# Module-level singleton for convenience
_SAFETY_AUDITOR = None


def get_safety_auditor() -> SafetyAuditorAgent:
    """Get or create the Safety Auditor agent singleton."""
    global _SAFETY_AUDITOR
    if _SAFETY_AUDITOR is None:
        _SAFETY_AUDITOR = SafetyAuditorAgent()
    return _SAFETY_AUDITOR


def run_safety_audit(vehicle_sequence: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convenience function to run a safety audit.
    
    Args:
        vehicle_sequence: List of vehicle state dictionaries
    
    Returns:
        Safety audit result dictionary
    """
    return get_safety_auditor().audit(vehicle_sequence)


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
