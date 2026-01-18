"""
Driver Style Profiler Agent

This agent analyzes driving patterns to classify driver behavior as
"Aggressive", "Cautious/Defensive", "Distracted", or "Normal" by comparing
against a "Golden Driver" standard stored in the RAG knowledge base.

Workflow:
1. ANALYZE: Process velocity/acceleration/lane change data
2. RETRIEVE: Query RAG for driver style benchmarks
3. CLASSIFY: Label the driving style with confidence
4. EXPLAIN: Generate natural language explanation with improvement suggestions
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from agent.llm.llama3_client import load_llama3
from agent.tools.profile_tool import analyze_driving_profile
from agent.tools.safety_rules_tool import consult_safety_rules


DRIVER_CHAIN_STEPS = [
    "ANALYZE: Run analyze_driving_profile to compute metrics (speed, acceleration variance, braking, jerk).",
    "RETRIEVE: Use consult_safety_rules to pull driver-style benchmarks and safety thresholds.",
    "CLASSIFY: Map metrics to Aggressive/Defensive/Distracted/Normal with a confidence score.",
    "EXPLAIN: Provide structured report and recommendations.",
    "NOTE: Do NOT generate new trajectory predictions; use the supplied LSTM output context (speed trend, accel) to support your analysis.",
]

DRIVER_PROFILER_PREFIX = """You are an expert Driver Behavior Analyst specializing in driving style classification.
Your mission is to analyze driving patterns and provide actionable feedback to improve safety.

You MUST follow this 4-step workflow for EVERY driver profiling request:

## Step 1: ANALYZE
Use the `analyze_driving_profile` tool to compute statistical metrics from the driving data.
Key metrics to examine:
- Mean velocity: Overall speed tendency
- Std acceleration: Smoothness of driving (low = smooth, high = jerky)
- Max braking: Intensity of hardest braking event
- Avg jerk: Rate of change of acceleration (comfort indicator)

## Step 2: RETRIEVE
Use the `consult_safety_rules` tool to query for driver style benchmarks.
Good queries:
- "What are the metrics for an aggressive driver?"
- "What defines a defensive/cautious driver?"
- "What indicates distracted driving behavior?"
- "What is the normal driver profile?"

## Step 3: CLASSIFY
Compare the ANALYZED metrics against RETRIEVED benchmarks:

| Style | Std Acceleration | Max Braking | Headway | Lane Changes |
|-------|-----------------|-------------|---------|--------------|
| Aggressive | > 1.5 m/s² | < -4.0 m/s² | < 1.5s | Frequent |
| Defensive | < 0.5 m/s² | > -2.0 m/s² | > 2.5s | Rare |
| Distracted | Variable | Delayed | Inconsistent | Erratic |
| Normal | 0.5-1.5 m/s² | -2.0 to -4.0 | 1.5-2.5s | Occasional |

Assign a CONFIDENCE score (0-100%) based on how clearly the data matches a profile.

## Step 4: EXPLAIN
Provide a structured driver profile report:

```
DRIVER PROFILE REPORT
=====================
Classification: [Aggressive | Defensive | Distracted | Normal]
Confidence: [X]%

Metrics Summary:
- Mean Speed: [X] m/s
- Acceleration Variability: [X] m/s² (std dev)
- Hardest Braking: [X] m/s²
- Jerk Index: [X]

Benchmark Comparison:
- [How this driver compares to the "Golden Driver" standard]

Improvement Recommendations:
1. [Specific actionable suggestion]
2. [Another suggestion if applicable]

Risk Assessment:
- [Overall risk level and explanation]
```

IMPORTANT: Always use BOTH tools before making a classification.
Be constructive in recommendations - the goal is to help drivers improve.

You will be given a precomputed trajectory prediction from the LSTM which contains predicted velocity and acceleration. Use this to see if the driver is trending towards safer or more dangerous behavior in the immediate future.
"""


def _build_profiler_tools() -> list:
    """Return tools used by the Driver Profiler agent."""
    return [
        analyze_driving_profile,
        consult_safety_rules,  # For retrieving benchmarks
    ]


class DriverProfileSchema(BaseModel):
    classification: str = Field(description="Driver style label: Aggressive, Defensive, Distracted, or Normal")
    confidence: int = Field(description="Confidence 0-100")
    recommendations: List[str] = Field(default_factory=list, description="Top improvement recommendations")
    report: str = Field(description="Full profile report")


class DriverProfilerAgent:
    """
    Dedicated agent for analyzing and classifying driver behavior.
    """
    
    def __init__(self):
        self.llm = load_llama3()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", DRIVER_PROFILER_PREFIX + """

IMPORTANT: You MUST respond with a valid JSON object containing exactly these fields:
{{
    "classification": "Aggressive" or "Defensive" or "Distracted" or "Normal",
    "confidence": 0-100 (integer),
    "recommendations": ["List of improvement suggestions"],
    "report": "Your full profile report here"
}}

Do NOT include any text before or after the JSON object. Only output valid JSON."""),
            (
                "human",
                "Metrics summary:\n{metrics_summary}\n\n"
                "Benchmarks:\n{benchmark_context}\n\n"
                "Supplied trajectory prediction (context):\n{prediction_summary}\n\n"
                "Raw data preview:\n{raw_data}\n\n"
                "Follow the 4-step workflow and provide classification, confidence, and recommendations as JSON.",
            ),
        ])
    
    def profile(
        self, 
        velocity_series: List[float],
        acceleration_series: List[float],
        lane_changes: Optional[int] = None,
        headway_series: Optional[List[float]] = None,
        predicted_trajectory: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze driving behavior and classify the driver style.
        
        Args:
            velocity_series: List of velocity values (m/s)
            acceleration_series: List of acceleration values (m/s²)
            lane_changes: Optional count of lane changes during the session
            headway_series: Optional list of headway values
            predicted_trajectory: Optional precomputed LSTM predictions for context (no re-prediction)
        
        Returns:
            Dictionary containing:
                - classification: Driver style label
                - confidence: Classification confidence (0-100)
                - report: Full profile report
                - recommendations: List of improvement suggestions
        """
        metrics = analyze_driving_profile.invoke({
            "velocity_series": velocity_series,
            "acceleration_series": acceleration_series,
            "lane_changes": lane_changes or 0,
            "headway_series": headway_series or [],
        })
        metrics_summary = str(metrics)
        
        benchmark_resp = consult_safety_rules.invoke({
            "query": "Provide driver style benchmarks for aggressive, defensive, distracted, and normal driving."
        })
        benchmark_context = (
            benchmark_resp.get("answer") if isinstance(benchmark_resp, dict) else str(benchmark_resp)
        )
        
        raw_data_preview = (
            f"- Velocities: {velocity_series[:10]}... (showing first 10)\n"
            f"- Accelerations: {acceleration_series[:10]}... (showing first 10)"
        )
        
        # Format prompt
        messages = self.prompt.format_messages(
            metrics_summary=metrics_summary,
            benchmark_context=benchmark_context,
            prediction_summary=self._format_prediction_summary(predicted_trajectory),
            raw_data=raw_data_preview,
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
        
        result["recommendations"] = (result.get("recommendations") or [])[:5]
        result["chain_steps"] = DRIVER_CHAIN_STEPS
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
        
        # Fallback: use the existing parsing method
        return self._parse_profile_result(response_text)
    
    def profile_from_sequence(
        self,
        vehicle_sequence: List[Dict[str, Any]],
        predicted_trajectory: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Profile driver from a full vehicle state sequence.
        
        Args:
            vehicle_sequence: List of vehicle state dictionaries
            predicted_trajectory: Optional precomputed prediction for context
        
        Returns:
            Driver profile result dictionary
        """
        velocities = [s.get("v_Vel", 0) for s in vehicle_sequence]
        accelerations = [s.get("v_Acc", 0) for s in vehicle_sequence]
        headways = [s.get("Space_Headway", 0) for s in vehicle_sequence]
        
        # Detect lane changes from i_l and i_r indicators
        lane_changes = self._count_lane_changes(vehicle_sequence)
        
        return self.profile(
            velocity_series=velocities,
            acceleration_series=accelerations,
            lane_changes=lane_changes,
            headway_series=headways,
            predicted_trajectory=predicted_trajectory,
        )
    
    def _count_lane_changes(self, sequence: List[Dict[str, Any]]) -> int:
        """Count lane change events from lane indicators."""
        changes = 0
        prev_lane_state = None
        
        for state in sequence:
            # i_l = 1 means vehicle is changing to left, i_r = 1 means right
            current_lane_state = (state.get("i_l", 0), state.get("i_r", 0))
            
            if prev_lane_state is not None:
                # Detect transition from no change to change
                if current_lane_state != (0, 0) and prev_lane_state == (0, 0):
                    changes += 1
            
            prev_lane_state = current_lane_state
        
        return changes
    
    def _parse_profile_result(self, output: str) -> Dict[str, Any]:
        """Parse agent output into structured result."""
        import re
        output_lower = output.lower()
        
        # Look for explicit classification patterns first
        # Pattern: "Classification: Aggressive" or "classified as: Normal"
        classification_patterns = [
            r'classification[:\s]+["\']*(\w+)',
            r'classified\s+as[:\s]+["\']*(\w+)',
            r'driver\s+(?:is|style)[:\s]+["\']*(\w+)',
            r'profile[:\s]+["\']*(\w+)',
        ]
        
        classification = None
        for pattern in classification_patterns:
            match = re.search(pattern, output_lower)
            if match:
                style = match.group(1).lower()
                if "aggress" in style:
                    classification = "Aggressive"
                elif "defens" in style or "cautious" in style:
                    classification = "Defensive"
                elif "distract" in style or "inattent" in style:
                    classification = "Distracted"
                elif "normal" in style:
                    classification = "Normal"
                if classification:
                    break
        
        # If no explicit pattern found, use metric-based heuristics
        if not classification:
            # Check for metric indicators in the text
            # High std acceleration = aggressive, low = defensive
            if any(phrase in output_lower for phrase in ["high acceleration variability", "hard braking", "aggressive acceleration", "erratic"]):
                classification = "Aggressive"
            elif any(phrase in output_lower for phrase in ["smooth", "gentle", "careful", "safe following", "low variability"]):
                classification = "Defensive"
            elif any(phrase in output_lower for phrase in ["inconsistent", "delayed reaction", "distracted"]):
                classification = "Distracted"
            else:
                classification = "Normal"  # Default to Normal, not Aggressive
        
        # Extract confidence (look for percentage)
        confidence_match = re.search(r'(\d+)\s*%', output)
        confidence = int(confidence_match.group(1)) if confidence_match else 70
        
        # Extract recommendations
        recommendations = []
        rec_keywords = ["recommend", "suggest", "should", "improve", "consider"]
        lines = output.split('\n')
        for line in lines:
            line_lower = line.lower()
            if any(kw in line_lower for kw in rec_keywords):
                # Clean up the line
                clean_line = line.strip().lstrip('-').lstrip('•').lstrip('1234567890.').strip()
                if clean_line and len(clean_line) > 10:
                    recommendations.append(clean_line)
        
        return {
            "classification": classification,
            "confidence": confidence,
            "report": output,
            "recommendations": recommendations[:5],  # Limit to 5
            "chain_steps": DRIVER_CHAIN_STEPS,
        }
    
    def clear_memory(self):
        """Clear conversation history."""
        return None

    @staticmethod
    def _format_prediction_summary(predicted: Optional[List[Dict[str, Any]]]) -> str:
        """Summarize supplied predicted trajectory for context."""
        if not predicted:
            return "No trajectory prediction provided."
        
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
_DRIVER_PROFILER = None


def get_driver_profiler() -> DriverProfilerAgent:
    """Get or create the Driver Profiler agent singleton."""
    global _DRIVER_PROFILER
    if _DRIVER_PROFILER is None:
        _DRIVER_PROFILER = DriverProfilerAgent()
    return _DRIVER_PROFILER


def run_driver_profile(
    velocity_series: List[float],
    acceleration_series: List[float],
    lane_changes: Optional[int] = None,
    headway_series: Optional[List[float]] = None,
    predicted_trajectory: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Convenience function to run driver profiling.
    
    Args:
        velocity_series: List of velocity values (m/s)
        acceleration_series: List of acceleration values (m/s²)
        lane_changes: Optional count of lane changes
        headway_series: Optional list of headway values
    
    Returns:
        Driver profile result dictionary
    """
    return get_driver_profiler().profile(
        velocity_series=velocity_series,
        acceleration_series=acceleration_series,
        lane_changes=lane_changes,
        headway_series=headway_series,
        predicted_trajectory=predicted_trajectory,
    )


def run_driver_profile_from_sequence(
    vehicle_sequence: List[Dict[str, Any]],
    predicted_trajectory: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Convenience function to profile from a full vehicle sequence.
    
    Args:
        vehicle_sequence: List of vehicle state dictionaries
    
    Returns:
        Driver profile result dictionary
    """
    return get_driver_profiler().profile_from_sequence(
        vehicle_sequence,
        predicted_trajectory=predicted_trajectory,
    )


if __name__ == "__main__":
    # Example usage with aggressive driving data
    import numpy as np
    
    # Simulate aggressive driving: high speed variance, hard braking
    np.random.seed(42)
    velocities = list(np.random.uniform(20, 35, 50))  # High speed variance
    accelerations = list(np.random.uniform(-5, 3, 50))  # Hard braking events
    
    result = run_driver_profile(
        velocity_series=velocities,
        acceleration_series=accelerations,
        lane_changes=5,
    )
    
    print(f"Classification: {result['classification']}")
    print(f"Confidence: {result['confidence']}%")
    print(f"\nRecommendations:")
    for rec in result['recommendations']:
        print(f"  - {rec}")
    print(f"\nFull Report:\n{result['report']}")
