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

from langchain.agents import initialize_agent
from langchain.memory import ConversationBufferMemory

from agent.llm.llama3_client import load_llama3
from agent.tools.profile_tool import analyze_driving_profile
from agent.tools.safety_rules_tool import consult_safety_rules


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
"""


def _build_profiler_tools() -> list:
    """Return tools used by the Driver Profiler agent."""
    return [
        analyze_driving_profile,
        consult_safety_rules,  # For retrieving benchmarks
    ]


class DriverProfilerAgent:
    """
    Dedicated agent for analyzing and classifying driver behavior.
    """
    
    def __init__(self):
        self.llm = load_llama3()
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.tools = _build_profiler_tools()
        
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent="structured-chat-zero-shot-react-description",
            verbose=True,
            memory=self.memory,
            handle_parsing_errors=True,
            agent_kwargs={
                "prefix": DRIVER_PROFILER_PREFIX,
            },
        )
    
    def profile(
        self, 
        velocity_series: List[float],
        acceleration_series: List[float],
        lane_changes: Optional[int] = None,
        headway_series: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze driving behavior and classify the driver style.
        
        Args:
            velocity_series: List of velocity values (m/s)
            acceleration_series: List of acceleration values (m/s²)
            lane_changes: Optional count of lane changes during the session
            headway_series: Optional list of headway values
        
        Returns:
            Dictionary containing:
                - classification: Driver style label
                - confidence: Classification confidence (0-100)
                - report: Full profile report
                - recommendations: List of improvement suggestions
        """
        # Build context string
        context_parts = [
            f"Velocity series ({len(velocity_series)} samples): min={min(velocity_series):.2f}, max={max(velocity_series):.2f}, mean={sum(velocity_series)/len(velocity_series):.2f} m/s",
            f"Acceleration series ({len(acceleration_series)} samples): min={min(acceleration_series):.2f}, max={max(acceleration_series):.2f}",
        ]
        
        if lane_changes is not None:
            context_parts.append(f"Lane changes detected: {lane_changes}")
        
        if headway_series:
            context_parts.append(
                f"Headway series: min={min(headway_series):.2f}m, avg={sum(headway_series)/len(headway_series):.2f}m"
            )
        
        context = "\n".join(context_parts)
        
        query = f"""Perform a DRIVER PROFILE analysis on the following driving session data:

{context}

Raw Data:
- Velocities: {velocity_series[:10]}... (showing first 10)
- Accelerations: {acceleration_series[:10]}... (showing first 10)

Follow the 4-step workflow to:
1. ANALYZE the driving metrics
2. RETRIEVE the driver style benchmarks
3. CLASSIFY the driving style
4. EXPLAIN with recommendations
"""
        
        result = self.agent.invoke({"input": query})
        output = result["output"]
        
        return self._parse_profile_result(output)
    
    def profile_from_sequence(self, vehicle_sequence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Profile driver from a full vehicle state sequence.
        
        Args:
            vehicle_sequence: List of vehicle state dictionaries
        
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
        )
    
    def profile_raw(self, query: str) -> str:
        """
        Run a raw natural language profiling query through the agent.
        """
        result = self.agent.invoke({"input": query})
        return result["output"]
    
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
        output_lower = output.lower()
        
        # Determine classification
        classification = "Normal"
        if "aggressive" in output_lower:
            classification = "Aggressive"
        elif "defensive" in output_lower or "cautious" in output_lower:
            classification = "Defensive"
        elif "distracted" in output_lower or "inattentive" in output_lower:
            classification = "Distracted"
        
        # Extract confidence (look for percentage)
        import re
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
        }
    
    def clear_memory(self):
        """Clear conversation history."""
        self.memory.clear()


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
    )


def run_driver_profile_from_sequence(vehicle_sequence: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convenience function to profile from a full vehicle sequence.
    
    Args:
        vehicle_sequence: List of vehicle state dictionaries
    
    Returns:
        Driver profile result dictionary
    """
    return get_driver_profiler().profile_from_sequence(vehicle_sequence)


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
