"""
LangChain agent pipeline orchestrating trajectory tools and Llama 3 reasoning.

This module provides:
1. SafetyAuditorAgent - Real-time safety violation detection
2. DriverProfilerAgent - Driving style classification
3. Unified orchestration for complex queries
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional

from langchain.agents import initialize_agent
from langchain.memory import ConversationBufferMemory

from agent.llm.llama3_client import load_llama3
from agent.tools.prediction_tool import predict_trajectory
from agent.tools.safety_rules_tool import consult_safety_rules
from agent.tools.profile_tool import analyze_driving_profile

# Import specialized agents
from agent.safety_auditor_agent import (
    SafetyAuditorAgent,
    get_safety_auditor,
    run_safety_audit,
)
from agent.driver_profiler_agent import (
    DriverProfilerAgent,
    get_driver_profiler,
    run_driver_profile,
    run_driver_profile_from_sequence,
)


def _build_tools() -> list:
    """
    Return the structured LangChain tools used by the unified agent.
    """
    return [
        predict_trajectory,
        consult_safety_rules,
        analyze_driving_profile,
    ]


_LLM = None
_MEMORY = None
_TOOLS = None
_AGENT = None


def _lazy_init():
    """Lazy initialization of the unified agent."""
    global _LLM, _MEMORY, _TOOLS, _AGENT
    
    if _AGENT is not None:
        return
    
    _LLM = load_llama3()
    _MEMORY = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    _TOOLS = _build_tools()
    
    PREFIX = """You are an expert Autonomous Driving Safety Auditor and Analyst.
Your goal is to ensure vehicle safety by predicting future behavior and cross-referencing it with traffic rules.

You have access to THREE specialized tools:

1. `predict_trajectory` - Predicts future vehicle positions using an LSTM model
2. `consult_safety_rules` - Retrieves traffic safety rules and thresholds from knowledge base
3. `analyze_driving_profile` - Analyzes driving behavior and classifies driver style

For SAFETY AUDIT requests, follow this 5-step Chain of Thought:
1. OBSERVATION: Analyze the input vehicle state.
2. PREDICTION: Use the `predict_trajectory` tool to forecast the next 3 seconds.
3. RETRIEVAL: Use the `consult_safety_rules` tool to find rules relevant to the predicted speed.
4. REASONING: Compare the PREDICTION against the RETRIEVED RULE.
5. OUTPUT: Provide a detailed violation warning or safety confirmation.

For DRIVER PROFILING requests, follow this 4-step workflow:
1. ANALYZE: Use `analyze_driving_profile` to compute statistical metrics.
2. RETRIEVE: Use `consult_safety_rules` to query for driver style benchmarks.
3. CLASSIFY: Label as "Aggressive", "Defensive", "Distracted", or "Normal".
4. EXPLAIN: Provide recommendations for improvement.

Always use the appropriate tools before making conclusions.
"""
    
    _AGENT = initialize_agent(
        tools=_TOOLS,
        llm=_LLM,
        agent="structured-chat-zero-shot-react-description",
        verbose=True,
        memory=_MEMORY,
        handle_parsing_errors=True,
        agent_kwargs={
            "prefix": PREFIX,
        },
    )


def run_agent(query: str) -> str:
    """
    Executes a natural-language query through the unified LangChain agent.
    Automatically routes to prediction, explanation, risk, or context tools.
    
    This is the general-purpose entry point. For specialized tasks, use:
    - run_safety_audit() for safety violation detection
    - run_driver_profile() for driving style classification
    """
    _lazy_init()
    result = _AGENT.invoke({"input": query})
    return result["output"]


# ============================================================================
# Convenience Functions for Specialized Agents
# ============================================================================

def audit_trajectory(vehicle_sequence: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Perform a safety audit on a vehicle trajectory sequence.
    
    Uses the SafetyAuditorAgent with specialized Chain of Thought reasoning.
    
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
    return run_safety_audit(vehicle_sequence)


def profile_driver(
    velocity_series: List[float],
    acceleration_series: List[float],
    lane_changes: Optional[int] = None,
    headway_series: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Analyze driving behavior and classify the driver style.
    
    Uses the DriverProfilerAgent with Golden Driver comparison.
    
    Args:
        velocity_series: List of velocity values (m/s)
        acceleration_series: List of acceleration values (m/s²)
        lane_changes: Optional count of lane changes
        headway_series: Optional list of headway values
    
    Returns:
        Dictionary containing:
            - classification: Driver style label
            - confidence: Classification confidence (0-100)
            - report: Full profile report
            - recommendations: List of improvement suggestions
    """
    return run_driver_profile(
        velocity_series=velocity_series,
        acceleration_series=acceleration_series,
        lane_changes=lane_changes,
        headway_series=headway_series,
    )


def profile_driver_from_sequence(vehicle_sequence: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Profile driver from a full vehicle state sequence.
    
    Convenience wrapper that extracts velocity, acceleration, and headway
    from the sequence and runs driver profiling.
    
    Args:
        vehicle_sequence: List of vehicle state dictionaries
    
    Returns:
        Driver profile result dictionary
    """
    return run_driver_profile_from_sequence(vehicle_sequence)


# ============================================================================
# Agent Access Functions
# ============================================================================

def get_agents() -> Dict[str, Any]:
    """
    Get references to both specialized agents.
    
    Returns:
        Dictionary with 'safety_auditor' and 'driver_profiler' keys.
    """
    return {
        "safety_auditor": get_safety_auditor(),
        "driver_profiler": get_driver_profiler(),
    }


# ============================================================================
# Re-export for backward compatibility
# ============================================================================

__all__ = [
    # Unified agent
    "run_agent",
    
    # Specialized agents
    "SafetyAuditorAgent",
    "DriverProfilerAgent",
    "get_safety_auditor",
    "get_driver_profiler",
    
    # Convenience functions
    "audit_trajectory",
    "profile_driver",
    "profile_driver_from_sequence",
    "run_safety_audit",
    "run_driver_profile",
    "run_driver_profile_from_sequence",
    "get_agents",
]


if __name__ == "__main__":
    # Example: Run a general query
    sample_query = (
        "Given this sequence of vehicle states, predict the next trajectory point and explain the reasoning."
    )
    print("=== Unified Agent ===")
    print(run_agent(sample_query))

