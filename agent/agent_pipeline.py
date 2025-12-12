"""
LangChain agent pipeline orchestrating trajectory tools and Llama 3 reasoning.
"""

from __future__ import annotations

from langchain.agents import initialize_agent
from langchain.memory import ConversationBufferMemory

from agent.llm.llama3_client import load_llama3
from agent.tools.prediction_tool import predict_trajectory

from agent.tools.safety_rules_tool import consult_safety_rules
from agent.tools.profile_tool import analyze_driving_profile


def _build_tools() -> list:
    """
    Return the structured LangChain tools used by the agent.
    """

    return [
        predict_trajectory,

        consult_safety_rules,
        analyze_driving_profile,

    ]


_LLM = load_llama3()
_MEMORY = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
_TOOLS = _build_tools()

PREFIX = """You are an expert Autonomous Driving Safety Auditor and Analyst.
Your goal is to ensure vehicle safety by predicting future behavior and cross-referencing it with traffic rules.

For SAFETY AUDIT requests, you MUST follow this strict 5-step "Chain of Thought":
1. OBSERVATION: Analyze the input vehicle state.
2. PREDICTION: Use the `predict_trajectory` tool to forecast the next 3 seconds.
3. RETRIEVAL: Use the `consult_safety_rules` tool to find rules relevant to the predicted speed (e.g., "safe following distance at 25m/s").
4. REASONING: Compare the PREDICTION (e.g., gap=10m) against the RETRIEVED RULE (e.g., required=50m).
5. OUTPUT: detailed violation warning or safety confirmation.
"""

AGENT = initialize_agent(
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
    Executes a natural-language query through the LangChain agent.
    Automatically routes to prediction, explanation, risk, or context tools.
    """

    result = AGENT.invoke({"input": query})
    return result["output"]


if __name__ == "__main__":
    sample_query = (
        "Given this sequence of vehicle states, predict the next trajectory point and explain the reasoning."
    )
    print(run_agent(sample_query))
