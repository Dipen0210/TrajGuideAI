"""
LangChain agent pipeline orchestrating trajectory tools and Llama 3 reasoning.
"""

from __future__ import annotations

from langchain.agents import AgentType, Tool, initialize_agent
from langchain.memory import ConversationBufferMemory

from agent.llm.llama3_client import load_llama3
from agent.tools.context_query_tool import context_query
from agent.tools.explanation_tool import explain_trajectory
from agent.tools.prediction_tool import predict_trajectory
from agent.tools.risk_assessment_tool import trajectory_risk_assessment


def _build_tools() -> list[Tool]:
    """
    Wrap tool functions with LangChain Tool objects and helpful descriptions.
    """

    return [
        Tool(
            name=predict_trajectory.name,
            func=predict_trajectory.func,
            description="Predict the next (Local_X, Local_Y) for a vehicle trajectory window.",
            args_schema=predict_trajectory.args_schema,
        ),
        Tool(
            name=explain_trajectory.name,
            func=explain_trajectory.func,
            description="Explain the predicted motion using telemetry context.",
            args_schema=explain_trajectory.args_schema,
        ),
        Tool(
            name=trajectory_risk_assessment.name,
            func=trajectory_risk_assessment.func,
            description="Assess risk score, causal factors, and recommendations for a prediction.",
            args_schema=trajectory_risk_assessment.args_schema,
        ),
        Tool(
            name=context_query.name,
            func=context_query.func,
            description="Fetch contextual driving knowledge (placeholder RAG hook).",
            args_schema=context_query.args_schema,
        ),
    ]


_LLM = load_llama3()
_MEMORY = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
_TOOLS = _build_tools()

AGENT = initialize_agent(
    tools=_TOOLS,
    llm=_LLM,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=_MEMORY,
    handle_parsing_errors=True,
)


def run_agent(query: str) -> str:
    """
    Executes a natural-language query through the LangChain agent.
    Automatically routes to prediction, explanation, risk, or context tools.
    """

    return AGENT.run(query)


if __name__ == "__main__":
    sample_query = (
        "Given this sequence of vehicle states, predict the next trajectory point and explain the reasoning."
    )
    print(run_agent(sample_query))
