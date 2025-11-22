"""
LangChain agent pipeline orchestrating trajectory tools and Llama 3 reasoning.
"""

from __future__ import annotations

from langchain.agents import initialize_agent
from langchain.memory import ConversationBufferMemory

from agent.llm.llama3_client import load_llama3
from agent.tools.context_query_tool import context_query
from agent.tools.explanation_tool import explain_trajectory_tool
from agent.tools.prediction_tool import predict_trajectory
from agent.tools.risk_assessment_tool import trajectory_risk_assessment_tool


def _build_tools() -> list:
    """
    Return the structured LangChain tools used by the agent.
    """

    return [
        predict_trajectory,
        explain_trajectory_tool,
        trajectory_risk_assessment_tool,
        context_query,
    ]


_LLM = load_llama3()
_MEMORY = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
_TOOLS = _build_tools()

AGENT = initialize_agent(
    tools=_TOOLS,
    llm=_LLM,
    agent="structured-chat-zero-shot-react-description",
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
