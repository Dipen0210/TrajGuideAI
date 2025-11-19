# Agent System

## LangChain Tools
1. **Prediction Tool (`predict_trajectory`)**
   - Wraps `TrajectoryInference` to normalize inputs and return the next `(Local_X, Local_Y)` pair.
2. **Explanation Tool (`explain_trajectory`)**
   - Prompts Llama 3 to produce human-readable narratives referencing telemetry context.
3. **Risk Assessment Tool (`trajectory_risk_assessment`)**
   - Instructs Llama 3 to return JSON with `risk_score`, factors, and recommendations.
4. **Context Query Tool (`context_query`)**
   - Calls the RAG pipeline to retrieve supporting knowledge and surface metadata about the sources.

## Llama 3 Integration
- `agent/llm/llama3_client.py` defines a LangChain-compatible LLM wrapper that issues HTTP POST requests to a deployed Llama 3 endpoint using credentials stored in `llama3_config.yaml`.
- The same client is used by standalone tools (explanation, risk) and the agent.

## Agent Pipeline
- `agent_pipeline.py` initializes:
  - Tool list (LangChain `Tool` objects referencing the decorated functions).
  - Conversation memory (`ConversationBufferMemory` storing prior interactions).
  - ReAct-style agent via `initialize_agent(..., agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)`.
- The agent can plan which tool to call, observe outputs, and craft a final response referencing predictions, explanations, risk scores, or retrieved knowledge.

## Memory
- `ConversationBufferMemory` preserves the `chat_history`, enabling multi-turn conversations where the agent can recall previous predictions, clarifications, or risk discussions.
