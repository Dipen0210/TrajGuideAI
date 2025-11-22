# LLM Tooling

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
- The same client is used by the standalone tools for explanation, risk analysis, and retrieval.

## Note on the Chat Agent
- The interactive chat agent and its `/agent/query` endpoint have been removed from the product surface.
- Tools remain available for direct use by backend endpoints and scripts without conversational orchestration.
