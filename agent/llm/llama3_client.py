"""
Utility for loading a configurable Llama 3 HTTP client.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import yaml
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain.llms.base import LLM

# Gemini (Google AI) support
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "llama3_config.yaml"
# ... (rest of constants)
HF_TOKEN_ENV = "HF_LLAMA33_TOKEN"
HF_MODEL_ENV = "HF_LLAMA33_MODEL"
LLAMA3_BASE_ENV = "LLAMA3_API_BASE"
LLAMA3_PROVIDER_ENV = "LLAMA3_PROVIDER"
HF_COMPAT_TOKEN_ENV = "HF_API_KEY"
HF_COMPAT_MODEL_ENV = "HF_MODEL_ID"
HF_COMPAT_BASE_ENV = "HF_API_BASE"

SYSTEM_PROMPT = "You are Llama 3 assisting with vehicle trajectory reasoning."

load_dotenv()


MOCK_MODE = False  # Set to True to bypass API and use simulated responses

@dataclass
class Llama3Config:
    api_base: str
    api_key: str
    model: str = "llama3"
    temperature: float = 0.2
    max_tokens: int = 512
    timeout: int = 60
    provider: str = "custom"


class Llama3LLM(LLM):
    """
    LangChain-compatible wrapper around a hosted Llama 3 endpoint.
    Supports MOCK_MODE for testing without API credits.
    """

    def __init__(self, config: Llama3Config) -> None:
        super().__init__()
        # BaseModel forbids normal __setattr__; bypass for simple storage.
        object.__setattr__(self, "config", config)
        object.__setattr__(self, "_hf_client", self._build_hf_client())

    @property
    def _llm_type(self) -> str:
        return "llama3_custom"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "api_base": self.config.api_base,
            "provider": self.config.provider,
            "mock_mode": MOCK_MODE,
        }

    def _build_hf_client(self) -> Optional[InferenceClient]:
        """
        Initialize a Hugging Face inference client.
        Uses HuggingFace Inference Providers (supports Cerebras, SambaNova, etc.).
        """
        if MOCK_MODE:
            return None
        
        # Use provider parameter for Inference Providers (Cerebras, SambaNova, etc.)
        provider = self.config.provider if self.config.provider != "custom" else None
        
        return InferenceClient(
            api_key=self.config.api_key,
            timeout=self.config.timeout,
            provider=provider,
        )

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """
        Issue a chat-completion style request to the configured endpoint.
        If MOCK_MODE is True, returns simulated responses based on prompt keywords.
        """
        if MOCK_MODE:
            # --- Mock Logic for Safety Auditor ---
            if "SAFETY AUDIT" in prompt or "safety rules" in prompt:
                return (
                    "Thought: The vehicle is moving at 25m/s. I need to check the safety rules.\n"
                    "Action: consult_safety_rules\n"
                    "Action Input: What is the safe following distance at 25m/s?\n"
                    "Observation: The rule states a 3-second gap is needed (approx 75m).\n"
                    "Final Answer: ⚠️ CRITICAL VIOLATION: The driver is following too closely. "
                    "At 25m/s, a safe gap is 75m, but the current gap is much smaller. "
                    "Risk of rear-end collision is HIGH."
                )

            # --- Mock Logic for Driver Profiler ---
            if "driver profile" in prompt or "profile" in prompt.lower():
                return (
                    "Thought: I need to analyze the driving behavior.\n"
                    "Final Answer: "
                    "{\n"
                    '  "style_classification": "Aggressive",\n'
                    '  "metrics": {\n'
                    '    "mean_velocity": 22.5,\n'
                    '    "std_acceleration": 2.87,\n'
                    '    "num_lane_changes": 4\n'
                    "  },\n"
                    '  "summary": "Driver exhibits aggressive tendencies with high acceleration variance and frequent lane changes."\n'
                    "}"
                )
            
            # --- Mock Logic for Risk Assessment ---
            if "risk" in prompt.lower():
                 return (
                    "{\n"
                    '  "risk_score": 0.85,\n'
                    '  "risk_factors": "High speed (25m/s), Low headway (<15m)",\n'
                    '  "recommendation": "Increase following distance immediately."\n'
                    "}"
                 )

            # --- Default Mock Response ---
            return "Mock Llama 3 Response: I have analyzed the trajectory and everything appears nominal."

        if getattr(self, "_hf_client", None) is not None:
            try:
                return self._call_hf_client(prompt, stop)
            except Exception as exc:  # pylint: disable=broad-except
                raise RuntimeError(f"Hugging Face inference failed: {exc}") from exc

        payload: Dict[str, Any] = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {"role": "user", "content": prompt},
            ],
        }
        if stop:
            payload["stop"] = stop

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(
            self.config.api_base,
            json=payload,
            headers=headers,
            timeout=self.config.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return self._extract_content(data)

    def generate_text(self, prompt: str) -> str:
        """
        Convenience helper for non-agent components.
        """
        return self._call(prompt)

    @staticmethod
    def _format_chat_prompt(user_prompt: str) -> str:
        """
        Format OpenAI-style chat messages into a single prompt for HF endpoints.
        """
        return (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            f"{SYSTEM_PROMPT}\n"
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            f"{user_prompt}\n"
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        )

    def _call_hf_client(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """
        Handle providers available through huggingface_hub.InferenceClient.
        Uses chat.completions.create for HuggingFace Inference Providers (Cerebras, SambaNova, etc.).
        """
        client = getattr(self, "_hf_client", None)
        if client is None:
            raise RuntimeError("Hugging Face client not initialized.")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        # Use chat.completions.create for Inference Providers (Cerebras, etc.)
        # The model format is: meta-llama/Llama-3.3-70B-Instruct:cerebras
        try:
            completion = client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            # Extract the response content
            return completion.choices[0].message.content
        except Exception as e:
            # Log the actual error for debugging
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Chat completion failed: {e}") from e

    @staticmethod
    def _parse_hf_chat_response(response: Dict[str, Any]) -> str:
        """
        Parse chat-completion style responses returned by Hugging Face providers.
        """
        choices = response.get("choices") if isinstance(response, dict) else None
        if not choices:
            raise ValueError("No choices in chat completion response.")
        message = choices[0].get("message") or {}
        # Groq/GPT-style responses may store content as list[dict] or string.
        content = message.get("content")
        if isinstance(content, list):
            text_parts = []
            for chunk in content:
                if isinstance(chunk, dict):
                    text_parts.append(chunk.get("text", ""))
                else:
                    text_parts.append(str(chunk))
            return " ".join(part for part in text_parts if part)
        if isinstance(content, str):
            return content
        # Some providers put text under 'text' key at top-level.
        if "text" in message:
            return str(message["text"])
        raise ValueError("Unable to parse chat completion content.")

    @staticmethod
    def _extract_content(response_data: Dict[str, Any]) -> str:
        choices = response_data.get("choices", [])
        if not choices:
            raise ValueError("No choices returned from Llama 3 response.")
        message = choices[0].get("message", {})
        content = message.get("content")
        if isinstance(content, list):
            return " ".join(chunk.get("text", "") if isinstance(chunk, dict) else str(chunk) for chunk in content)
        if isinstance(content, str):
            return content
        raise ValueError("Unexpected response format from Llama 3.")


def load_llama3(config_path: Optional[Path] = None) -> Any:
    """
    Load configuration from YAML and return a LangChain-compatible client.
    Supports HuggingFace Inference Providers (Cerebras, SambaNova, etc.).
    """
    path = config_path or DEFAULT_CONFIG_PATH
    config_dict = {}
    
    if path.exists():
        with open(path, "r", encoding="utf-8") as config_file:
            config_dict = yaml.safe_load(config_file) or {}
    else:
        # If config file is missing, we rely entirely on environment variables
        pass

    merged_config: Dict[str, Any] = dict(config_dict)
    
    # Check for various environment variable patterns
    token_override = (
        os.getenv(HF_TOKEN_ENV) 
        or os.getenv(HF_COMPAT_TOKEN_ENV)
        or os.getenv("HF_TOKEN")
    )
    model_override = (
        os.getenv(HF_MODEL_ENV) 
        or os.getenv(HF_COMPAT_MODEL_ENV)
    )
    base_override = (
        os.getenv(LLAMA3_BASE_ENV) 
        or os.getenv(HF_COMPAT_BASE_ENV)
    )
    provider_override = os.getenv(LLAMA3_PROVIDER_ENV)

    if token_override:
        merged_config["api_key"] = token_override
    if model_override:
        merged_config["model"] = model_override
    if base_override:
        merged_config["api_base"] = base_override
    if provider_override:
        merged_config["provider"] = provider_override
    
    # Check for Gemini first
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if gemini_key and HAS_GEMINI:
        model_id = (
            os.getenv("GEMINI_MODEL_ID")
            or os.getenv("GEMINI_MODEL")
        )
        if not model_id:
            raise ValueError("GEMINI_MODEL_ID (or GEMINI_MODEL) must be set in the environment for Gemini usage.")
        return ChatGoogleGenerativeAI(
            model=model_id,
            google_api_key=gemini_key,
            temperature=merged_config.get("temperature", 0.2),
            convert_system_message_to_human=False,
            timeout=None,
        )
    
    # For HuggingFace Inference Providers, we only need api_key and model
    if not merged_config.get("api_key"):
        raise ValueError(
            "HF_TOKEN must be set in the environment for HuggingFace Inference Providers."
        )
    
    # Set a default api_base if not provided (not used for HF Inference Providers)
    if not merged_config.get("api_base"):
        merged_config["api_base"] = "https://api-inference.huggingface.co"
    
    config = Llama3Config(**merged_config)
    return Llama3LLM(config)
