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


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "llama3_config.yaml"
HF_TOKEN_ENV = "HF_LLAMA33_TOKEN"
HF_MODEL_ENV = "HF_LLAMA33_MODEL"
LLAMA3_BASE_ENV = "LLAMA3_API_BASE"
LLAMA3_PROVIDER_ENV = "LLAMA3_PROVIDER"
HF_COMPAT_TOKEN_ENV = "HF_API_KEY"
HF_COMPAT_MODEL_ENV = "HF_MODEL_ID"
HF_COMPAT_BASE_ENV = "HF_API_BASE"

SYSTEM_PROMPT = "You are Llama 3 assisting with vehicle trajectory reasoning."

load_dotenv()


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
        }

    def _build_hf_client(self) -> Optional[InferenceClient]:
        """
        Initialize a Hugging Face inference client when requested.
        """
        provider = (self.config.provider or "").lower()
        base = (self.config.api_base or "").lower()
        if provider == "huggingface" or "huggingface" in base:
            return InferenceClient(
                model=self.config.model,
                token=self.config.api_key,
                timeout=self.config.timeout,
            )
        return None

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """
        Issue a chat-completion style request to the configured endpoint.
        """
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
        Attempts chat-completions first (needed for Groq providers), then falls back to text-generation.
        """
        client = getattr(self, "_hf_client", None)
        if client is None:
            raise RuntimeError("Hugging Face client not initialized.")

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            },
        ]

        # Prefer chat completion for providers that only expose conversational APIs.
        if hasattr(client, "chat_completion"):
            response = client.chat_completion(
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                stop=stop,
            )
            return self._parse_hf_chat_response(response)

        # Fallback to text generation for providers that support it.
        chat_prompt = self._format_chat_prompt(prompt)
        return client.text_generation(
            chat_prompt,
            temperature=self.config.temperature,
            max_new_tokens=self.config.max_tokens,
            stop_sequences=stop,
            return_full_text=False,
        )

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


def load_llama3(config_path: Optional[Path] = None) -> Llama3LLM:
    """
    Load configuration from YAML and return a LangChain-compatible client.
    """
    path = config_path or DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Llama 3 config missing at {path}. Provide api_base and api_key entries."
        )
    with open(path, "r", encoding="utf-8") as config_file:
        config_dict = yaml.safe_load(config_file) or {}

    merged_config: Dict[str, Any] = dict(config_dict)
    token_override = os.getenv(HF_TOKEN_ENV) or os.getenv(HF_COMPAT_TOKEN_ENV)
    model_override = os.getenv(HF_MODEL_ENV) or os.getenv(HF_COMPAT_MODEL_ENV)
    base_override = os.getenv(LLAMA3_BASE_ENV) or os.getenv(HF_COMPAT_BASE_ENV)
    provider_override = os.getenv(LLAMA3_PROVIDER_ENV)
    if token_override:
        merged_config["api_key"] = token_override
    if model_override:
        merged_config["model"] = model_override
    if base_override:
        merged_config["api_base"] = base_override
    if provider_override:
        merged_config["provider"] = provider_override

    required_fields = {"api_base", "api_key"}
    missing = [field for field in required_fields if not merged_config.get(field)]
    if missing:
        raise ValueError(
            f"Llama 3 config missing fields: {missing}. Provide them in the YAML file or .env."
        )

    config = Llama3Config(**merged_config)
    return Llama3LLM(config)
