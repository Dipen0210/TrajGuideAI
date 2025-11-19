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
from langchain.llms.base import LLM


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "llama3_config.yaml"
HF_TOKEN_ENV = "HF_LLAMA33_TOKEN"
HF_MODEL_ENV = "HF_LLAMA33_MODEL"
LLAMA3_BASE_ENV = "LLAMA3_API_BASE"

load_dotenv()


@dataclass
class Llama3Config:
    api_base: str
    api_key: str
    model: str = "llama3"
    temperature: float = 0.2
    max_tokens: int = 512
    timeout: int = 60


class Llama3LLM(LLM):
    """
    LangChain-compatible wrapper around a hosted Llama 3 endpoint.
    """

    def __init__(self, config: Llama3Config) -> None:
        super().__init__()
        # BaseModel forbids normal __setattr__; bypass for simple storage.
        object.__setattr__(self, "config", config)

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
        }

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """
        Issue a chat-completion style request to the configured endpoint.
        """
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "messages": [
                {
                    "role": "system",
                    "content": "You are Llama 3 assisting with vehicle trajectory reasoning.",
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
    token_override = os.getenv(HF_TOKEN_ENV)
    model_override = os.getenv(HF_MODEL_ENV)
    base_override = os.getenv(LLAMA3_BASE_ENV)
    if token_override:
        merged_config["api_key"] = token_override
    if model_override:
        merged_config["model"] = model_override
    if base_override:
        merged_config["api_base"] = base_override

    required_fields = {"api_base", "api_key"}
    missing = [field for field in required_fields if not merged_config.get(field)]
    if missing:
        raise ValueError(
            f"Llama 3 config missing fields: {missing}. Provide them in the YAML file or .env."
        )

    config = Llama3Config(**merged_config)
    return Llama3LLM(config)
