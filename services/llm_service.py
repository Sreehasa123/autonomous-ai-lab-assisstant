"""LLM Service Abstraction with graceful rule-based fallbacks."""
import os
import json
import logging
from typing import Dict, Any, Optional, Tuple
from config.settings import settings

logger = logging.getLogger(__name__)

class LLMService:
    """Manages optional LLM interactions with transparent rule-based fallbacks."""

    def __init__(self, provider: Optional[str] = None, api_key: Optional[str] = None, model: Optional[str] = None):
        self.provider = provider or settings.llm_provider
        self.api_key = api_key or settings.llm_api_key
        self.model = model or settings.llm_model

    def is_configured(self) -> bool:
        """Check if an external LLM provider with API key is available."""
        if not self.api_key or self.provider.lower() in ("none", "", "disabled"):
            return False
        return True

    def generate_completion(self, system_prompt: str, user_prompt: str) -> Tuple[bool, str, str]:
        """
        Attempts LLM completion if configured; otherwise returns failure reason.
        Returns: (success: bool, content: str, source: 'llm' | 'rule_based')
        """
        if not self.is_configured():
            return False, "LLM not configured. Falling back to rule-based assistant.", "rule_based"

        provider = self.provider.lower()
        try:
            if "gemini" in provider:
                # Optional google-genai or requests to Gemini endpoint
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=self.api_key)
                    g_model = genai.GenerativeModel(self.model or "gemini-1.5-flash")
                    response = g_model.generate_content(f"{system_prompt}\n\nUser Request: {user_prompt}")
                    return True, response.text, "llm"
                except Exception as ex:
                    logger.warning(f"Gemini API invocation failed: {ex}")
                    return False, str(ex), "rule_based"

            elif "openai" in provider:
                try:
                    import openai
                    client = openai.OpenAI(api_key=self.api_key)
                    completion = client.chat.completions.create(
                        model=self.model or "gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ]
                    )
                    return True, completion.choices[0].message.content, "llm"
                except Exception as ex:
                    logger.warning(f"OpenAI API invocation failed: {ex}")
                    return False, str(ex), "rule_based"

            elif "anthropic" in provider:
                try:
                    import anthropic
                    client = anthropic.Anthropic(api_key=self.api_key)
                    message = client.messages.create(
                        model=self.model or "claude-3-haiku-20240307",
                        max_tokens=1000,
                        system=system_prompt,
                        messages=[{"role": "user", "content": user_prompt}]
                    )
                    return True, message.content[0].text, "llm"
                except Exception as ex:
                    logger.warning(f"Anthropic API invocation failed: {ex}")
                    return False, str(ex), "rule_based"

            return False, f"Unsupported LLM provider: {self.provider}", "rule_based"
        except Exception as e:
            logger.error(f"Error in LLM service: {e}")
            return False, str(e), "rule_based"
