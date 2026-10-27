"""Base Agent Class."""
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from services.llm_service import LLMService

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """Abstract base class for all AI Lab Assistant agents."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.llm_service = LLMService()

    @abstractmethod
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Executes the agent's specific responsibility."""
        pass

    def log(self, message: str) -> None:
        """Helper for standard logging."""
        logger.info(f"[{self.name}] {message}")
