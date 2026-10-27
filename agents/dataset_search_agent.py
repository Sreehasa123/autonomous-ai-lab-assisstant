"""Dataset Search Agent."""
from typing import Dict, Any, List
from agents.base_agent import BaseAgent
from services.dataset_service import DatasetService

class DatasetSearchAgent(BaseAgent):
    """Executes multi-provider dataset discovery across HuggingFace, OpenML, UCI, and Kaggle."""

    def __init__(self):
        super().__init__(
            name="Dataset Search Agent",
            description="Searches active dataset repositories to find candidate datasets matching experimental requirements."
        )
        self.dataset_service = DatasetService()

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        query = context.get("search_query") or context.get("problem_statement", "")
        # Clean query for effective keyword search
        terms = [w for w in query.replace(",", " ").replace(".", " ").split() if len(w) > 2]
        clean_query = " ".join(terms[:4]) if terms else "tabular"

        search_output = self.dataset_service.search_all_providers(clean_query, limit_per_provider=5)
        return {
            "search_query": clean_query,
            "datasets": [d.dict() for d in search_output["datasets"]],
            "providers": search_output["providers"],
            "total_found": search_output["total_found"]
        }
