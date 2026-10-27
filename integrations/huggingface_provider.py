"""Hugging Face Datasets Provider."""
import logging
import requests
from typing import List, Optional
from schemas.dataset_schema import DatasetMetadata

logger = logging.getLogger(__name__)

class HuggingFaceProvider:
    """Discovers datasets from Hugging Face Hub API."""

    BASE_URL = "https://huggingface.co/api/datasets"

    def __init__(self, token: Optional[str] = None):
        self.token = token

    def search(self, query: str, limit: int = 10) -> List[DatasetMetadata]:
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        results: List[DatasetMetadata] = []
        try:
            params = {
                "search": query,
                "limit": limit,
                "full": "true"
            }
            resp = requests.get(self.BASE_URL, headers=headers, params=params, timeout=8)
            if resp.status_code == 200:
                data = resp.json()
                for item in data:
                    ds_id = item.get("id", "")
                    description = item.get("description", "") or (item.get("cardData", {}) or {}).get("description", "")
                    if not description:
                        description = f"Hugging Face dataset: {ds_id}"
                    
                    # Truncate long descriptions
                    if len(description) > 300:
                        description = description[:297] + "..."

                    tags = item.get("tags", [])
                    task_type = "classification"
                    for t in tags:
                        if "classification" in t:
                            task_type = "classification"
                            break
                        elif "regression" in t:
                            task_type = "regression"
                            break
                        elif "nlp" in t or "translation" in t:
                            task_type = "nlp"
                            break
                        elif "computer-vision" in t or "image" in t:
                            task_type = "cv"
                            break

                    dataset_meta = DatasetMetadata(
                        dataset_id=f"hf_{ds_id.replace('/', '_')}",
                        name=ds_id,
                        description=description,
                        source_provider="huggingface",
                        url=f"https://huggingface.co/datasets/{ds_id}",
                        data_format="parquet/arrow",
                        task_type=task_type,
                        license=item.get("license", "Unknown"),
                        is_verified=True,
                        tags=tags[:5]
                    )
                    results.append(dataset_meta)
            else:
                logger.warning(f"Hugging Face API returned status {resp.status_code}")
        except Exception as e:
            logger.warning(f"Error querying Hugging Face API: {e}")

        return results
