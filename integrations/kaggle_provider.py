"""Kaggle Datasets Provider."""
import os
import logging
from typing import List, Optional, Tuple
from schemas.dataset_schema import DatasetMetadata

logger = logging.getLogger(__name__)

class KaggleProvider:
    """Discovers datasets from Kaggle via the Kaggle API if credentials exist."""

    def __init__(self, username: Optional[str] = None, key: Optional[str] = None):
        self.username = username or os.getenv("KAGGLE_USERNAME")
        self.key = key or os.getenv("KAGGLE_KEY")

    def check_credentials(self) -> Tuple[bool, str]:
        """Verify whether Kaggle API credentials are provided."""
        kaggle_json_path = os.path.expanduser("~/.kaggle/kaggle.json")
        if self.username and self.key:
            return True, "Kaggle credentials configured via environment variables."
        elif os.path.exists(kaggle_json_path):
            return True, f"Kaggle credentials found at {kaggle_json_path}."
        else:
            return False, "Kaggle credentials not configured. Set KAGGLE_USERNAME and KAGGLE_KEY or place kaggle.json in ~/.kaggle/."

    def search(self, query: str, limit: int = 10) -> List[DatasetMetadata]:
        has_creds, reason = self.check_credentials()
        if not has_creds:
            logger.info(f"Skipping Kaggle search: {reason}")
            return []

        results: List[DatasetMetadata] = []
        try:
            # Try importing kaggle python package if installed
            import kaggle
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            datasets = api.dataset_list(search=query, sort_by="hottest")
            for ds in datasets[:limit]:
                results.append(
                    DatasetMetadata(
                        dataset_id=f"kaggle_{ds.ref.replace('/', '_')}",
                        name=f"Kaggle: {ds.title}",
                        description=f"Kaggle dataset by {ds.creatorName}. Size: {getattr(ds, 'size', 'Unknown')}.",
                        source_provider="kaggle",
                        url=f"https://www.kaggle.com/datasets/{ds.ref}",
                        data_format="csv/zip",
                        is_verified=True,
                        tags=["kaggle", "community"]
                    )
                )
        except Exception as e:
            logger.warning(f"Error executing Kaggle search: {e}")

        return results
