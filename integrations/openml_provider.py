"""OpenML Datasets Provider."""
import logging
import requests
from typing import List, Optional
from schemas.dataset_schema import DatasetMetadata

logger = logging.getLogger(__name__)

class OpenMLProvider:
    """Discovers machine learning datasets from OpenML."""

    BASE_URL = "https://www.openml.org/api/v1/json/data/list"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    def search(self, query: str, limit: int = 10) -> List[DatasetMetadata]:
        results: List[DatasetMetadata] = []
        try:
            # Query openml data list with data_name filter
            clean_query = query.strip().split()[0] if query.strip() else "wine"
            url = f"{self.BASE_URL}/data_name/{clean_query}/limit/{limit}"
            resp = requests.get(url, timeout=8)
            if resp.status_code == 200:
                data = resp.json()
                data_list = (data.get("data", {}) or {}).get("dataset", [])
                for item in data_list:
                    did = str(item.get("did", ""))
                    name = item.get("name", f"OpenML Dataset {did}")
                    format_type = item.get("format", "ARFF/CSV")
                    
                    # OpenML task type
                    qualities = item.get("quality", [])
                    num_features = None
                    num_instances = None
                    for q in qualities:
                        if q.get("name") == "NumberOfFeatures":
                            num_features = int(float(q.get("value", 0)))
                        elif q.get("name") == "NumberOfInstances":
                            num_instances = int(float(q.get("value", 0)))

                    results.append(
                        DatasetMetadata(
                            dataset_id=f"openml_{did}",
                            name=f"OpenML: {name}",
                            description=f"Standard benchmark dataset from OpenML (DID: {did}). Format: {format_type}.",
                            source_provider="openml",
                            url=f"https://www.openml.org/search?type=data&id={did}",
                            download_url=f"https://www.openml.org/data/download/{did}",
                            data_format="csv",
                            num_rows=num_instances,
                            num_features=num_features,
                            task_type="classification",
                            license="Public Domain / CC-BY",
                            is_verified=True,
                            tags=["openml", "tabular"]
                        )
                    )
            else:
                logger.debug(f"OpenML search returned {resp.status_code} for query {query}")
        except Exception as e:
            logger.warning(f"Error querying OpenML: {e}")

        return results
