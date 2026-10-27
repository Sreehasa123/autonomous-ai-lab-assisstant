"""UCI Machine Learning Repository Provider."""
import logging
import requests
from typing import List, Optional
from schemas.dataset_schema import DatasetMetadata

logger = logging.getLogger(__name__)

# Standard curated benchmark datasets from UCI / Scikit-Learn standard collection
BENCHMARK_UCI_DATASETS = [
    {
        "id": "uci_iris",
        "name": "Iris Flower Dataset",
        "description": "Famous dataset containing 3 classes of 50 instances each, where each class refers to a type of iris plant (Setosa, Versicolour, Virginica).",
        "task_type": "classification",
        "num_rows": 150,
        "num_features": 4,
        "target_column": "target",
        "license": "CC-BY 4.0",
        "url": "https://archive.ics.uci.edu/dataset/53/iris",
        "tags": ["iris", "flowers", "botany", "classification", "tabular"]
    },
    {
        "id": "uci_wine",
        "name": "Wine Recognition Dataset",
        "description": "Chemical analysis of wines grown in the same region in Italy derived from three different cultivars.",
        "task_type": "classification",
        "num_rows": 178,
        "num_features": 13,
        "target_column": "target",
        "license": "CC-BY 4.0",
        "url": "https://archive.ics.uci.edu/dataset/109/wine",
        "tags": ["wine", "chemistry", "agriculture", "classification", "tabular"]
    },
    {
        "id": "uci_breast_cancer",
        "name": "Breast Cancer Wisconsin (Diagnostic)",
        "description": "Features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. 30 real-valued features describing cell nuclei.",
        "task_type": "classification",
        "num_rows": 569,
        "num_features": 30,
        "target_column": "target",
        "license": "CC-BY 4.0",
        "url": "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
        "tags": ["medical", "cancer", "healthcare", "classification", "tabular"]
    },
    {
        "id": "uci_diabetes",
        "name": "Diabetes Diagnostic Dataset",
        "description": "Ten baseline variables, age, sex, BMI, average blood pressure, and six blood serum measurements obtained for each of n=442 diabetes patients.",
        "task_type": "regression",
        "num_rows": 442,
        "num_features": 10,
        "target_column": "target",
        "license": "CC-BY 4.0",
        "url": "https://archive.ics.uci.edu/dataset/34/diabetes",
        "tags": ["medical", "diabetes", "health", "regression", "tabular"]
    },
    {
        "id": "uci_california_housing",
        "name": "California Housing Prices",
        "description": "Median house values for California districts derived from the 1990 census. Standard benchmark for spatial regression.",
        "task_type": "regression",
        "num_rows": 20640,
        "num_features": 8,
        "target_column": "MedHouseVal",
        "license": "Public Domain",
        "url": "https://archive.ics.uci.edu/dataset/477/real+estate+valuation+data+set",
        "tags": ["housing", "real estate", "prices", "regression", "tabular"]
    },
    {
        "id": "uci_heart_disease",
        "name": "Heart Disease Cleveland Dataset",
        "description": "Clinical database containing 76 attributes, with experiments using a subset of 14 features to predict presence of heart disease.",
        "task_type": "classification",
        "num_rows": 303,
        "num_features": 13,
        "target_column": "target",
        "license": "CC-BY 4.0",
        "url": "https://archive.ics.uci.edu/dataset/45/heart+disease",
        "tags": ["medical", "cardiology", "health", "classification", "tabular"]
    }
]

class UCIRepositoryProvider:
    """Discovers datasets from the UCI Machine Learning Repository."""

    API_URL = "https://archive.ics.uci.edu/api/datasets"

    def search(self, query: str, limit: int = 10) -> List[DatasetMetadata]:
        results: List[DatasetMetadata] = []
        q_lower = query.lower().strip()
        tokens = q_lower.split()

        # 1. First, search verified curated collection
        for ds in BENCHMARK_UCI_DATASETS:
            match = False
            if not q_lower:
                match = True
            else:
                for token in tokens:
                    if token in ds["name"].lower() or token in ds["description"].lower() or any(token in t for t in ds["tags"]):
                        match = True
                        break
            if match:
                results.append(
                    DatasetMetadata(
                        dataset_id=ds["id"],
                        name=f"UCI: {ds['name']}",
                        description=ds["description"],
                        source_provider="uci",
                        url=ds["url"],
                        data_format="csv",
                        num_rows=ds["num_rows"],
                        num_features=ds["num_features"],
                        target_column=ds["target_column"],
                        task_type=ds["task_type"],
                        license=ds["license"],
                        is_verified=True,
                        tags=ds["tags"]
                    )
                )

        # 2. Query UCI REST API if possible
        try:
            params = {"search": query, "take": limit}
            resp = requests.get(self.API_URL, params=params, timeout=6)
            if resp.status_code == 200:
                data = resp.json()
                items = data.get("data", [])
                for item in items:
                    uid = str(item.get("ID", item.get("id", "")))
                    name = item.get("Name", item.get("name", f"UCI Dataset {uid}"))
                    desc = item.get("Abstract", item.get("description", "")) or f"UCI Machine Learning Repository dataset {name}"
                    ds_meta = DatasetMetadata(
                        dataset_id=f"uci_{uid}",
                        name=f"UCI: {name}",
                        description=desc[:280],
                        source_provider="uci",
                        url=f"https://archive.ics.uci.edu/dataset/{uid}",
                        data_format="csv",
                        num_rows=item.get("NumInstances"),
                        num_features=item.get("NumAttributes"),
                        task_type=item.get("TaskType", "tabular"),
                        license="CC-BY / Academic",
                        is_verified=True,
                        tags=["uci", "benchmark"]
                    )
                    # Deduplicate if already added
                    if not any(r.dataset_id == ds_meta.dataset_id or r.name.lower() == ds_meta.name.lower() for r in results):
                        results.append(ds_meta)
        except Exception as e:
            logger.debug(f"UCI live API query returned: {e}")

        return results[:limit]
