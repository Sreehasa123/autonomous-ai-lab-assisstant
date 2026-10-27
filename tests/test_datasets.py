"""Dataset Discovery and Inspection Tests."""
import pytest
import pandas as pd
from services.dataset_service import DatasetService
from integrations.uci_provider import UCIRepositoryProvider
from integrations.openml_provider import OpenMLProvider
from integrations.huggingface_provider import HuggingFaceProvider
from integrations.kaggle_provider import KaggleProvider

def test_uci_provider_search():
    provider = UCIRepositoryProvider()
    results = provider.search("iris", limit=3)
    assert len(results) > 0
    iris_ds = results[0]
    assert "iris" in iris_ds.name.lower() or "iris" in iris_ds.description.lower()
    assert iris_ds.source_provider == "uci"
    assert iris_ds.num_rows is not None

def test_dataset_service_search_all():
    service = DatasetService()
    search_res = service.search_all_providers("wine", limit_per_provider=2)
    assert search_res["total_found"] > 0
    assert "uci" in search_res["providers"]
    assert "openml" in search_res["providers"]
    assert "kaggle" in search_res["providers"]

def test_dataset_analysis_and_quality():
    service = DatasetService()
    # Create test dataframe with missing values, duplicate, and identifier column
    df = pd.DataFrame({
        "id": range(50),
        "feature_1": [1.0, 2.0, None, 4.0, 5.0] * 10,
        "feature_2": [10.0, 20.0, 30.0, 40.0, 50.0] * 10,
        "constant_col": [99] * 50,
        "target": [0, 1] * 25
    })
    # Add an explicit duplicate row
    df.loc[49] = df.loc[48]

    analysis = service.inspect_and_analyze(df, target_column="target")
    assert analysis.shape == [50, 5]
    assert "id" in analysis.identifier_columns or "id" in analysis.potential_leakage_columns
    assert "constant_col" in analysis.constant_columns
    assert analysis.duplicate_rows >= 1
    assert any(f.category == "missing_values" for f in analysis.findings)
