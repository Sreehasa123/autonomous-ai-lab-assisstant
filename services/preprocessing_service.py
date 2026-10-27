"""Data Preprocessing Service."""
import logging
from typing import Dict, Any, Tuple, List, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from schemas.experiment_schema import PreprocessingConfig

logger = logging.getLogger(__name__)

class PreprocessingService:
    """Prepares and transforms datasets cleanly without data leakage."""

    def preprocess_and_split(
        self,
        df: pd.DataFrame,
        target_column: str,
        config: PreprocessingConfig,
        task_type: str = "classification"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], Optional[LabelEncoder]]:
        """
        Executes complete preprocessing pipeline:
        Returns: (X_train, X_test, y_train, y_test, feature_names, target_encoder)
        """
        data = df.copy()

        # 1. Deduplicate if enabled
        if config.remove_duplicates:
            data = data.drop_duplicates()

        # 2. Separate Target and Features
        if target_column not in data.columns:
            target_column = data.columns[-1]

        y_raw = data[target_column]
        X_raw = data.drop(columns=[target_column])

        # 3. Drop constant columns
        constant_cols = [c for c in X_raw.columns if X_raw[c].nunique() <= 1]
        if constant_cols:
            X_raw = X_raw.drop(columns=constant_cols)

        # 4. Handle Target Encoding for classification
        target_encoder = None
        if "classification" in task_type.lower():
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(y_raw.astype(str))
        else:
            y = pd.to_numeric(y_raw, errors="coerce").fillna(y_raw.median() if hasattr(y_raw, 'median') else 0).values

        # 5. Train / Test Split before fitting any transformers to prevent leakage
        stratify = y if ("classification" in task_type.lower() and len(np.unique(y)) > 1 and np.min(np.bincount(y)) >= 2) else None
        
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_raw,
            y,
            test_size=config.test_size,
            random_state=config.random_state,
            stratify=stratify
        )

        # 6. Numeric & Categorical handling
        numeric_cols = list(X_raw.select_dtypes(include=[np.number]).columns)
        cat_cols = list(X_raw.select_dtypes(exclude=[np.number]).columns)

        X_train_processed_parts = []
        X_test_processed_parts = []
        final_feature_names = []

        # Handle Numeric
        if numeric_cols:
            # Impute
            strategy = "median" if config.handle_missing == "median_mode" else ("mean" if config.handle_missing == "mean" else "median")
            imputer = SimpleImputer(strategy=strategy)
            num_train_imp = imputer.fit_transform(X_train_raw[numeric_cols])
            num_test_imp = imputer.transform(X_test_raw[numeric_cols])

            # Scale
            if config.scale_numeric == "standard":
                scaler = StandardScaler()
                num_train_scaled = scaler.fit_transform(num_train_imp)
                num_test_scaled = scaler.transform(num_test_imp)
            elif config.scale_numeric == "minmax":
                scaler = MinMaxScaler()
                num_train_scaled = scaler.fit_transform(num_train_imp)
                num_test_scaled = scaler.transform(num_test_imp)
            elif config.scale_numeric == "robust":
                scaler = RobustScaler()
                num_train_scaled = scaler.fit_transform(num_train_imp)
                num_test_scaled = scaler.transform(num_test_imp)
            else:
                num_train_scaled = num_train_imp
                num_test_scaled = num_test_imp

            X_train_processed_parts.append(num_train_scaled)
            X_test_processed_parts.append(num_test_scaled)
            final_feature_names.extend(numeric_cols)

        # Handle Categorical
        if cat_cols:
            cat_imputer = SimpleImputer(strategy="most_frequent")
            cat_train_imp = pd.DataFrame(cat_imputer.fit_transform(X_train_raw[cat_cols]), columns=cat_cols)
            cat_test_imp = pd.DataFrame(cat_imputer.transform(X_test_raw[cat_cols]), columns=cat_cols)

            # One-hot encode using pandas get_dummies aligned
            cat_train_encoded = pd.get_dummies(cat_train_imp, drop_first=True)
            cat_test_encoded = pd.get_dummies(cat_test_imp, drop_first=True)
            cat_train_encoded, cat_test_encoded = cat_train_encoded.align(cat_test_encoded, join="left", axis=1, fill_value=0)

            X_train_processed_parts.append(cat_train_encoded.values)
            X_test_processed_parts.append(cat_test_encoded.values)
            final_feature_names.extend(list(cat_train_encoded.columns))

        # Combine
        if X_train_processed_parts:
            X_train = np.hstack(X_train_processed_parts)
            X_test = np.hstack(X_test_processed_parts)
        else:
            # Fallback if no valid features
            X_train = np.zeros((len(X_train_raw), 1))
            X_test = np.zeros((len(X_test_raw), 1))
            final_feature_names = ["dummy_feature"]

        return X_train, X_test, y_train, y_test, final_feature_names, target_encoder
