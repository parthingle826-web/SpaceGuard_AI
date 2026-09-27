"""
SpaceGuard AI - ML Preprocessing Module
Handles data ingestion, validation, feature engineering, scaling, and train-test splitting.
NOTE: Academic simulation only - Not real NASA/spacecraft mission data.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from backend.config import (
    RAW_DATASET_PATH,
    TRAIN_DATASET_PATH,
    TEST_DATASET_PATH,
    MODELS_DIR,
    TELEMETRY_FEATURES,
    NOMINAL_RANGES,
    ANOMALY_CLASSES
)

# Base telemetry input features expected from sensors
BASE_FEATURES = TELEMETRY_FEATURES.copy()

# Engineered features derived deterministically from single records
ENGINEERED_FEATURES = [
    "power_estimated_eps",  # battery_voltage * battery_current
    "solar_power_gen",      # solar_panel_voltage * solar_panel_current
    "temp_delta_cpu",       # cpu_subsystem_temp - temperature
    "v_bat_dev",            # abs(battery_voltage - nominal 28.0)
    "temp_dev",             # abs(temperature - nominal 22.0)
    "pressure_dev",         # abs(pressure - nominal 101.3)
    "pwr_dev"               # abs(power_consumption - nominal 200.0)
]

ALL_MODEL_FEATURES = BASE_FEATURES + ENGINEERED_FEATURES

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes deterministic domain-specific satellite engineering features.
    Works seamlessly on both multi-row DataFrames and single-row inference payloads.
    """
    df_out = df.copy()
    
    # 1. Electrical Power Subsystem (EPS) internal power product
    df_out["power_estimated_eps"] = df_out["battery_voltage"] * df_out["battery_current"]
    
    # 2. Solar array generation power product
    df_out["solar_power_gen"] = df_out["solar_panel_voltage"] * df_out["solar_panel_current"]
    
    # 3. CPU vs. Chassis thermal gradient
    df_out["temp_delta_cpu"] = df_out["cpu_subsystem_temp"] - df_out["temperature"]
    
    # 4. Deviations from nominal baselines
    df_out["v_bat_dev"] = (df_out["battery_voltage"] - 28.0).abs()
    df_out["temp_dev"] = (df_out["temperature"] - 22.0).abs()
    df_out["pressure_dev"] = (df_out["pressure"] - 101.3).abs()
    df_out["pwr_dev"] = (df_out["power_consumption"] - 200.0).abs()
    
    return df_out

def validate_telemetry_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validates that all required telemetry features are present and non-empty."""
    missing = [feat for feat in BASE_FEATURES if feat not in df.columns]
    if missing:
        return False, [f"Missing required telemetry columns: {', '.join(missing)}"]
    
    # Check for NaN / null values
    null_counts = df[BASE_FEATURES].isnull().sum()
    null_cols = null_counts[null_counts > 0].to_dict()
    if null_cols:
        return False, [f"Null values detected in columns: {null_cols}"]
        
    return True, []

def prepare_data(
    data_path: Path = RAW_DATASET_PATH,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Loads raw CSV, executes schema validation, cleans missing values,
    engineers features, fits standard scaler and label encoder,
    and returns stratified train-test splits.
    """
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")
        
    df = pd.read_csv(data_path)
    is_valid, errors = validate_telemetry_schema(df)
    if not is_valid:
        raise ValueError(f"Schema validation failed: {errors}")
        
    # Apply feature engineering
    df_proc = engineer_features(df)
    
    # Target variables
    y_raw = df_proc["anomaly_type"]
    
    # Fit label encoder on all standard anomaly classes
    label_encoder = LabelEncoder()
    label_encoder.fit(ANOMALY_CLASSES)
    y_encoded = label_encoder.transform(y_raw)
    
    X = df_proc[ALL_MODEL_FEATURES]
    
    # Stratified train-test split for reproducible academic comparison
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y_encoded, df_proc.index,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded
    )
    
    # Fit StandardScaler on training data only to prevent data leakage
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=ALL_MODEL_FEATURES,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=ALL_MODEL_FEATURES,
        index=X_test.index
    )
    
    # Save scaler and label encoder for inference
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    joblib.dump(label_encoder, MODELS_DIR / "label_encoder.joblib")
    joblib.dump(ALL_MODEL_FEATURES, MODELS_DIR / "feature_names.joblib")
    
    # Save processed CSVs
    train_df = df_proc.loc[idx_train].copy()
    test_df = df_proc.loc[idx_test].copy()
    train_df.to_csv(TRAIN_DATASET_PATH, index=False)
    test_df.to_csv(TEST_DATASET_PATH, index=False)
    
    return {
        "X_train": X_train,
        "X_test": X_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "feature_names": ALL_MODEL_FEATURES,
        "train_df": train_df,
        "test_df": test_df
    }

def transform_single_record(
    record: Dict[str, Any],
    scaler: StandardScaler,
    feature_names: List[str]
) -> np.ndarray:
    """Transforms a single telemetry dictionary record into scaled feature array for inference."""
    df_single = pd.DataFrame([record])
    df_eng = engineer_features(df_single)
    X_single = df_eng[feature_names]
    return scaler.transform(X_single)

if __name__ == "__main__":
    print("Executing ML Preprocessing pipeline...")
    prep = prepare_data()
    print(f"Features count: {len(prep['feature_names'])}")
    print(f"Features list: {prep['feature_names']}")
    print(f"Train samples: {len(prep['X_train'])}")
    print(f"Test samples: {len(prep['X_test'])}")
    print("Preprocessing completed and artifacts saved to backend/models/")
