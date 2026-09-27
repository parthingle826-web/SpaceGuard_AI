"""
SpaceGuard AI - Inference & Decision Support Module
Executes predictions using selected ML models, extracts feature explainability,
and generates rule-based operator advisory recommendations.

NOTE: All recommendations are strictly decision-support suggestions for academic
simulation analysis, NOT autonomous commands dispatched to a spacecraft.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from backend.config import (
    MODELS_DIR,
    TELEMETRY_FEATURES,
    NOMINAL_RANGES,
    DECISION_SUPPORT_RULES,
    ANOMALY_CLASSES
)
from backend.ml.preprocess import engineer_features, ALL_MODEL_FEATURES

logger = logging.getLogger(__name__)

# Model Cache for Fast Sub-Millisecond Inferences
_MODEL_CACHE: Dict[str, Any] = {}
_SCALER = None
_LABEL_ENCODER = None
_FEATURE_NAMES = None

def load_inference_artifacts():
    """Loads and caches scaler, label encoder, and models in memory."""
    global _SCALER, _LABEL_ENCODER, _FEATURE_NAMES
    
    if _SCALER is None and (MODELS_DIR / "scaler.joblib").exists():
        _SCALER = joblib.load(MODELS_DIR / "scaler.joblib")
    if _LABEL_ENCODER is None and (MODELS_DIR / "label_encoder.joblib").exists():
        _LABEL_ENCODER = joblib.load(MODELS_DIR / "label_encoder.joblib")
    if _FEATURE_NAMES is None and (MODELS_DIR / "feature_names.joblib").exists():
        _FEATURE_NAMES = joblib.load(MODELS_DIR / "feature_names.joblib")
    else:
        _FEATURE_NAMES = ALL_MODEL_FEATURES

def get_model(model_name: str = "XGBoost"):
    """Loads and returns requested estimator from disk or in-memory cache."""
    load_inference_artifacts()
    
    model_filenames = {
        "Decision Tree": "decision_tree.joblib",
        "SVM": "svm.joblib",
        "XGBoost": "xgboost.joblib"
    }
    
    filename = model_filenames.get(model_name, "xgboost.joblib")
    if model_name not in _MODEL_CACHE:
        model_path = MODELS_DIR / filename
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}. Please train models first.")
        _MODEL_CACHE[model_name] = joblib.load(model_path)
        
    return _MODEL_CACHE[model_name]

def determine_severity_and_explanation(
    record: Dict[str, Any],
    predicted_type: str,
    confidence: float,
    global_importances: Optional[Dict[str, float]] = None
) -> Tuple[str, List[Dict[str, Any]], str]:
    """
    Computes severity (Normal, Warning, Critical), identifies top driving features
    by combining local deviations from nominal baselines with model feature weights,
    and constructs a hedged explanatory string.
    """
    if predicted_type == "Normal":
        return "Normal", [], "All ingested telemetry parameters remain within verified nominal operating bounds."
    
    # Calculate deviations from nominal ranges for each base parameter
    deviations = []
    for feat in TELEMETRY_FEATURES:
        if feat not in record or feat == "comm_status":
            continue
        val = float(record[feat])
        nom = NOMINAL_RANGES[feat]["nominal"]
        unit = NOMINAL_RANGES[feat]["unit"]
        min_nom = NOMINAL_RANGES[feat]["min"]
        max_nom = NOMINAL_RANGES[feat]["max"]
        
        # Nominal spread
        spread = max(abs(max_nom - nom), abs(nom - min_nom), 1.0)
        z_score = abs(val - nom) / spread
        
        # Weight by model importance if available
        weight = global_importances.get(feat, 0.05) if global_importances else 0.05
        combined_score = z_score * (1.0 + weight * 5.0)
        
        deviations.append({
            "feature": feat,
            "value": val,
            "nominal": nom,
            "unit": unit,
            "z_score": round(float(z_score), 3),
            "score": round(float(combined_score), 3)
        })
        
    # Check comm_status specifically
    comm_val = int(record.get("comm_status", 1))
    if comm_val == 0:
        deviations.append({
            "feature": "comm_status",
            "value": 0,
            "nominal": 1,
            "unit": "binary",
            "z_score": 3.0,
            "score": 4.5
        })
        
    deviations.sort(key=lambda d: d["score"], reverse=True)
    top_features = deviations[:4]
    
    # Evaluate severity based on maximum deviation and confidence
    max_z = max([d["z_score"] for d in deviations]) if deviations else 1.0
    
    # Critical criteria
    is_critical = (
        max_z >= 2.0 or
        predicted_type == "Multiple simultaneous abnormalities" and max_z >= 1.4 or
        record.get("comm_status") == 0 or
        float(record.get("radiation_level", 0.05)) > 0.8 or
        float(record.get("battery_voltage", 28.0)) < 23.0 or
        float(record.get("temperature", 22.0)) > 55.0
    )
    
    severity = "Critical" if is_critical else "Warning"
    
    # Build hedged explanatory narrative (indicative, not absolute causal claims)
    feature_highlights = []
    for f in top_features[:2]:
        feature_highlights.append(f"{f['feature'].replace('_', ' ')} ({f['value']} {f['unit']})")
        
    if feature_highlights:
        explanation = f"Observation indicates {', '.join(feature_highlights)} exhibited significant deviation from nominal baselines, which served as prominent indicators for the {predicted_type} classification."
    else:
        explanation = f"Subsystem telemetry exhibited multi-parameter anomalous divergence characteristic of {predicted_type}."
        
    return severity, top_features, explanation

def get_decision_support(anomaly_type: str, severity: str) -> Dict[str, Any]:
    """Retrieves structured advisory recommendations from the rule engine."""
    if anomaly_type == "Normal":
        return DECISION_SUPPORT_RULES.get("Normal", {
            "summary": "Nominal operations.",
            "actions": ["Maintain orbital polling cadence."]
        })
        
    rules_for_type = DECISION_SUPPORT_RULES.get(anomaly_type, {})
    recommendation = rules_for_type.get(severity) or rules_for_type.get("Warning")
    
    if not recommendation:
        recommendation = {
            "summary": f"Telemetry divergence observed for {anomaly_type}.",
            "actions": [
                "Verify subsystem health telemetry.",
                "Review recent mission command sequence.",
                "Cross-check secondary sensor readings.",
                "Consult subsystem engineering lead."
            ]
        }
    return recommendation

def predict_single_telemetry(
    record: Dict[str, Any],
    model_name: str = "XGBoost"
) -> Dict[str, Any]:
    """
    Main inference entrypoint for a single satellite telemetry record.
    Returns predicted status, severity, confidence, anomaly category,
    feature explainability, and decision-support guidance.
    """
    load_inference_artifacts()
    estimator = get_model(model_name)
    
    # Convert input to DataFrame and apply feature engineering
    df_raw = pd.DataFrame([record])
    df_eng = engineer_features(df_raw)
    
    # Ensure all required features are present
    feature_cols = _FEATURE_NAMES or ALL_MODEL_FEATURES
    X_unscaled = df_eng[feature_cols]
    X_scaled_arr = _SCALER.transform(X_unscaled)
    X_scaled = pd.DataFrame(X_scaled_arr, columns=feature_cols)
    
    # Prediction
    pred_encoded = estimator.predict(X_scaled)[0]
    predicted_type = _LABEL_ENCODER.inverse_transform([pred_encoded])[0]
    
    # Probability / Confidence
    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(X_scaled)[0]
        confidence = float(np.max(probabilities))
    else:
        confidence = 0.95
        
    # Global feature importances for explainability weighting
    global_importances = {}
    if hasattr(estimator, "feature_importances_"):
        for feat, imp in zip(feature_cols, estimator.feature_importances_):
            global_importances[feat] = float(imp)
            
    # Compute severity and explanation
    severity, top_features, explanation = determine_severity_and_explanation(
        record, predicted_type, confidence, global_importances
    )
    
    predicted_status = "Normal" if predicted_type == "Normal" else "Anomaly"
    
    # Decision support advisory actions
    decision_support = get_decision_support(predicted_type, severity)
    
    return {
        "status": predicted_status,
        "anomaly_type": predicted_type,
        "severity": severity,
        "confidence": round(confidence, 4),
        "model_used": model_name,
        "contributing_features": top_features,
        "explanation": explanation,
        "decision_support": decision_support,
        "disclaimer": "Academic simulation only - Not real NASA/spacecraft mission data. Recommendations are operator decision-support guidance, not automated spacecraft commands."
    }

def predict_batch_telemetry(
    df: pd.DataFrame,
    model_name: str = "XGBoost"
) -> pd.DataFrame:
    """Performs high-throughput batch prediction on a pandas DataFrame."""
    load_inference_artifacts()
    estimator = get_model(model_name)
    
    df_eng = engineer_features(df)
    feature_cols = _FEATURE_NAMES or ALL_MODEL_FEATURES
    X_scaled_arr = _SCALER.transform(df_eng[feature_cols])
    X_scaled = pd.DataFrame(X_scaled_arr, columns=feature_cols)
    
    pred_encoded = estimator.predict(X_scaled)
    predicted_types = _LABEL_ENCODER.inverse_transform(pred_encoded)
    
    if hasattr(estimator, "predict_proba"):
        probs = estimator.predict_proba(X_scaled)
        confidences = np.max(probs, axis=1)
    else:
        confidences = np.full(len(df), 0.95)
        
    df_results = df.copy()
    df_results["predicted_type"] = predicted_types
    df_results["predicted_status"] = ["Normal" if t == "Normal" else "Anomaly" for t in predicted_types]
    df_results["confidence"] = np.round(confidences, 4)
    df_results["model_used"] = model_name
    
    return df_results

if __name__ == "__main__":
    # Test nominal record
    nominal_test = {
        "battery_voltage": 28.2,
        "battery_current": 3.4,
        "temperature": 21.8,
        "pressure": 101.2,
        "solar_panel_voltage": 48.5,
        "solar_panel_current": 6.4,
        "power_consumption": 195.0,
        "cpu_subsystem_temp": 36.5,
        "signal_strength": -74.0,
        "comm_status": 1,
        "radiation_level": 0.04
    }
    
    # Test thermal anomaly record
    thermal_test = {
        "battery_voltage": 27.9,
        "battery_current": 3.6,
        "temperature": 68.5,  # High temperature spike
        "pressure": 101.1,
        "solar_panel_voltage": 48.0,
        "solar_panel_current": 6.5,
        "power_consumption": 220.0,
        "cpu_subsystem_temp": 78.0,  # High CPU temperature
        "signal_strength": -75.0,
        "comm_status": 1,
        "radiation_level": 0.05
    }
    
    print("Testing Nominal Record Prediction:")
    res_nom = predict_single_telemetry(nominal_test, "XGBoost")
    print(f"Status: {res_nom['status']} | Type: {res_nom['anomaly_type']} | Severity: {res_nom['severity']} | Conf: {res_nom['confidence']}")
    
    print("\nTesting Thermal Anomaly Record Prediction:")
    res_thm = predict_single_telemetry(thermal_test, "Decision Tree")
    print(f"Status: {res_thm['status']} | Type: {res_thm['anomaly_type']} | Severity: {res_thm['severity']} | Conf: {res_thm['confidence']}")
    print(f"Explanation: {res_thm['explanation']}")
    print(f"Decision Support Summary: {res_thm['decision_support']['summary']}")
    print(f"Actions: {res_thm['decision_support']['actions']}")
