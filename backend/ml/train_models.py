"""
SpaceGuard AI - Model Training Module
Trains Decision Tree, SVM, and XGBoost classifiers with an extensible MODEL_REGISTRY.
NOTE: Academic simulation only - Not real NASA/spacecraft mission data.
"""

import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import joblib

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from backend.config import MODELS_DIR
from backend.ml.preprocess import prepare_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Extensible Model Registry: Adding a 4th model (e.g. Random Forest) is a one-line addition
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "Decision Tree": {
        "estimator": DecisionTreeClassifier(
            max_depth=14,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42
        ),
        "filename": "decision_tree.joblib",
        "family": "Decision Trees / CART",
        "description": "Decision Tree classifier providing transparent, hierarchical feature thresholds and native feature importances."
    },
    "SVM": {
        "estimator": SVC(
            kernel="rbf",
            C=8.0,
            gamma="scale",
            probability=True,
            random_state=42
        ),
        "filename": "svm.joblib",
        "family": "Kernel Methods / SVM",
        "description": "Support Vector Classifier with Radial Basis Function kernel projecting non-linear telemetry patterns into high-dimensional space."
    },
    "XGBoost": {
        "estimator": XGBClassifier(
            n_estimators=120,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1
        ),
        "filename": "xgboost.joblib",
        "family": "Gradient Boosted Trees",
        "description": "Extreme Gradient Boosting utilizing regularization and second-order Taylor expansion loss optimization."
    }
}

def train_all_models(save_models: bool = True) -> Dict[str, Any]:
    """
    Trains all registered models on preprocessed data, measuring training duration.
    Saves trained estimators as .joblib files in backend/models/.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading preprocessed training and validation splits...")
    prep = prepare_data()
    X_train_scaled = prep["X_train_scaled"]
    y_train = prep["y_train"]
    
    training_results = {}
    
    for model_name, config in MODEL_REGISTRY.items():
        logger.info(f"--> Training {model_name}...")
        estimator = config["estimator"]
        
        # Measure high-precision training time
        t_start = time.perf_counter()
        estimator.fit(X_train_scaled, y_train)
        train_time = time.perf_counter() - t_start
        
        logger.info(f"    Completed in {train_time:.3f} seconds.")
        
        if save_models:
            save_path = MODELS_DIR / config["filename"]
            joblib.dump(estimator, save_path)
            logger.info(f"    Saved model checkpoint to {save_path}")
            
        training_results[model_name] = {
            "estimator": estimator,
            "filename": config["filename"],
            "train_time": round(train_time, 4),
            "family": config["family"],
            "description": config["description"]
        }
        
    return training_results

if __name__ == "__main__":
    print("=" * 60)
    print("SpaceGuard AI - Training Comparative Machine Learning Models")
    print("=" * 60)
    results = train_all_models(save_models=True)
    print("\nTraining Summary:")
    for name, res in results.items():
        print(f" - {name:15}: Train Time = {res['train_time']:.4f}s | Saved as {res['filename']}")
