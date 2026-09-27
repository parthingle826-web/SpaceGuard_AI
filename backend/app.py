"""
SpaceGuard AI - Flask REST API & Web Server
Satellite Telemetry Anomaly Detection and Decision Support System
NOTE: Academic simulation only - Not real NASA/spacecraft mission data.
"""

import os
import sys
import json
import time
import math
import random
import logging
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS

# Add root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.config import (
    BACKEND_DIR,
    PROJECT_ROOT,
    MODELS_DIR,
    METRICS_PATH,
    TELEMETRY_FEATURES,
    NOMINAL_RANGES,
    ANOMALY_CLASSES,
    SEVERITY_LEVELS,
    HOST,
    PORT,
    DEBUG
)
from backend.database.db import (
    init_db,
    seed_telemetry_if_empty,
    log_anomaly_record,
    get_anomaly_history,
    clear_anomaly_history,
    log_telemetry_point,
    log_telemetry_batch,
    get_recent_telemetry,
    get_db_connection
)
from backend.ml.predict import (
    predict_single_telemetry,
    predict_batch_telemetry,
    get_decision_support
)
from backend.ml.evaluate import evaluate_models

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("SpaceGuardAI")

# Initialize Flask App
FRONTEND_DIR = PROJECT_ROOT / "frontend"
app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
CORS(app)

# =========================================================================
# REAL-TIME SATELLITE TELEMETRY SIMULATION CONTROLLER
# =========================================================================
class SatelliteSimulationController:
    """
    Simulates real-time LEO orbit telemetry stream with periodic anomaly injection.
    Runs asynchronously in a background daemon thread.
    """
    def __init__(self):
        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.interval_sec = 2.5
        self.orbit_phase_rad = 0.0
        self.active_model = "XGBoost"
        self.forced_anomaly: Optional[str] = None
        
        # State tracking
        self.current_state = {
            "satellite_id": "SG-ALPHA-1",
            "orbital_altitude_km": 540.2,
            "orbital_period_min": 95.4,
            "orbit_phase": "Sunlit",
            "status": "Normal",
            "severity": "Normal",
            "anomaly_type": "Normal",
            "confidence": 0.995,
            "health_score": 98.5,
            "active_alert_message": "Satellite telemetry is within expected range.",
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "telemetry": {
                "battery_voltage": 28.2,
                "battery_current": 3.4,
                "temperature": 22.1,
                "pressure": 101.3,
                "solar_panel_voltage": 48.2,
                "solar_panel_current": 6.5,
                "power_consumption": 196.0,
                "cpu_subsystem_temp": 36.8,
                "signal_strength": -73.5,
                "comm_status": 1,
                "radiation_level": 0.045
            }
        }
        self.lock = threading.Lock()

    def generate_simulated_point(self) -> Dict[str, Any]:
        """Calculates orbital physics telemetry point with occasional anomaly injection."""
        self.orbit_phase_rad = (self.orbit_phase_rad + 0.08) % (2 * math.pi)
        sun_exposure = math.sin(self.orbit_phase_rad)
        is_sunlit = 1 if sun_exposure > -0.15 else 0
        orbit_mode = "Sunlit Pass" if is_sunlit else "Eclipse Pass"

        # Baseline physics
        solar_v = round(max(0.0, 48.0 * is_sunlit + random.gauss(0, 0.8)), 2)
        solar_i = round(max(0.0, 6.4 * is_sunlit + random.gauss(0, 0.25)), 3)
        bat_v = round(27.4 + 2.0 * math.sin(self.orbit_phase_rad) + random.gauss(0, 0.2), 3)
        bat_i = round(3.5 + 1.8 * (1 - is_sunlit) + random.gauss(0, 0.15), 3)
        temp = round(20.0 + 8.0 * math.sin(self.orbit_phase_rad - math.pi / 4) + random.gauss(0, 0.9), 2)
        cpu_temp = round(temp + 14.5 + random.gauss(0, 0.8), 2)
        pressure = round(101.3 + random.gauss(0, 0.25), 2)
        power = round(190.0 + 35.0 * is_sunlit + random.gauss(0, 6.0), 2)
        rssi = round(-74.0 + 8.0 * math.sin(self.orbit_phase_rad * 2) + random.gauss(0, 2.0), 2)
        comm_status = 1
        radiation = round(max(0.01, 0.04 + random.expovariate(70)), 4)

        point = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "battery_voltage": bat_v,
            "battery_current": bat_i,
            "temperature": temp,
            "pressure": pressure,
            "solar_panel_voltage": solar_v,
            "solar_panel_current": solar_i,
            "power_consumption": power,
            "cpu_subsystem_temp": cpu_temp,
            "signal_strength": rssi,
            "comm_status": comm_status,
            "radiation_level": radiation
        }

        # Check for forced or spontaneous anomaly injection
        inject_type = None
        if self.forced_anomaly:
            inject_type = self.forced_anomaly
            self.forced_anomaly = None
        elif random.random() < 0.16:  # 16% spontaneous anomaly tick
            inject_type = random.choice([
                "Temperature spike",
                "Battery voltage drop",
                "Current surge",
                "Excessive power consumption",
                "Sensor value drift",
                "Communication signal drop",
                "Multiple simultaneous abnormalities"
            ])

        # Apply injection modifications
        if inject_type == "Temperature spike":
            point["temperature"] += round(random.uniform(32.0, 48.0), 2)
            point["cpu_subsystem_temp"] += round(random.uniform(25.0, 40.0), 2)
        elif inject_type == "Battery voltage drop":
            point["battery_voltage"] = round(point["battery_voltage"] - random.uniform(6.0, 8.5), 3)
            point["battery_current"] = round(point["battery_current"] + random.uniform(3.0, 5.0), 3)
        elif inject_type == "Current surge":
            point["battery_current"] = round(point["battery_current"] + random.uniform(6.0, 9.5), 3)
            point["power_consumption"] = round(point["power_consumption"] + random.uniform(100.0, 160.0), 2)
        elif inject_type == "Excessive power consumption":
            point["power_consumption"] = round(point["power_consumption"] + random.uniform(150.0, 240.0), 2)
            point["battery_voltage"] = round(point["battery_voltage"] - random.uniform(3.0, 5.0), 3)
        elif inject_type == "Sensor value drift":
            point["pressure"] = round(point["pressure"] - random.uniform(22.0, 36.0), 2)
        elif inject_type == "Communication signal drop":
            point["signal_strength"] = round(point["signal_strength"] - random.uniform(35.0, 50.0), 2)
            point["comm_status"] = 0
        elif inject_type == "Multiple simultaneous abnormalities":
            point["radiation_level"] = round(point["radiation_level"] + random.uniform(1.2, 2.4), 4)
            point["temperature"] += round(random.uniform(20.0, 35.0), 2)
            point["battery_voltage"] = round(point["battery_voltage"] - random.uniform(4.0, 6.0), 3)
            point["signal_strength"] = round(point["signal_strength"] - random.uniform(25.0, 40.0), 2)

        return point, orbit_mode

    def _worker(self):
        """Simulation loop running on background thread."""
        logger.info("Satellite telemetry simulation loop started.")
        while not self.stop_event.is_set():
            try:
                point, orbit_mode = self.generate_simulated_point()
                
                # Run inference using the active ML model
                pred_result = predict_single_telemetry(point, model_name=self.active_model)
                
                status = pred_result["status"]
                severity = pred_result["severity"]
                anom_type = pred_result["anomaly_type"]
                confidence = pred_result["confidence"]
                
                # Log telemetry to stream table
                point["is_anomaly"] = 1 if status == "Anomaly" else 0
                point["anomaly_type"] = anom_type
                log_telemetry_point(point)
                
                # If anomaly detected, log to anomaly_history
                if status == "Anomaly":
                    log_anomaly_record(
                        telemetry_snapshot=point,
                        predicted_status=status,
                        severity=severity,
                        anomaly_type=anom_type,
                        confidence=confidence,
                        recommendation=pred_result["decision_support"],
                        model_used=self.active_model,
                        contributing_features=pred_result["contributing_features"],
                        source="simulation",
                        timestamp=point["timestamp"]
                    )
                
                # Update alert message
                if severity == "Critical":
                    alert_msg = f"CRITICAL: {anom_type.upper()} detected. Telemetry indicates severe subsystem departure."
                elif severity == "Warning":
                    alert_msg = f"WARNING: Parameter deviation flagged for {anom_type}. Subsystem watch advised."
                else:
                    alert_msg = "Satellite telemetry is within expected range."

                # Update in-memory state
                with self.lock:
                    self.current_state.update({
                        "orbit_phase": orbit_mode,
                        "status": status,
                        "severity": severity,
                        "anomaly_type": anom_type,
                        "confidence": confidence,
                        "active_alert_message": alert_msg,
                        "last_updated": point["timestamp"],
                        "telemetry": point,
                        "explanation": pred_result.get("explanation", ""),
                        "decision_support": pred_result.get("decision_support", {})
                    })
                    
            except Exception as e:
                logger.error(f"Error in simulation loop: {e}", exc_info=True)

            self.stop_event.wait(self.interval_sec)

        logger.info("Satellite telemetry simulation loop stopped.")

    def start(self):
        """Starts real-time simulation thread."""
        with self.lock:
            if not self.is_running:
                self.is_running = True
                self.stop_event.clear()
                self.thread = threading.Thread(target=self._worker, daemon=True)
                self.thread.start()
                return True
            return False

    def stop(self):
        """Stops real-time simulation thread."""
        with self.lock:
            if self.is_running:
                self.is_running = False
                self.stop_event.set()
                if self.thread and self.thread.is_alive():
                    self.thread.join(timeout=2.0)
                return True
            return False

    def trigger_anomaly(self, anomaly_type: str):
        """Forces an anomaly injection on the next tick."""
        with self.lock:
            self.forced_anomaly = anomaly_type

    def get_status(self) -> Dict[str, Any]:
        """Returns current simulation state."""
        with self.lock:
            state = dict(self.current_state)
            state["simulation_running"] = self.is_running
            state["active_model"] = self.active_model
            return state

simulation_controller = SatelliteSimulationController()

# =========================================================================
# CENTRALIZED ERROR HANDLERS
# =========================================================================
@app.errorhandler(400)
def bad_request(error):
    return jsonify({
        "status": "error",
        "code": 400,
        "message": getattr(error, "description", "Bad Request")
    }), 400

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "status": "error",
        "code": 404,
        "message": "The requested API endpoint or resource was not found."
    }), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({
        "status": "error",
        "code": 500,
        "message": "An internal server error occurred while processing the request."
    }), 500

# =========================================================================
# PAGE ROUTING (Serves frontend static views)
# =========================================================================
@app.route("/favicon.ico")
def favicon():
    """Serves a dynamic SVG satellite favicon to prevent browser 404 errors."""
    svg_icon = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
        <circle cx="32" cy="32" r="30" fill="#070a13" stroke="#00f2fe" stroke-width="2"/>
        <path d="M22 22 L42 42 M20 28 L28 20 M36 44 L44 36 M26 26 L38 38" stroke="#00f2fe" stroke-width="3" stroke-linecap="round"/>
        <circle cx="32" cy="32" r="5" fill="#38bdf8"/>
    </svg>"""
    return svg_icon, 200, {"Content-Type": "image/svg+xml"}

@app.route("/data/<path:filename>")
def download_data_file(filename):
    """Serves data files such as sample CSVs for download."""
    from backend.config import DATA_DIR
    target = DATA_DIR / filename
    if not target.exists():
        return jsonify({"status": "error", "message": "File not found"}), 404
    return send_file(target, as_attachment=True)

@app.route("/")
def index_page():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/dashboard")
def dashboard_page():
    return send_from_directory(FRONTEND_DIR, "dashboard.html")

@app.route("/telemetry")
def telemetry_page():
    return send_from_directory(FRONTEND_DIR, "telemetry.html")

@app.route("/anomaly")
def anomaly_page():
    return send_from_directory(FRONTEND_DIR, "anomaly.html")

@app.route("/models")
def models_page():
    return send_from_directory(FRONTEND_DIR, "models.html")

@app.route("/history")
def history_page():
    return send_from_directory(FRONTEND_DIR, "history.html")

@app.route("/about")
def about_page():
    return send_from_directory(FRONTEND_DIR, "about.html")

# =========================================================================
# REST API ENDPOINTS
# =========================================================================

@app.route("/api/health", methods=["GET"])
def get_health():
    """Returns current satellite telemetry snapshot, health score, and subsystem ratings."""
    state = simulation_controller.get_status()
    
    # Calculate health score dynamically from recent telemetry stream
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT is_anomaly FROM telemetry_stream ORDER BY id DESC LIMIT 50")
        rows = cursor.fetchall()
        
    if rows:
        anomaly_count = sum(r["is_anomaly"] for r in rows)
        total_recent = len(rows)
        health_score = round(((total_recent - anomaly_count) / total_recent) * 100, 1)
    else:
        health_score = 98.5

    # Subsystem ratings
    tel = state["telemetry"]
    subsystems = {
        "EPS": "Optimal" if 24.5 <= tel.get("battery_voltage", 28) <= 31.0 and tel.get("battery_current", 3) < 8.0 else ("Warning" if tel.get("battery_voltage", 28) > 22.0 else "Critical"),
        "Thermal": "Optimal" if -5 <= tel.get("temperature", 22) <= 45 and tel.get("cpu_subsystem_temp", 36) <= 55 else ("Warning" if tel.get("temperature", 22) <= 55 else "Critical"),
        "Comms": "Optimal" if tel.get("comm_status", 1) == 1 and tel.get("signal_strength", -75) > -90 else ("Warning" if tel.get("comm_status", 1) == 1 else "Critical"),
        "Avionics": "Optimal" if 98.0 <= tel.get("pressure", 101.3) <= 104.0 else "Warning",
        "Radiation": "Nominal" if tel.get("radiation_level", 0.04) < 0.25 else ("Elevated" if tel.get("radiation_level", 0.04) < 0.8 else "Severe")
    }

    return jsonify({
        "status": "success",
        "satellite_health_score": health_score,
        "satellite_status": state["status"],
        "severity": state["severity"],
        "anomaly_type": state["anomaly_type"],
        "alert_message": state["active_alert_message"],
        "confidence": state["confidence"],
        "orbit_phase": state["orbit_phase"],
        "last_updated": state["last_updated"],
        "telemetry": tel,
        "subsystems": subsystems,
        "simulation_running": state["simulation_running"],
        "disclaimer": "Academic simulation only - Not real NASA/spacecraft mission data."
    })

@app.route("/api/dashboard/stats", methods=["GET"])
def get_dashboard_stats():
    """Returns top KPI statistics, counts, and anomaly distribution for dashboard cards."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Telemetry stream counts
        cursor.execute("SELECT COUNT(*) as total, SUM(is_anomaly) as anomalies FROM telemetry_stream")
        stream_row = cursor.fetchone()
        total_stream = stream_row["total"] or 0
        total_anomalies_stream = stream_row["anomalies"] or 0
        
        # Anomaly history counts
        cursor.execute("SELECT COUNT(*) as total FROM anomaly_history")
        history_count = cursor.fetchone()["total"]
        
        cursor.execute("SELECT severity, COUNT(*) as count FROM anomaly_history GROUP BY severity")
        severity_counts = {r["severity"]: r["count"] for r in cursor.fetchall()}
        
        cursor.execute("SELECT anomaly_type, COUNT(*) as count FROM anomaly_history GROUP BY anomaly_type")
        category_counts = {r["anomaly_type"]: r["count"] for r in cursor.fetchall()}

    health_ratio = ((total_stream - total_anomalies_stream) / total_stream * 100) if total_stream > 0 else 98.0
    
    return jsonify({
        "status": "success",
        "health_score": round(health_ratio, 1),
        "total_records_ingested": total_stream,
        "total_anomalies_detected": history_count,
        "critical_alerts": severity_counts.get("Critical", 0),
        "warning_alerts": severity_counts.get("Warning", 0),
        "anomaly_rate_percent": round((total_anomalies_stream / total_stream * 100), 2) if total_stream > 0 else 0.0,
        "anomaly_distribution": category_counts,
        "active_model": simulation_controller.active_model,
        "simulation_active": simulation_controller.is_running,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

@app.route("/api/telemetry", methods=["GET"])
def get_telemetry():
    """
    Retrieves telemetry records from database with optional filters.
    Query parameters:
    - limit: int (default 60, max 300)
    - offset: int (default 0)
    - parameter: string (e.g. 'temperature', 'battery_voltage', 'all')
    - anomaly_only: 'true'/'false'
    """
    limit = min(int(request.args.get("limit", 60)), 300)
    offset = int(request.args.get("offset", 0))
    parameter = request.args.get("parameter", "all")
    anomaly_only = request.args.get("anomaly_only", "false").lower() in ("true", "1")

    conditions = []
    params = []

    if anomaly_only:
        conditions.append("is_anomaly = 1")

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Total count
        cursor.execute(f"SELECT COUNT(*) as total FROM telemetry_stream {where_clause}", params)
        total = cursor.fetchone()["total"]
        
        # Select records
        query = f"""
            SELECT * FROM telemetry_stream
            {where_clause}
            ORDER BY id DESC
            LIMIT ? OFFSET ?
        """
        cursor.execute(query, params + [limit, offset])
        rows = cursor.fetchall()

    records = [dict(r) for r in reversed(rows)]
    
    return jsonify({
        "status": "success",
        "total": total,
        "limit": limit,
        "offset": offset,
        "parameter_filter": parameter,
        "nominal_ranges": NOMINAL_RANGES,
        "records": records
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Performs anomaly inference on single telemetry record or batch CSV.
    Payload:
    - JSON: { "telemetry": { ... }, "model_name": "XGBoost", "log_to_history": true }
    OR
    - File: CSV file upload with parameter 'file'
    """
    model_name = request.args.get("model", "XGBoost")
    
    # Handle CSV File Batch Upload
    if "file" in request.files:
        file = request.files["file"]
        if not file.filename.endswith(".csv"):
            return jsonify({"status": "error", "message": "Only .csv files are supported."}), 400
            
        try:
            df = pd.read_csv(file)
            missing = [f for f in TELEMETRY_FEATURES if f not in df.columns]
            if missing:
                return jsonify({
                    "status": "error",
                    "message": f"Uploaded CSV missing required telemetry columns: {', '.join(missing)}"
                }), 400
                
            pred_df = predict_batch_telemetry(df, model_name=model_name)
            
            # Anomaly summary
            anom_counts = pred_df["predicted_type"].value_counts().to_dict()
            total_rows = len(pred_df)
            anom_total = int((pred_df["predicted_status"] == "Anomaly").sum())
            
            # Sample preview
            preview = pred_df.head(25).to_dict(orient="records")
            
            return jsonify({
                "status": "success",
                "type": "batch",
                "total_records": total_rows,
                "anomalies_detected": anom_total,
                "anomaly_rate": round(anom_total / total_rows * 100, 2),
                "model_used": model_name,
                "distribution": anom_counts,
                "preview": preview
            })
            
        except Exception as e:
            logger.error(f"Error processing CSV prediction: {e}")
            return jsonify({"status": "error", "message": f"Failed to parse CSV: {str(e)}"}), 400

    # Handle JSON Single Record
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"status": "error", "message": "No JSON payload or CSV file provided."}), 400

    telemetry = data.get("telemetry", data)
    requested_model = data.get("model_name", model_name)
    log_to_history = data.get("log_to_history", True)

    # Validate presence of required features with graceful defaults
    validated_telemetry = {}
    for feat in TELEMETRY_FEATURES:
        if feat in telemetry:
            try:
                validated_telemetry[feat] = float(telemetry[feat]) if feat != "comm_status" else int(telemetry[feat])
            except (ValueError, TypeError):
                return jsonify({"status": "error", "message": f"Invalid numerical value for field: {feat}"}), 400
        else:
            # Fallback to nominal default
            validated_telemetry[feat] = NOMINAL_RANGES[feat]["nominal"]

    try:
        result = predict_single_telemetry(validated_telemetry, model_name=requested_model)
        
        # Log to SQLite history if requested
        if log_to_history:
            log_anomaly_record(
                telemetry_snapshot=validated_telemetry,
                predicted_status=result["status"],
                severity=result["severity"],
                anomaly_type=result["anomaly_type"],
                confidence=result["confidence"],
                recommendation=result["decision_support"],
                model_used=requested_model,
                contributing_features=result["contributing_features"],
                source="manual"
            )
            
        return jsonify({
            "status": "success",
            "type": "single",
            "result": result
        })
        
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        return jsonify({"status": "error", "message": f"Inference execution failed: {str(e)}"}), 500

@app.route("/api/models/performance", methods=["GET"])
def get_model_performance():
    """Returns comparative metrics, confusion matrices, and feature importances for all 3 models."""
    try:
        if METRICS_PATH.exists():
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        else:
            logger.info("metrics.json not found. Running model evaluation...")
            metrics = evaluate_models()

        return jsonify({
            "status": "success",
            "metrics": metrics
        })
    except Exception as e:
        logger.error(f"Failed to fetch model metrics: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/history", methods=["GET"])
def get_history():
    """Retrieves logged anomaly history from SQLite database with pagination and filters."""
    limit = min(int(request.args.get("limit", 50)), 200)
    offset = int(request.args.get("offset", 0))
    severity = request.args.get("severity", "ALL")
    anomaly_type = request.args.get("type", "ALL")

    history = get_anomaly_history(limit=limit, offset=offset, severity=severity, anomaly_type=anomaly_type)
    return jsonify({
        "status": "success",
        "data": history
    })

@app.route("/api/history/clear", methods=["POST"])
def clear_history():
    """Clears all anomaly history records from SQLite database."""
    clear_anomaly_history()
    return jsonify({
        "status": "success",
        "message": "Anomaly history cleared successfully."
    })

@app.route("/api/dataset/upload", methods=["POST"])
def upload_dataset():
    """Handles CSV telemetry upload, validates schema, and ingests into stream."""
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file part in upload request."}), 400

    file = request.files["file"]
    if not file.filename.endswith(".csv"):
        return jsonify({"status": "error", "message": "Only CSV files are accepted."}), 400

    try:
        df = pd.read_csv(file)
        missing = [f for f in TELEMETRY_FEATURES if f not in df.columns]
        if missing:
            return jsonify({
                "status": "error",
                "message": f"CSV is missing required telemetry columns: {', '.join(missing)}"
            }), 400

        # Ingest records into database telemetry stream via high-performance batch insert
        records = df.to_dict(orient="records")
        ingested = log_telemetry_batch(records)

        return jsonify({
            "status": "success",
            "message": f"Successfully validated and ingested {ingested} telemetry observations.",
            "total_rows": len(df),
            "columns": list(df.columns)
        })

    except Exception as e:
        logger.error(f"Failed to process uploaded dataset: {e}")
        return jsonify({"status": "error", "message": f"Upload processing failed: {str(e)}"}), 500

@app.route("/api/simulation/start", methods=["POST"])
def start_simulation():
    """Starts the real-time telemetry simulation engine."""
    started = simulation_controller.start()
    return jsonify({
        "status": "success",
        "started": started,
        "message": "Real-time telemetry simulation started." if started else "Simulation already active."
    })

@app.route("/api/simulation/stop", methods=["POST"])
def stop_simulation():
    """Stops the real-time telemetry simulation engine."""
    stopped = simulation_controller.stop()
    return jsonify({
        "status": "success",
        "stopped": stopped,
        "message": "Real-time telemetry simulation stopped." if stopped else "Simulation was not active."
    })

@app.route("/api/simulation/status", methods=["GET"])
def simulation_status():
    """Returns simulation running status and current simulated state."""
    return jsonify({
        "status": "success",
        "simulation": simulation_controller.get_status()
    })

@app.route("/api/simulation/inject_anomaly", methods=["POST"])
def inject_simulation_anomaly():
    """Forces an anomaly injection on the next simulation tick."""
    data = request.get_json(silent=True) or {}
    anomaly_type = data.get("anomaly_type", "Temperature spike")
    if anomaly_type not in ANOMALY_CLASSES:
        return jsonify({
            "status": "error",
            "message": f"Invalid anomaly type. Allowed: {ANOMALY_CLASSES}"
        }), 400

    simulation_controller.trigger_anomaly(anomaly_type)
    return jsonify({
        "status": "success",
        "message": f"Injected '{anomaly_type}' anomaly scheduled for next simulation cycle."
    })

@app.route("/api/models/set_active", methods=["POST"])
def set_active_model():
    """Sets active model for real-time simulation."""
    data = request.get_json(silent=True) or {}
    model_name = data.get("model_name")
    if model_name not in ["Decision Tree", "SVM", "XGBoost"]:
        return jsonify({"status": "error", "message": "Invalid model name."}), 400

    simulation_controller.active_model = model_name
    return jsonify({
        "status": "success",
        "active_model": model_name
    })

# =========================================================================
# APPLICATION ENTRYPOINT
# =========================================================================
def initialize_application():
    """Initializes database schema and seeds initial telemetry data."""
    init_db()
    seed_telemetry_if_empty(seed_limit=150)
    # Start simulation by default so dashboard is immediately live
    simulation_controller.start()
    logger.info("SpaceGuard AI Application initialized successfully.")

if __name__ == "__main__":
    initialize_application()
    print("=" * 70)
    print("SpaceGuard AI - Academic Satellite Telemetry Anomaly Detection")
    print(f"Mission Control Dashboard running at: http://{HOST}:{PORT}")
    print("DISCLAIMER: Academic simulation only - Not real NASA/mission data.")
    print("=" * 70)
    app.run(host=HOST, port=PORT, debug=DEBUG, use_reloader=False)
