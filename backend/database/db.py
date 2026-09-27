"""
SpaceGuard AI - Database Helper Module
Manages SQLite connection, schema migrations, and queries for anomalies and telemetry.
Provides seamless fallback to in-memory stores in serverless / Vercel environments.
NOTE: Academic simulation only - Not real NASA/spacecraft mission data.
"""

import sqlite3
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from backend.config import DATABASE_PATH, DATABASE_DIR, SCHEMA_FILE_PATH, IS_VERCEL

logger = logging.getLogger(__name__)

# In-memory stores for serverless environments (Vercel) where local disk is ephemeral or read-only
_IN_MEMORY_ANOMALY_HISTORY: List[Dict[str, Any]] = []
_IN_MEMORY_TELEMETRY_STREAM: List[Dict[str, Any]] = []
_MAX_IN_MEMORY_STREAM = 500

def get_db_connection() -> sqlite3.Connection:
    """Creates a connection to SQLite database with dict row factory, falling back to in-memory if needed."""
    try:
        DATABASE_DIR.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(DATABASE_PATH), timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        logger.warning(f"Could not connect to SQLite file at {DATABASE_PATH}: {e}. Using in-memory database.")
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        return conn

def init_db():
    """Initializes tables using schema.sql if they do not exist."""
    if not SCHEMA_FILE_PATH.exists():
        logger.error(f"Schema file not found at {SCHEMA_FILE_PATH}")
        return

    try:
        with open(SCHEMA_FILE_PATH, "r", encoding="utf-8") as f:
            schema_sql = f.read()

        with get_db_connection() as conn:
            conn.executescript(schema_sql)
            conn.commit()
        logger.info(f"Database initialized at {DATABASE_PATH} (Serverless: {IS_VERCEL})")
    except Exception as e:
        logger.warning(f"Could not run schema.sql at {DATABASE_PATH}: {e}")

def log_anomaly_record(
    telemetry_snapshot: Dict[str, Any],
    predicted_status: str,
    severity: str,
    anomaly_type: str,
    confidence: float,
    recommendation: Dict[str, Any],
    model_used: str,
    contributing_features: Optional[List[Dict[str, Any]]] = None,
    source: str = "manual",
    timestamp: Optional[str] = None
) -> int:
    """Inserts an anomaly detection record into anomaly_history table and in-memory audit log."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # In-memory record
    record_id = len(_IN_MEMORY_ANOMALY_HISTORY) + 1
    mem_record = {
        "id": record_id,
        "timestamp": timestamp,
        "telemetry_snapshot": telemetry_snapshot,
        "predicted_status": predicted_status,
        "severity": severity,
        "anomaly_type": anomaly_type,
        "confidence": round(confidence, 4),
        "recommendation": recommendation,
        "model_used": model_used,
        "contributing_features": contributing_features or [],
        "source": source
    }
    _IN_MEMORY_ANOMALY_HISTORY.insert(0, mem_record)
    if len(_IN_MEMORY_ANOMALY_HISTORY) > 200:
        _IN_MEMORY_ANOMALY_HISTORY.pop()

    # SQLite persistence
    query = """
        INSERT INTO anomaly_history (
            timestamp, telemetry_snapshot, predicted_status, severity,
            anomaly_type, confidence, recommendation, model_used,
            contributing_features, source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (
                timestamp,
                json.dumps(telemetry_snapshot),
                predicted_status,
                severity,
                anomaly_type,
                round(confidence, 4),
                json.dumps(recommendation),
                model_used,
                json.dumps(contributing_features or []),
                source
            ))
            conn.commit()
            return cursor.lastrowid
    except Exception as e:
        logger.debug(f"SQLite log_anomaly_record fallback to in-memory: {e}")
        return record_id

def get_anomaly_history(
    limit: int = 50,
    offset: int = 0,
    severity: Optional[str] = None,
    anomaly_type: Optional[str] = None
) -> Dict[str, Any]:
    """Retrieves paginated anomaly detection history with optional filters."""
    conditions = []
    params = []

    if severity and severity != "ALL":
        conditions.append("severity = ?")
        params.append(severity)
    if anomaly_type and anomaly_type != "ALL":
        conditions.append("anomaly_type = ?")
        params.append(anomaly_type)

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    count_query = f"SELECT COUNT(*) as total FROM anomaly_history {where_clause}"
    select_query = f"""
        SELECT * FROM anomaly_history
        {where_clause}
        ORDER BY timestamp DESC
        LIMIT ? OFFSET ?
    """

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(count_query, params)
            total = cursor.fetchone()["total"]

            cursor.execute(select_query, params + [limit, offset])
            rows = cursor.fetchall()

        results = []
        for row in rows:
            item = dict(row)
            try:
                item["telemetry_snapshot"] = json.loads(item["telemetry_snapshot"])
            except Exception:
                pass
            try:
                item["recommendation"] = json.loads(item["recommendation"])
            except Exception:
                pass
            try:
                item["contributing_features"] = json.loads(item["contributing_features"])
            except Exception:
                pass
            results.append(item)

        if total > 0 or not _IN_MEMORY_ANOMALY_HISTORY:
            return {
                "total": total,
                "limit": limit,
                "offset": offset,
                "records": results
            }
    except Exception as e:
        logger.debug(f"SQLite get_anomaly_history error ({e}), using in-memory store.")

    # In-memory fallback
    filtered = _IN_MEMORY_ANOMALY_HISTORY
    if severity and severity != "ALL":
        filtered = [r for r in filtered if r.get("severity") == severity]
    if anomaly_type and anomaly_type != "ALL":
        filtered = [r for r in filtered if r.get("anomaly_type") == anomaly_type]

    total = len(filtered)
    paged = filtered[offset:offset + limit]
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "records": paged
    }

def clear_anomaly_history():
    """Clears all records from anomaly history table and in-memory store."""
    global _IN_MEMORY_ANOMALY_HISTORY
    _IN_MEMORY_ANOMALY_HISTORY = []
    try:
        with get_db_connection() as conn:
            conn.execute("DELETE FROM anomaly_history")
            conn.commit()
    except Exception as e:
        logger.debug(f"SQLite clear_anomaly_history fallback: {e}")

def log_telemetry_point(record: Dict[str, Any]) -> int:
    """Inserts a telemetry observation into telemetry_stream table and in-memory buffer."""
    timestamp = record.get("timestamp") or datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    clean_record = {
        "timestamp": timestamp,
        "battery_voltage": float(record.get("battery_voltage", 28.0)),
        "battery_current": float(record.get("battery_current", 3.5)),
        "temperature": float(record.get("temperature", 22.0)),
        "pressure": float(record.get("pressure", 101.3)),
        "solar_panel_voltage": float(record.get("solar_panel_voltage", 48.0)),
        "solar_panel_current": float(record.get("solar_panel_current", 6.5)),
        "power_consumption": float(record.get("power_consumption", 200.0)),
        "cpu_subsystem_temp": float(record.get("cpu_subsystem_temp", 38.0)),
        "signal_strength": float(record.get("signal_strength", -75.0)),
        "comm_status": int(record.get("comm_status", 1)),
        "radiation_level": float(record.get("radiation_level", 0.05)),
        "is_anomaly": int(record.get("is_anomaly", 0)),
        "anomaly_type": str(record.get("anomaly_type", "Normal"))
    }
    
    _IN_MEMORY_TELEMETRY_STREAM.append(clean_record)
    if len(_IN_MEMORY_TELEMETRY_STREAM) > _MAX_IN_MEMORY_STREAM:
        _IN_MEMORY_TELEMETRY_STREAM.pop(0)

    query = """
        INSERT INTO telemetry_stream (
            timestamp, battery_voltage, battery_current, temperature,
            pressure, solar_panel_voltage, solar_panel_current,
            power_consumption, cpu_subsystem_temp, signal_strength,
            comm_status, radiation_level, is_anomaly, anomaly_type
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (
                clean_record["timestamp"],
                clean_record["battery_voltage"],
                clean_record["battery_current"],
                clean_record["temperature"],
                clean_record["pressure"],
                clean_record["solar_panel_voltage"],
                clean_record["solar_panel_current"],
                clean_record["power_consumption"],
                clean_record["cpu_subsystem_temp"],
                clean_record["signal_strength"],
                clean_record["comm_status"],
                clean_record["radiation_level"],
                clean_record["is_anomaly"],
                clean_record["anomaly_type"]
            ))
            conn.commit()
            return cursor.lastrowid
    except Exception as e:
        logger.debug(f"SQLite log_telemetry_point fallback: {e}")
        return len(_IN_MEMORY_TELEMETRY_STREAM)

def log_telemetry_batch(records: List[Dict[str, Any]]) -> int:
    """Inserts a batch of telemetry observations inside a single transaction and in-memory buffer."""
    if not records:
        return 0
    query = """
        INSERT INTO telemetry_stream (
            timestamp, battery_voltage, battery_current, temperature,
            pressure, solar_panel_voltage, solar_panel_current,
            power_consumption, cpu_subsystem_temp, signal_strength,
            comm_status, radiation_level, is_anomaly, anomaly_type
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    rows = []
    now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    for r in records:
        clean = {
            "timestamp": r.get("timestamp") or now_str,
            "battery_voltage": float(r.get("battery_voltage", 28.0)),
            "battery_current": float(r.get("battery_current", 3.5)),
            "temperature": float(r.get("temperature", 22.0)),
            "pressure": float(r.get("pressure", 101.3)),
            "solar_panel_voltage": float(r.get("solar_panel_voltage", 48.0)),
            "solar_panel_current": float(r.get("solar_panel_current", 6.5)),
            "power_consumption": float(r.get("power_consumption", 200.0)),
            "cpu_subsystem_temp": float(r.get("cpu_subsystem_temp", 38.0)),
            "signal_strength": float(r.get("signal_strength", -75.0)),
            "comm_status": int(r.get("comm_status", 1)),
            "radiation_level": float(r.get("radiation_level", 0.05)),
            "is_anomaly": int(r.get("is_anomaly", 0)),
            "anomaly_type": str(r.get("anomaly_type", "Normal"))
        }
        rows.append((
            clean["timestamp"],
            clean["battery_voltage"],
            clean["battery_current"],
            clean["temperature"],
            clean["pressure"],
            clean["solar_panel_voltage"],
            clean["solar_panel_current"],
            clean["power_consumption"],
            clean["cpu_subsystem_temp"],
            clean["signal_strength"],
            clean["comm_status"],
            clean["radiation_level"],
            clean["is_anomaly"],
            clean["anomaly_type"]
        ))
        _IN_MEMORY_TELEMETRY_STREAM.append(clean)
        if len(_IN_MEMORY_TELEMETRY_STREAM) > _MAX_IN_MEMORY_STREAM:
            _IN_MEMORY_TELEMETRY_STREAM.pop(0)

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(query, rows)
            conn.commit()
    except Exception as e:
        logger.debug(f"SQLite log_telemetry_batch fallback: {e}")
    return len(rows)

def get_recent_telemetry(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieves recent telemetry stream records ordered by time."""
    query = """
        SELECT * FROM telemetry_stream
        ORDER BY id DESC
        LIMIT ?
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (limit,))
            rows = cursor.fetchall()

        if rows:
            return [dict(row) for row in reversed(rows)]
    except Exception as e:
        logger.debug(f"SQLite get_recent_telemetry error ({e}), using in-memory store.")

    # In-memory fallback
    return list(_IN_MEMORY_TELEMETRY_STREAM[-limit:])

def seed_telemetry_if_empty(seed_limit: int = 150):
    """Seeds initial telemetry stream from generated synthetic CSV if empty."""
    from backend.config import RAW_DATASET_PATH
    import pandas as pd
    
    count = 0
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as count FROM telemetry_stream")
            count = cursor.fetchone()["count"]
    except Exception:
        count = len(_IN_MEMORY_TELEMETRY_STREAM)
        
    if count == 0 and RAW_DATASET_PATH.exists():
        logger.info(f"Seeding telemetry_stream with initial {seed_limit} observations...")
        try:
            df = pd.read_csv(RAW_DATASET_PATH)
            sample_df = df.iloc[:seed_limit]
            records = sample_df.to_dict(orient="records")
            log_telemetry_batch(records)
            logger.info("Telemetry stream seeded successfully.")
        except Exception as e:
            logger.warning(f"Error seeding telemetry stream: {e}")

def seed_demo_anomalies_if_empty():
    """Seeds realistic initial anomaly incidents for demo on fresh cold-starts."""
    count = 0
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as count FROM anomaly_history")
            count = cursor.fetchone()["count"]
    except Exception:
        count = len(_IN_MEMORY_ANOMALY_HISTORY)

    if count == 0:
        logger.info("Seeding initial demo anomaly incidents...")
        demo_incidents = [
            {
                "telemetry_snapshot": {"battery_voltage": 20.8, "battery_current": 9.2, "temperature": 28.5, "pressure": 101.2, "solar_panel_voltage": 42.0, "solar_panel_current": 4.5, "power_consumption": 260.0, "cpu_subsystem_temp": 42.0, "signal_strength": -75.0, "comm_status": 1, "radiation_level": 0.05},
                "predicted_status": "Anomaly",
                "severity": "Critical",
                "anomaly_type": "Battery voltage drop",
                "confidence": 0.998,
                "recommendation": {
                    "summary": "Severe under-voltage detected: Electrical power subsystem (EPS) bus stability threatened.",
                    "actions": [
                        "Alert operations lead: Power bus collapse danger.",
                        "Recommend switching EPS to priority power preservation profile.",
                        "Isolate auxiliary loads; sustain only Command and Data Handling (C&DH) and primary transponder.",
                        "Verify battery heater circuit is not stuck in continuous active draw.",
                        "Prepare low-power safe-hold configuration procedures."
                    ]
                },
                "model_used": "XGBoost",
                "contributing_features": [{"feature": "battery_voltage", "value": 20.8, "nominal": 28.0, "z_score": 2.2, "unit": "V"}],
                "source": "simulation",
                "timestamp": (datetime.now() - timedelta(minutes=18)).strftime("%Y-%m-%d %H:%M:%S")
            },
            {
                "telemetry_snapshot": {"battery_voltage": 27.6, "battery_current": 3.6, "temperature": 68.2, "pressure": 101.3, "solar_panel_voltage": 48.0, "solar_panel_current": 6.5, "power_consumption": 235.0, "cpu_subsystem_temp": 78.4, "signal_strength": -74.0, "comm_status": 1, "radiation_level": 0.045},
                "predicted_status": "Anomaly",
                "severity": "Critical",
                "anomaly_type": "Temperature spike",
                "confidence": 0.994,
                "recommendation": {
                    "summary": "Thermal runaway imminent: Subsystem temperatures exceed safe operational envelopes.",
                    "actions": [
                        "Immediate operator notification: Thermal dissipation threshold exceeded.",
                        "Recommend shedding thermal dissipation loads (power down high-heat instruments).",
                        "Orient satellite attitude to maximize passive radiator exposure to deep space.",
                        "Monitor thermal rise rate; if >1.5°C/min, initiate safe-mode standby.",
                        "Log thermal event in spacecraft operations logbook."
                    ]
                },
                "model_used": "XGBoost",
                "contributing_features": [{"feature": "temperature", "value": 68.2, "nominal": 22.0, "z_score": 1.9, "unit": "°C"}],
                "source": "simulation",
                "timestamp": (datetime.now() - timedelta(minutes=52)).strftime("%Y-%m-%d %H:%M:%S")
            },
            {
                "telemetry_snapshot": {"battery_voltage": 27.9, "battery_current": 3.4, "temperature": 22.4, "pressure": 101.3, "solar_panel_voltage": 48.0, "solar_panel_current": 6.5, "power_consumption": 192.0, "cpu_subsystem_temp": 37.0, "signal_strength": -118.0, "comm_status": 0, "radiation_level": 0.048},
                "predicted_status": "Anomaly",
                "severity": "Critical",
                "anomaly_type": "Communication signal drop",
                "confidence": 0.999,
                "recommendation": {
                    "summary": "Telemetry link degradation or complete loss of downlink signal.",
                    "actions": [
                        "Check ground station antenna pointing angle and ephemeris tracking accuracy.",
                        "Verify transponder power status and RF amplifier supply voltage.",
                        "Wait for orbital AOS/LOS boundary if spacecraft is transitioning between horizons.",
                        "Attempt emergency carrier sweep on backup frequency channel (S-band/UHF).",
                        "Initiate telemetry blind acquisition sequence."
                    ]
                },
                "model_used": "XGBoost",
                "contributing_features": [{"feature": "signal_strength", "value": -118.0, "nominal": -75.0, "z_score": 2.8, "unit": "dBm"}],
                "source": "simulation",
                "timestamp": (datetime.now() - timedelta(hours=2, minutes=15)).strftime("%Y-%m-%d %H:%M:%S")
            }
        ]
        for inc in demo_incidents:
            log_anomaly_record(**inc)
        logger.info("Demo anomaly incidents seeded successfully.")
