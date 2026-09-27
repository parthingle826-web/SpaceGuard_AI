"""
SpaceGuard AI - Database Helper Module
Manages SQLite connection, schema migrations, and queries for anomalies and telemetry.
NOTE: Academic simulation only - Not real NASA/spacecraft mission data.
"""

import sqlite3
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from backend.config import DATABASE_PATH, DATABASE_DIR

logger = logging.getLogger(__name__)

def get_db_connection() -> sqlite3.Connection:
    """Creates a connection to the SQLite database with dictionary-like row factory."""
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DATABASE_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes tables using schema.sql if they do not exist."""
    schema_file = DATABASE_DIR / "schema.sql"
    if not schema_file.exists():
        logger.error(f"Schema file not found at {schema_file}")
        return

    with open(schema_file, "r", encoding="utf-8") as f:
        schema_sql = f.read()

    with get_db_connection() as conn:
        conn.executescript(schema_sql)
        conn.commit()
    logger.info(f"Database initialized at {DATABASE_PATH}")

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
    """Inserts an anomaly detection record into anomaly_history table."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    query = """
        INSERT INTO anomaly_history (
            timestamp, telemetry_snapshot, predicted_status, severity,
            anomaly_type, confidence, recommendation, model_used,
            contributing_features, source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
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

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "records": results
    }

def clear_anomaly_history():
    """Clears all records from anomaly history table."""
    with get_db_connection() as conn:
        conn.execute("DELETE FROM anomaly_history")
        conn.commit()

def log_telemetry_point(record: Dict[str, Any]) -> int:
    """Inserts a telemetry observation into telemetry_stream table."""
    query = """
        INSERT INTO telemetry_stream (
            timestamp, battery_voltage, battery_current, temperature,
            pressure, solar_panel_voltage, solar_panel_current,
            power_consumption, cpu_subsystem_temp, signal_strength,
            comm_status, radiation_level, is_anomaly, anomaly_type
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    timestamp = record.get("timestamp") or datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, (
            timestamp,
            float(record.get("battery_voltage", 28.0)),
            float(record.get("battery_current", 3.5)),
            float(record.get("temperature", 22.0)),
            float(record.get("pressure", 101.3)),
            float(record.get("solar_panel_voltage", 48.0)),
            float(record.get("solar_panel_current", 6.5)),
            float(record.get("power_consumption", 200.0)),
            float(record.get("cpu_subsystem_temp", 38.0)),
            float(record.get("signal_strength", -75.0)),
            int(record.get("comm_status", 1)),
            float(record.get("radiation_level", 0.05)),
            int(record.get("is_anomaly", 0)),
            str(record.get("anomaly_type", "Normal"))
        ))
        conn.commit()
        return cursor.lastrowid

def get_recent_telemetry(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieves recent telemetry stream records ordered by time."""
    query = """
        SELECT * FROM telemetry_stream
        ORDER BY id DESC
        LIMIT ?
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, (limit,))
        rows = cursor.fetchall()

    return [dict(row) for row in reversed(rows)]

def seed_telemetry_if_empty(seed_limit: int = 150):
    """Seeds initial telemetry stream from generated synthetic CSV if empty."""
    from backend.config import RAW_DATASET_PATH
    import pandas as pd
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM telemetry_stream")
        count = cursor.fetchone()["count"]
        
    if count == 0 and RAW_DATASET_PATH.exists():
        logger.info(f"Seeding telemetry_stream with initial {seed_limit} observations...")
        df = pd.read_csv(RAW_DATASET_PATH)
        sample_df = df.iloc[:seed_limit]
        for _, row in sample_df.iterrows():
            log_telemetry_point(row.to_dict())
        logger.info("Telemetry stream seeded successfully.")

