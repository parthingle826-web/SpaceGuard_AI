-- SpaceGuard AI Database Schema
-- Academic Simulation / Satellite Telemetry Decision Support System
-- NOTE: Academic simulation only - Not real NASA/spacecraft mission data.

CREATE TABLE IF NOT EXISTS anomaly_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    telemetry_snapshot TEXT NOT NULL,
    predicted_status TEXT NOT NULL,
    severity TEXT NOT NULL,
    anomaly_type TEXT NOT NULL,
    confidence REAL NOT NULL,
    recommendation TEXT NOT NULL,
    model_used TEXT NOT NULL,
    contributing_features TEXT,
    source TEXT DEFAULT 'manual'
);

CREATE TABLE IF NOT EXISTS telemetry_stream (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    battery_voltage REAL NOT NULL,
    battery_current REAL NOT NULL,
    temperature REAL NOT NULL,
    pressure REAL NOT NULL,
    solar_panel_voltage REAL NOT NULL,
    solar_panel_current REAL NOT NULL,
    power_consumption REAL NOT NULL,
    cpu_subsystem_temp REAL NOT NULL,
    signal_strength REAL NOT NULL,
    comm_status INTEGER NOT NULL,
    radiation_level REAL NOT NULL,
    is_anomaly INTEGER DEFAULT 0,
    anomaly_type TEXT DEFAULT 'Normal'
);

CREATE INDEX IF NOT EXISTS idx_anomaly_timestamp ON anomaly_history(timestamp);
CREATE INDEX IF NOT EXISTS idx_anomaly_severity ON anomaly_history(severity);
CREATE INDEX IF NOT EXISTS idx_telemetry_timestamp ON telemetry_stream(timestamp);
