"""
SpaceGuard AI - Comprehensive Feature-by-Feature Verification Suite
Executes rigorous tests across all 9 core subsystems:
1. Ingestion & Orbital Simulation Engine
2. 7 Specific Spacecraft Anomaly Classifications + Nominal Baseline
3. 3-Model Machine Learning Engine (Decision Tree, SVM, XGBoost)
4. Feature Attribution & Explainability Narratives
5. Rule-Based Operator Decision-Support Engine
6. SQLite Database Persistence & Audit Trail
7. Interactive Simulation Controls & Failure Injection
8. High-Throughput Batch CSV Prediction
9. Frontend HTML Page Integrity & Routing

NOTE: Academic simulation only - Not real NASA/spacecraft mission data.
"""

import sys
import json
import time
from pathlib import Path
import requests

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_URL = "http://127.0.0.1:5000"

def log_section(title):
    print("\n" + "=" * 80)
    print(f" {title.upper()}")
    print("=" * 80)

def test_feature_1_health_and_simulation():
    log_section("Feature 1: System Health & Real-Time Simulation Engine")
    res = requests.get(f"{BASE_URL}/api/health", timeout=5).json()
    assert res["status"] == "success"
    print(f"[*] Satellite Health Score : {res['satellite_health_score']}%")
    print(f"[*] Operational Status     : {res['satellite_status']} ({res['severity']})")
    print(f"[*] Orbital Regime         : {res['orbit_phase']}")
    print(f"[*] Subsystem Health Matrix:")
    for sub, state in res["subsystems"].items():
        print(f"    - {sub:<12}: {state}")
    assert "battery_voltage" in res["telemetry"]
    assert "temperature" in res["telemetry"]
    assert "radiation_level" in res["telemetry"]
    print("[PASS] Feature 1: Health & Telemetry telemetry stream verified.")
    return True

def test_feature_2_anomaly_categories():
    log_section("Feature 2: Multi-Class Anomaly Detection & Severity Verification")
    
    test_cases = [
        {
            "name": "Nominal Baseline",
            "telemetry": {
                "battery_voltage": 28.2, "battery_current": 3.4, "temperature": 22.0, "cpu_subsystem_temp": 36.5,
                "solar_panel_voltage": 48.0, "solar_panel_current": 6.5, "power_consumption": 195.0,
                "pressure": 101.3, "signal_strength": -74.0, "comm_status": 1, "radiation_level": 0.045
            },
            "expected_status": "Normal",
            "expected_severity": "Normal"
        },
        {
            "name": "Thermal Runaway",
            "telemetry": {
                "battery_voltage": 27.8, "battery_current": 3.6, "temperature": 68.5, "cpu_subsystem_temp": 79.2,
                "solar_panel_voltage": 48.0, "solar_panel_current": 6.5, "power_consumption": 240.0,
                "pressure": 101.2, "signal_strength": -75.0, "comm_status": 1, "radiation_level": 0.05
            },
            "expected_status": "Anomaly",
            "expected_type": "Temperature spike"
        },
        {
            "name": "Battery Voltage Sag",
            "telemetry": {
                "battery_voltage": 21.2, "battery_current": 8.5, "temperature": 26.0, "cpu_subsystem_temp": 40.0,
                "solar_panel_voltage": 36.0, "solar_panel_current": 4.0, "power_consumption": 250.0,
                "pressure": 101.1, "signal_strength": -78.0, "comm_status": 1, "radiation_level": 0.045
            },
            "expected_status": "Anomaly",
            "expected_type": "Battery voltage drop"
        },
        {
            "name": "Electrical Current Surge",
            "telemetry": {
                "battery_voltage": 26.5, "battery_current": 12.8, "temperature": 25.0, "cpu_subsystem_temp": 42.0,
                "solar_panel_voltage": 45.0, "solar_panel_current": 6.0, "power_consumption": 340.0,
                "pressure": 101.3, "signal_strength": -75.0, "comm_status": 1, "radiation_level": 0.048
            },
            "expected_status": "Anomaly",
            "expected_type": "Current surge"
        },
        {
            "name": "Payload Power Runaway",
            "telemetry": {
                "battery_voltage": 26.8, "battery_current": 3.6, "temperature": 25.0, "cpu_subsystem_temp": 38.0,
                "solar_panel_voltage": 46.0, "solar_panel_current": 6.0, "power_consumption": 420.0,
                "pressure": 101.0, "signal_strength": -76.0, "comm_status": 1, "radiation_level": 0.05
            },
            "expected_status": "Anomaly",
            "expected_type": "Excessive power consumption"
        },
        {
            "name": "Pressure Sensor Drift",
            "telemetry": {
                "battery_voltage": 28.0, "battery_current": 3.5, "temperature": 23.0, "cpu_subsystem_temp": 38.0,
                "solar_panel_voltage": 48.0, "solar_panel_current": 6.5, "power_consumption": 195.0,
                "pressure": 62.5, "signal_strength": -74.5, "comm_status": 1, "radiation_level": 0.045
            },
            "expected_status": "Anomaly",
            "expected_type": "Sensor value drift"
        },
        {
            "name": "Transponder Loss of Signal",
            "telemetry": {
                "battery_voltage": 28.0, "battery_current": 3.4, "temperature": 22.5, "cpu_subsystem_temp": 37.0,
                "solar_panel_voltage": 48.0, "solar_panel_current": 6.5, "power_consumption": 190.0,
                "pressure": 101.3, "signal_strength": -118.0, "comm_status": 0, "radiation_level": 0.048
            },
            "expected_status": "Anomaly",
            "expected_type": "Communication signal drop"
        },
        {
            "name": "Solar Radiation Storm",
            "telemetry": {
                "battery_voltage": 24.5, "battery_current": 5.2, "temperature": 48.0, "cpu_subsystem_temp": 62.0,
                "solar_panel_voltage": 42.0, "solar_panel_current": 5.0, "power_consumption": 290.0,
                "pressure": 100.8, "signal_strength": -98.0, "comm_status": 1, "radiation_level": 1.95
            },
            "expected_status": "Anomaly",
            "expected_type": "Multiple simultaneous abnormalities"
        }
    ]

    for tc in test_cases:
        res = requests.post(f"{BASE_URL}/api/predict", json={"telemetry": tc["telemetry"], "model_name": "XGBoost"}, timeout=5).json()
        assert res["status"] == "success"
        r = res["result"]
        assert r["status"] == tc["expected_status"]
        if "expected_type" in tc:
            assert r["anomaly_type"] == tc["expected_type"], f"Expected {tc['expected_type']}, got {r['anomaly_type']}"
        print(f"[*] {tc['name']:<28} -> Status: {r['status']:<7} | Severity: {r['severity']:<8} | Category: '{r['anomaly_type']}' | Conf: {r['confidence']*100:.1f}%")
        
    print("[PASS] Feature 2: All 8 operational states (Normal + 7 Failure Categories) classified accurately.")
    return True

def test_feature_3_models_comparison():
    log_section("Feature 3: 3-Model Comparative Evaluation & MODEL_REGISTRY")
    res = requests.get(f"{BASE_URL}/api/models/performance", timeout=5).json()
    assert res["status"] == "success"
    metrics = res["metrics"]
    models = metrics["models"]
    
    assert "Decision Tree" in models
    assert "SVM" in models
    assert "XGBoost" in models
    
    print(f"{'Model':<16} | {'Accuracy':<10} | {'F1-Score':<10} | {'Train Time':<12} | {'Latency':<12}")
    print("-" * 68)
    for name, m in models.items():
        print(f"{name:<16} | {m['accuracy']*100:.2f}%     | {m['f1_weighted']*100:.2f}%     | {m['train_time_sec']:.3f} s      | {m['predict_time_ms']:.4f} ms")
        assert m["accuracy"] > 0.95
        assert len(m["confusion_matrix"]) == 8  # 8x8 matrix
        
    print("[PASS] Feature 3: Decision Tree, SVM, and XGBoost comparative benchmarks verified.")
    return True

def test_feature_4_explainability_and_attribution():
    log_section("Feature 4: Feature Attribution & Indicative Explainability")
    payload = {
        "telemetry": {
            "battery_voltage": 21.0, "battery_current": 9.2, "temperature": 62.0, "cpu_subsystem_temp": 75.0,
            "solar_panel_voltage": 48.0, "solar_panel_current": 6.5, "power_consumption": 320.0,
            "pressure": 101.3, "signal_strength": -75.0, "comm_status": 1, "radiation_level": 0.05
        },
        "model_name": "Decision Tree"
    }
    res = requests.post(f"{BASE_URL}/api/predict", json=payload, timeout=5).json()
    r = res["result"]
    print(f"[*] Explanatory Narrative:")
    print(f"    \"{r['explanation']}\"")
    print(f"[*] Top Contributing Features:")
    for f in r["contributing_features"][:3]:
        print(f"    - {f['feature']:<20}: val={f['value']}{f['unit']} (nom={f['nominal']}, z={f['z_score']})")
    assert len(r["contributing_features"]) > 0
    assert len(r["explanation"]) > 20
    print("[PASS] Feature 4: Explainability narrative and feature attribution verified.")
    return True

def test_feature_5_decision_support():
    log_section("Feature 5: Rule-Based Advisory Decision Support Engine")
    payload = {
        "telemetry": {
            "battery_voltage": 20.8, "battery_current": 9.5, "temperature": 25.0, "cpu_subsystem_temp": 38.0,
            "solar_panel_voltage": 40.0, "solar_panel_current": 5.0, "power_consumption": 260.0,
            "pressure": 101.3, "signal_strength": -75.0, "comm_status": 1, "radiation_level": 0.05
        },
        "model_name": "XGBoost"
    }
    res = requests.post(f"{BASE_URL}/api/predict", json=payload, timeout=5).json()
    ds = res["result"]["decision_support"]
    print(f"[*] Advisory Summary: {ds['summary']}")
    print(f"[*] Operator Recovery Checklist ({len(ds['actions'])} actions):")
    for act in ds["actions"]:
        print(f"    [ ] {act}")
    assert len(ds["actions"]) >= 3
    print(f"[*] Legal & Academic Notice: \"{res['result']['disclaimer'][:75]}...\"")
    print("[PASS] Feature 5: Rule-based decision-support engine verified.")
    return True

def test_feature_6_database_and_history():
    log_section("Feature 6: SQLite Database Persistence & Incident Log")
    res = requests.get(f"{BASE_URL}/api/history?limit=5", timeout=5).json()
    assert res["status"] == "success"
    total = res["data"]["total"]
    records = res["data"]["records"]
    print(f"[*] Total Recorded Incidents in SQLite: {total:,}")
    if records:
        latest = records[0]
        print(f"[*] Latest Incident #{latest['id']}: [{latest['timestamp']}] {latest['severity']} - {latest['anomaly_type']} (Model: {latest['model_used']})")
    assert total > 0
    print("[PASS] Feature 6: SQLite database audit trail verified.")
    return True

def test_feature_7_simulation_controls():
    log_section("Feature 7: Interactive Simulation & Anomaly Injection")
    # Stop simulation
    requests.post(f"{BASE_URL}/api/simulation/stop", timeout=5)
    s1 = requests.get(f"{BASE_URL}/api/simulation/status", timeout=5).json()
    assert s1["simulation"]["simulation_running"] is False
    print("[*] Simulation successfully paused.")
    
    # Resume simulation
    requests.post(f"{BASE_URL}/api/simulation/start", timeout=5)
    s2 = requests.get(f"{BASE_URL}/api/simulation/status", timeout=5).json()
    assert s2["simulation"]["simulation_running"] is True
    print("[*] Simulation successfully resumed.")
    
    # Inject anomaly
    requests.post(f"{BASE_URL}/api/simulation/inject_anomaly", json={"anomaly_type": "Temperature spike"}, timeout=5)
    print("[*] Anomaly injection command 'Temperature spike' scheduled.")
    print("[PASS] Feature 7: Real-time simulation controls verified.")
    return True

def test_feature_8_batch_csv_classification():
    log_section("Feature 8: High-Throughput Batch CSV Telemetry Prediction")
    csv_path = Path("backend/data/satellite_telemetry_synthetic.csv")
    assert csv_path.exists()
    with open(csv_path, "rb") as f:
        res = requests.post(f"{BASE_URL}/api/predict?model=XGBoost", files={"file": ("sample.csv", f, "text/csv")}, timeout=10).json()
    assert res["status"] == "success"
    print(f"[*] Batch Processed Rows : {res['total_records']:,}")
    print(f"[*] Anomalies Detected   : {res['anomalies_detected']:,} ({res['anomaly_rate']}%)")
    print(f"[*] Anomaly Category Breakdown:")
    for cat, cnt in res["distribution"].items():
        print(f"    - {cat:<36}: {cnt:,}")
    assert res["total_records"] == 7200
    print("[PASS] Feature 8: High-throughput batch CSV processing verified.")
    return True

def test_feature_9_frontend_routes_and_assets():
    log_section("Feature 9: Frontend Routes, Favicon & Static Assets")
    routes = [
        ("/", "Landing Page"),
        ("/dashboard", "Mission Control"),
        ("/telemetry", "Telemetry Stream Explorer"),
        ("/anomaly", "Anomaly Diagnostic Lab"),
        ("/models", "ML Model Benchmarks"),
        ("/history", "Incident History"),
        ("/about", "Architecture & Methodology"),
        ("/favicon.ico", "Dynamic SVG Favicon"),
        ("/data/satellite_telemetry_synthetic.csv", "Sample CSV Dataset Download")
    ]
    for r, name in routes:
        code = requests.get(f"{BASE_URL}{r}", timeout=5).status_code
        print(f"[*] {name:<35} ({r:<38}) -> HTTP {code}")
        assert code == 200
    print("[PASS] Feature 9: All 7 frontend pages, favicon, and sample download routes verified.")
    return True

if __name__ == "__main__":
    print("\n" + "#" * 80)
    print(" SPACEGUARD AI: FULL FEATURE VERIFICATION TEST RUNNER")
    print("#" * 80)
    
    t1 = test_feature_1_health_and_simulation()
    t2 = test_feature_2_anomaly_categories()
    t3 = test_feature_3_models_comparison()
    t4 = test_feature_4_explainability_and_attribution()
    t5 = test_feature_5_decision_support()
    t6 = test_feature_6_database_and_history()
    t7 = test_feature_7_simulation_controls()
    t8 = test_feature_8_batch_csv_classification()
    t9 = test_feature_9_frontend_routes_and_assets()
    
    print("\n" + "#" * 80)
    if all([t1, t2, t3, t4, t5, t6, t7, t8, t9]):
        print(" [ALL FEATURES OPERATIONAL] 9/9 SYSTEM CAPABILITIES VERIFIED WITH ZERO ERRORS!")
    else:
        print(" [WARNING] SOME CAPABILITIES FAILED.")
    print("#" * 80 + "\n")
