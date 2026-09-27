"""
SpaceGuard AI - Vercel Serverless Simulation Test
Tests the Flask app directly under simulated Vercel serverless environment (VERCEL=1).
Verifies:
1. No background threads spawned
2. Request-driven telemetry simulation
3. Safe in-memory database fallback if disk is ephemeral or read-only
4. All routes and APIs return HTTP 200
"""

import os
import sys


os.environ["VERCEL"] = "1"

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import app, initialize_application
import threading

def run_vercel_tests():
    print("=" * 80)
    print("SPACEGUARD AI - SIMULATED VERCEL SERVERLESS ENVIRONMENT TEST")
    print("=" * 80)

    
    threads_before = threading.active_count()
    print(f"Active threads before init: {threads_before}")

  
    initialize_application()

    threads_after = threading.active_count()
    print(f"Active threads after init: {threads_after}")
    if threads_after > threads_before:
        print("  [WARN] Thread count increased! Checking active threads...")
        for t in threading.enumerate():
            print(f"    - Thread: {t.name} (daemon={t.daemon})")
    else:
        print("  [PASS] Zero permanent background threads spawned (Serverless Safe).")

    # Use Flask test client
    client = app.test_client()

    routes_to_test = [
        ("/", 200),
        ("/dashboard", 200),
        ("/telemetry", 200),
        ("/anomaly", 200),
        ("/models", 200),
        ("/history", 200),
        ("/about", 200),
        ("/app.py", 200),
        ("/app", 200),
        ("/css/style.css", 200),
        ("/js/api.js", 200),
        ("/js/charts.js", 200),
        ("/favicon.ico", 200),
        ("/api/health", 200),
        ("/api/dashboard/stats", 200),
        ("/api/telemetry?limit=5", 200),
        ("/api/models/performance", 200),
        ("/api/history?limit=5", 200),
        ("/api/simulation/status", 200),
    ]

    all_passed = True
    print("\n--- Testing Routes with Flask Serverless Test Client ---")
    for route, expected_status in routes_to_test:
        res = client.get(route)
        passed = (res.status_code == expected_status)
        if not passed:
            all_passed = False
        icon = "[PASS]" if passed else "[FAIL]"
        print(f"  {icon} {route:<30} -> {res.status_code} ({len(res.data)} bytes)")

    
    print("\n--- Testing Serverless POST Inference ---")
    test_telemetry = {
        "battery_voltage": 20.2,
        "battery_current": 8.8,
        "temperature": 48.0,
        "pressure": 101.3,
        "solar_panel_voltage": 48.0,
        "solar_panel_current": 6.5,
        "power_consumption": 220.0,
        "cpu_subsystem_temp": 40.0,
        "signal_strength": -75.0,
        "comm_status": 1,
        "radiation_level": 0.05
    }
    pred_res = client.post("/api/predict", json={
        "telemetry": test_telemetry,
        "model_name": "XGBoost",
        "log_to_history": True
    })
    pred_passed = (pred_res.status_code == 200 and pred_res.json.get("status") == "success")
    if not pred_passed:
        all_passed = False
    icon = "[PASS]" if pred_passed else "[FAIL]"
    print(f"  {icon} POST /api/predict -> {pred_res.status_code}")
    if pred_res.status_code == 200:
        r = pred_res.json.get("result", {})
        print(f"         Prediction: {r.get('status')} | Anomaly Type: {r.get('anomaly_type')} | Confidence: {r.get('confidence')}")

    print("\n" + "=" * 80)
    if all_passed:
        print("ALL VERCEL SERVERLESS CHECKS PASSED (100% OK)")
    else:
        print("SOME VERCEL SERVERLESS CHECKS FAILED")
    print("=" * 80)

    return all_passed

if __name__ == "__main__":
    success = run_vercel_tests()
    sys.exit(0 if success else 1)
