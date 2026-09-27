"""
SpaceGuard AI - Comprehensive Smoke Test Suite
Validates all HTML page routes, API endpoints, static assets, and ML model inference.
"""

import urllib.request
import urllib.parse
import json
import time
import sys

BASE_URL = "http://127.0.0.1:5000"

PAGES = [
    ("/", "Index / Mission Overview"),
    ("/dashboard", "Telemetry Operations Dashboard"),
    ("/telemetry", "Telemetry Stream Explorer"),
    ("/anomaly", "Anomaly Detection & Decision Support"),
    ("/models", "ML Model Evaluation Lab"),
    ("/history", "Anomaly History & Incident Log"),
    ("/about", "Architecture & Academic Methodology"),
    ("/app.py", "Vercel /app.py Entrypoint Fallback"),
    ("/app", "Vercel /app Entrypoint Fallback"),
]

STATIC_FILES = [
    ("/css/style.css", "CSS Stylesheet"),
    ("/js/api.js", "Frontend API Layer"),
    ("/js/charts.js", "Chart.js Helper Functions"),
    ("/favicon.ico", "Dynamic Satellite SVG Favicon"),
    ("/data/satellite_telemetry_synthetic.csv", "Sample Telemetry Dataset"),
]

APIS = [
    ("/api/health", "GET", None, "Satellite Telemetry & Health Score"),
    ("/api/dashboard/stats", "GET", None, "Dashboard Overview KPI Stats"),
    ("/api/telemetry?limit=10", "GET", None, "Telemetry Observations Stream"),
    ("/api/models/performance", "GET", None, "Model Metrics & Feature Importance"),
    ("/api/history?limit=5", "GET", None, "Logged Anomaly Incidents"),
    ("/api/simulation/status", "GET", None, "Simulation Engine Status"),
]

def test_endpoint(path, method="GET", data=None, headers=None):
    url = f"{BASE_URL}{path}"
    req = urllib.request.Request(url, method=method)
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    if data:
        req.data = data.encode("utf-8") if isinstance(data, str) else data

    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            status = response.status
            content = response.read()
            elapsed_ms = round((time.time() - start) * 1000, 1)
            return status, len(content), content, elapsed_ms, None
    except urllib.error.HTTPError as e:
        elapsed_ms = round((time.time() - start) * 1000, 1)
        return e.code, 0, e.read(), elapsed_ms, str(e)
    except Exception as e:
        elapsed_ms = round((time.time() - start) * 1000, 1)
        return 0, 0, b"", elapsed_ms, str(e)

def run_tests():
    print("=" * 80)
    print("SPACEGUARD AI - LOCAL SMOKE TEST SUITE")
    print(f"Target Server: {BASE_URL}")
    print("=" * 80)

    results = {"pages": [], "static": [], "apis": [], "predictions": []}
    all_passed = True

    # 1. Test HTML Pages
    print("\n--- 1. Testing HTML Page Routes ---")
    for path, desc in PAGES:
        status, length, content, elapsed, err = test_endpoint(path)
        passed = (status == 200 and length > 200)
        if not passed:
            all_passed = False
        results["pages"].append({
            "route": path,
            "description": desc,
            "status": status,
            "bytes": length,
            "latency_ms": elapsed,
            "passed": passed,
            "error": err
        })
        icon = "[PASS]" if passed else "[FAIL]"
        print(f"  {icon} {path:<15} {status} ({length:>6} bytes, {elapsed:>5} ms) - {desc}")

    # 2. Test Static Assets
    print("\n--- 2. Testing Static Files ---")
    for path, desc in STATIC_FILES:
        status, length, content, elapsed, err = test_endpoint(path)
        passed = (status == 200 and length > 50)
        if not passed:
            all_passed = False
        results["static"].append({
            "route": path,
            "description": desc,
            "status": status,
            "bytes": length,
            "latency_ms": elapsed,
            "passed": passed,
            "error": err
        })
        icon = "[PASS]" if passed else "[FAIL]"
        print(f"  {icon} {path:<40} {status} ({length:>6} bytes, {elapsed:>5} ms) - {desc}")

    # 3. Test API Endpoints
    print("\n--- 3. Testing API Endpoints ---")
    for path, method, payload, desc in APIS:
        status, length, content, elapsed, err = test_endpoint(path, method=method, data=payload)
        parsed_json = None
        passed = (status == 200)
        try:
            parsed_json = json.loads(content.decode("utf-8"))
            if parsed_json.get("status") != "success":
                passed = False
        except Exception:
            passed = False
            
        if not passed:
            all_passed = False
        results["apis"].append({
            "endpoint": path,
            "method": method,
            "status": status,
            "status_field": parsed_json.get("status") if parsed_json else None,
            "latency_ms": elapsed,
            "passed": passed,
            "error": err
        })
        icon = "[PASS]" if passed else "[FAIL]"
        status_info = f"status={parsed_json.get('status')}" if parsed_json else f"err={err}"
        print(f"  {icon} {method} {path:<30} {status} ({elapsed:>5} ms) - {status_info} ({desc})")

    # 4. Test Single & Batch Inference
    print("\n--- 4. Testing ML Model Prediction APIs ---")
    single_payload = json.dumps({
        "telemetry": {
            "battery_voltage": 20.5,
            "battery_current": 8.5,
            "temperature": 45.0,
            "pressure": 101.3,
            "solar_panel_voltage": 48.0,
            "solar_panel_current": 6.5,
            "power_consumption": 220.0,
            "cpu_subsystem_temp": 40.0,
            "signal_strength": -75.0,
            "comm_status": 1,
            "radiation_level": 0.05
        },
        "model_name": "XGBoost",
        "log_to_history": False
    })
    status, length, content, elapsed, err = test_endpoint(
        "/api/predict",
        method="POST",
        data=single_payload,
        headers={"Content-Type": "application/json"}
    )
    passed = (status == 200)
    prediction_result = None
    try:
        res = json.loads(content.decode("utf-8"))
        if res.get("status") == "success":
            prediction_result = res.get("result", {})
        else:
            passed = False
    except Exception:
        passed = False

    if not passed:
        all_passed = False
    icon = "[PASS]" if passed else "[FAIL]"
    print(f"  {icon} POST /api/predict (Single Telemetry JSON) -> {status} ({elapsed:>5} ms)")
    if prediction_result:
        print(f"         Prediction: {prediction_result.get('status')} | Anomaly Type: {prediction_result.get('anomaly_type')} | Severity: {prediction_result.get('severity')} | Confidence: {prediction_result.get('confidence')}")

    # Summary
    print("\n" + "=" * 80)
    if all_passed:
        print("ALL SMOKE TESTS PASSED PERFECTLY (100% HTTP 200 OK)")
    else:
        print("SOME SMOKE TESTS FAILED - CHECK OUTPUT ABOVE")
    print("=" * 80)
    
    return all_passed, results

if __name__ == "__main__":
    passed, res = run_tests()
    sys.exit(0 if passed else 1)
