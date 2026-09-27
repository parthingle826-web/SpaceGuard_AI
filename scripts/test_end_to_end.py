"""
SpaceGuard AI - End-to-End Automated Test Suite
Verifies all 7 frontend pages, DOM structures, static assets, and all 11 REST API endpoints.

NOTE: Academic simulation only - Not real NASA/spacecraft mission data.
"""

import sys
import json
import time
from pathlib import Path
import requests
from bs4 import BeautifulSoup

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_URL = "http://127.0.0.1:5000"

def test_page(path: str, expected_title_keyword: str, required_dom_selectors: list) -> bool:
    """Verifies that an HTML page returns 200, parses cleanly, and contains required DOM elements."""
    url = f"{BASE_URL}{path}"
    try:
        res = requests.get(url, timeout=5)
        if res.status_code != 200:
            print(f"[PAGE FAIL] {path}: Returned HTTP {res.status_code}")
            return False
            
        soup = BeautifulSoup(res.text, "html.parser")
        title = soup.title.string if soup.title else ""
        if expected_title_keyword.lower() not in title.lower():
            print(f"[PAGE FAIL] {path}: Title '{title}' missing keyword '{expected_title_keyword}'")
            return False
            
        missing_elements = []
        for selector in required_dom_selectors:
            if not soup.select(selector):
                missing_elements.append(selector)
                
        if missing_elements:
            print(f"[PAGE FAIL] {path}: Missing DOM elements: {missing_elements}")
            return False
            
        print(f"[PAGE PASS] {path:<16} | Title: '{title[:45]}...' | Elements verified: {len(required_dom_selectors)}")
        return True
    except Exception as e:
        print(f"[PAGE ERROR] {path}: {e}")
        return False

def test_all_pages():
    """Tests all 7 frontend HTML pages."""
    print("\n" + "=" * 75)
    print(" 1. FRONTEND PAGES & DOM STRUCTURE VERIFICATION")
    print("=" * 75)
    
    pages = [
        ("/", "SpaceGuard", ["aside.sidebar", "#topbarStatusBadge", "#heroHealthRing", "#heroHealthScore", "a[href='/dashboard']"]),
        ("/dashboard", "Mission Control", ["#missionAlertBanner", "#kpiHealthScore", "#chartBattery", "#chartThermal", "#chartPower", "#chipEPS", "#chipThermal", "#chipComms"]),
        ("/telemetry", "Telemetry Stream", ["#paramSelect", "#anomalyOnlySwitch", "#nomMin", "#nomMax", "#explorerChart", "#telemetryTableBody", "#uploadForm"]),
        ("/anomaly", "Anomaly", ["#anomalyPredictionForm", "#inputBatV", "#inputTemp", "#inputPower", "#predictionStatusBox", "#resultSeverityBadge", "#inferenceModelSelect"]),
        ("/models", "Benchmarks", ["#modelCardsRow", "#modelComparisonChart", "#latencyChart", "#featureImportanceChart", "#cmTabs", "#modelsSummaryTableBody"]),
        ("/history", "Incident History", ["#filterSeverity", "#filterType", "#historyTableBody", "#paginationInfo", "#incidentModal"]),
        ("/about", "Architecture", ["svg", ".table-space", ".decision-support-box, .space-card", "ol"])
    ]
    
    passed = 0
    for path, keyword, selectors in pages:
        if test_page(path, keyword, selectors):
            passed += 1
            
    print(f"\nFrontend Pages Result: {passed}/{len(pages)} Passed")
    return passed == len(pages)

def test_static_assets():
    """Verifies that custom CSS and JavaScript files load cleanly."""
    print("\n" + "=" * 75)
    print(" 2. STATIC ASSETS INTEGRITY VERIFICATION")
    print("=" * 75)
    assets = ["/css/style.css", "/js/api.js", "/js/charts.js"]
    passed = 0
    for asset in assets:
        res = requests.get(f"{BASE_URL}{asset}", timeout=3)
        if res.status_code == 200 and len(res.text) > 100:
            print(f"[ASSET PASS] {asset:<20} | HTTP 200 | Size: {len(res.text):,} bytes")
            passed += 1
        else:
            print(f"[ASSET FAIL] {asset:<20} | HTTP {res.status_code}")
    return passed == len(assets)

def test_rest_api_endpoints():
    """Verifies all 11 REST API endpoints with assertions."""
    print("\n" + "=" * 75)
    print(" 3. REST API ENDPOINTS & LOGIC VERIFICATION")
    print("=" * 75)
    
    # 1. Health
    r = requests.get(f"{BASE_URL}/api/health", timeout=5).json()
    assert r["status"] == "success" and "satellite_health_score" in r and "subsystems" in r
    print(f"[API PASS] GET  /api/health               | Health Score: {r['satellite_health_score']}% | Status: {r['satellite_status']}")
    
    # 2. Dashboard Stats
    r = requests.get(f"{BASE_URL}/api/dashboard/stats", timeout=5).json()
    assert r["status"] == "success" and "total_records_ingested" in r
    print(f"[API PASS] GET  /api/dashboard/stats      | Records: {r['total_records_ingested']:,} | Anomalies: {r['total_anomalies_detected']:,}")
    
    # 3. Telemetry Stream
    r = requests.get(f"{BASE_URL}/api/telemetry?limit=10", timeout=5).json()
    assert r["status"] == "success" and len(r["records"]) > 0
    print(f"[API PASS] GET  /api/telemetry            | Retained stream: {r['total']:,} rows | Sample: {r['records'][0]['battery_voltage']}V")
    
    # 4. Model Performance
    r = requests.get(f"{BASE_URL}/api/models/performance", timeout=5).json()
    assert r["status"] == "success" and "metrics" in r
    models = r["metrics"]["models"]
    assert "Decision Tree" in models and "SVM" in models and "XGBoost" in models
    print(f"[API PASS] GET  /api/models/performance   | Verified 3 Models: DT, SVM, XGBoost")
    for name, m in models.items():
        print(f"    - {name:<15}: Acc = {m['accuracy']*100:.2f}% | F1 = {m['f1_weighted']*100:.2f}% | Latency = {m['predict_time_ms']:.4f} ms")
        
    # 5. Single Inference: Nominal
    nom_payload = {
        "telemetry": {
            "battery_voltage": 28.2, "battery_current": 3.5, "temperature": 22.0, "pressure": 101.3,
            "solar_panel_voltage": 48.0, "solar_panel_current": 6.5, "power_consumption": 195.0,
            "cpu_subsystem_temp": 36.5, "signal_strength": -74.0, "comm_status": 1, "radiation_level": 0.045
        },
        "model_name": "XGBoost"
    }
    r = requests.post(f"{BASE_URL}/api/predict", json=nom_payload, timeout=5).json()
    assert r["status"] == "success" and r["result"]["status"] == "Normal"
    print(f"[API PASS] POST /api/predict (Nominal)    | Result: {r['result']['status']} ({r['result']['severity']}) | Conf: {r['result']['confidence']*100:.1f}%")
    
    # 6. Single Inference: Thermal Anomaly
    thm_payload = {
        "telemetry": {
            "battery_voltage": 27.8, "battery_current": 3.6, "temperature": 68.5, "pressure": 101.2,
            "solar_panel_voltage": 48.0, "solar_panel_current": 6.5, "power_consumption": 230.0,
            "cpu_subsystem_temp": 79.0, "signal_strength": -75.0, "comm_status": 1, "radiation_level": 0.05
        },
        "model_name": "Decision Tree"
    }
    r = requests.post(f"{BASE_URL}/api/predict", json=thm_payload, timeout=5).json()
    assert r["status"] == "success" and r["result"]["status"] == "Anomaly"
    print(f"[API PASS] POST /api/predict (Thermal)    | Result: {r['result']['status']} ({r['result']['severity']}) | Type: '{r['result']['anomaly_type']}'")
    print(f"    - Attribution: {r['result']['explanation']}")
    print(f"    - Advisory actions: {len(r['result']['decision_support']['actions'])} steps generated")
    
    # 7. Anomaly History Log
    r = requests.get(f"{BASE_URL}/api/history?limit=5", timeout=5).json()
    assert r["status"] == "success" and "records" in r["data"]
    print(f"[API PASS] GET  /api/history              | Stored SQLite incidents: {r['data']['total']:,}")
    
    # 8. Simulation Controls
    r_stop = requests.post(f"{BASE_URL}/api/simulation/stop", timeout=5).json()
    assert r_stop["status"] == "success"
    r_start = requests.post(f"{BASE_URL}/api/simulation/start", timeout=5).json()
    assert r_start["status"] == "success"
    r_stat = requests.get(f"{BASE_URL}/api/simulation/status", timeout=5).json()
    assert r_stat["status"] == "success" and r_stat["simulation"]["simulation_running"] is True
    print(f"[API PASS] SIMULATION ENGINE CONTROLS     | Start / Stop / Status verified active")
    
    # 9. Anomaly Injection
    r_inj = requests.post(f"{BASE_URL}/api/simulation/inject_anomaly", json={"anomaly_type": "Battery voltage drop"}, timeout=5).json()
    assert r_inj["status"] == "success"
    print(f"[API PASS] POST /api/simulation/inject    | Injected: 'Battery voltage drop'")
    
    # 10. Set Active Model
    r_mod = requests.post(f"{BASE_URL}/api/models/set_active", json={"model_name": "SVM"}, timeout=5).json()
    assert r_mod["status"] == "success" and r_mod["active_model"] == "SVM"
    requests.post(f"{BASE_URL}/api/models/set_active", json={"model_name": "XGBoost"}, timeout=5)
    print(f"[API PASS] POST /api/models/set_active    | Swapped active estimator to SVM then back to XGBoost")
    
    # 11. Batch CSV Upload
    raw_csv = Path("backend/data/satellite_telemetry_synthetic.csv")
    if raw_csv.exists():
        with open(raw_csv, "rb") as f:
            r_batch = requests.post(f"{BASE_URL}/api/predict?model=XGBoost", files={"file": ("test.csv", f, "text/csv")}, timeout=10).json()
            assert r_batch["status"] == "success" and r_batch["type"] == "batch"
            print(f"[API PASS] POST /api/predict (Batch CSV)  | Classified {r_batch['total_records']:,} rows | Anomalies: {r_batch['anomalies_detected']:,} ({r_batch['anomaly_rate']}%)")
            
    return True

if __name__ == "__main__":
    print("\n" + "=" * 75)
    print(" SPACEGUARD AI - COMPREHENSIVE END-TO-END VALIDATION SUITE")
    print("=" * 75)
    
    p1 = test_all_pages()
    p2 = test_static_assets()
    p3 = test_rest_api_endpoints()
    
    print("\n" + "=" * 75)
    if p1 and p2 and p3:
        print(" [SUCCESS] ALL END-TO-END TESTS PASSED WITH 100% SUCCESS!")
    else:
        print(" [WARNING] SOME TESTS FAILED. PLEASE REVIEW LOG ABOVE.")
    print("=" * 75 + "\n")
