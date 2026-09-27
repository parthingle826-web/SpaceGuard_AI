"""
SpaceGuard AI - Configuration Module
Academic Simulation & Decision Support System
NOTE: Academic simulation only - Not real NASA/spacecraft mission data.
"""

import os
from pathlib import Path

# Base Paths
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
DATA_DIR = BACKEND_DIR / "data"
MODELS_DIR = BACKEND_DIR / "models"
DATABASE_DIR = BACKEND_DIR / "database"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATABASE_DIR.mkdir(parents=True, exist_ok=True)

# Database path
DATABASE_PATH = DATABASE_DIR / "spaceguard.db"

# Data file paths
RAW_DATASET_PATH = DATA_DIR / "satellite_telemetry_synthetic.csv"
TRAIN_DATASET_PATH = DATA_DIR / "telemetry_train_processed.csv"
TEST_DATASET_PATH = DATA_DIR / "telemetry_test_processed.csv"
METRICS_PATH = MODELS_DIR / "metrics.json"

# Server configuration
PORT = int(os.environ.get("PORT", 5000))
HOST = os.environ.get("HOST", "127.0.0.1")
DEBUG = os.environ.get("FLASK_DEBUG", "True").lower() in ("true", "1")

# Telemetry Feature Definitions with Nominal Ranges
# Used for input validation, baseline comparison, and anomaly simulation
TELEMETRY_FEATURES = [
    "battery_voltage",
    "battery_current",
    "temperature",
    "pressure",
    "solar_panel_voltage",
    "solar_panel_current",
    "power_consumption",
    "cpu_subsystem_temp",
    "signal_strength",
    "comm_status",
    "radiation_level"
]

NOMINAL_RANGES = {
    "battery_voltage": {"min": 24.0, "nominal": 28.0, "max": 32.0, "unit": "V"},
    "battery_current": {"min": 1.0, "nominal": 3.5, "max": 8.0, "unit": "A"},
    "temperature": {"min": -10.0, "nominal": 22.0, "max": 50.0, "unit": "°C"},
    "pressure": {"min": 85.0, "nominal": 101.3, "max": 105.0, "unit": "kPa"},
    "solar_panel_voltage": {"min": 35.0, "nominal": 48.0, "max": 55.0, "unit": "V"},
    "solar_panel_current": {"min": 2.0, "nominal": 6.5, "max": 9.5, "unit": "A"},
    "power_consumption": {"min": 120.0, "nominal": 200.0, "max": 280.0, "unit": "W"},
    "cpu_subsystem_temp": {"min": 20.0, "nominal": 38.0, "max": 65.0, "unit": "°C"},
    "signal_strength": {"min": -95.0, "nominal": -75.0, "max": -50.0, "unit": "dBm"},
    "comm_status": {"min": 1, "nominal": 1, "max": 1, "unit": "binary"},
    "radiation_level": {"min": 0.01, "nominal": 0.05, "max": 0.30, "unit": "rad/s"}
}

# Anomaly Classes
ANOMALY_CLASSES = [
    "Normal",
    "Temperature spike",
    "Battery voltage drop",
    "Current surge",
    "Excessive power consumption",
    "Sensor value drift",
    "Communication signal drop",
    "Multiple simultaneous abnormalities"
]

# Severity definitions
SEVERITY_LEVELS = ["Normal", "Warning", "Critical"]

# Decision Support Rules (Rule-based recommendations)
# Explicitly labeled as operator advisory / decision-support guidance
DECISION_SUPPORT_RULES = {
    "Normal": {
        "summary": "Subsystem operating parameters within verified nominal thresholds.",
        "actions": [
            "Maintain standard orbital tracking and telemetry polling cadence (10s intervals).",
            "Routine background health log archival.",
            "No subsystem intervention required."
        ]
    },
    "Temperature spike": {
        "Warning": {
            "summary": "Elevated thermal reading detected in subsystem/payload module.",
            "actions": [
                "Verify redundant thermal sensor channel to rule out thermistor jitter.",
                "Review satellite attitude / sun-pointing orientation relative to orbital beta angle.",
                "Inspect heat pipe transport telemetry and passive radiator radiator efficiency.",
                "Prepare payload duty-cycle throttling if temperature exceeds 55°C."
            ]
        },
        "Critical": {
            "summary": "Thermal runaway risk: Subsystem temperature critically exceeds operational thermal envelope.",
            "actions": [
                "Recommend operator initiation of thermal contingency review.",
                "Shed non-essential secondary scientific payloads to reduce internal heat dissipation.",
                "Verify active louvers or thermal dissipation dissipation pathways.",
                "Assess reorienting satellite face toward deep space cold sink.",
                "Monitor CPU clock throttling to safeguard onboard computer (OBC) integrity."
            ]
        }
    },
    "Battery voltage drop": {
        "Warning": {
            "summary": "Battery bus voltage trending below nominal operating boundary during orbit.",
            "actions": [
                "Cross-check current orbital phase (eclipse entry vs. sunlit pass).",
                "Verify state of charge (SoC) estimation algorithms and shunt regulator status.",
                "Examine battery cell balancing telemetry across modular pack units.",
                "Log voltage degradation gradient for power management team review."
            ]
        },
        "Critical": {
            "summary": "Severe under-voltage detected: Electrical power subsystem (EPS) bus stability threatened.",
            "actions": [
                "Alert operations lead: Power bus collapse danger.",
                "Recommend switching EPS to priority power preservation profile.",
                "Isolate auxiliary loads; sustain only Command and Data Handling (C&DH) and primary transponder.",
                "Verify battery heater circuit is not stuck in continuous active draw.",
                "Prepare low-power safe-hold configuration procedures."
            ]
        }
    },
    "Current surge": {
        "Warning": {
            "summary": "Abnormal current spike registered on electrical distribution harness.",
            "actions": [
                "Examine subsystem power distribution unit (PDU) branch current monitors.",
                "Review recent command dispatch history for uncoordinated actuator/transmitter firing.",
                "Check for electrostatic discharge (ESD) signatures in telemetry logs.",
                "Increase sampling frequency on affected power line to 1 Hz."
            ]
        },
        "Critical": {
            "summary": "Critical overcurrent condition: Potential latch-up or short-circuit event.",
            "actions": [
                "Advise operator to inspect Solid-State Power Controller (SSPC) trip registers.",
                "Evaluate soft-reset or power-cycling of affected peripheral subsystem.",
                "Cross-reference radiation environment telemetry to determine Single Event Latch-up (SEL) probability.",
                "Prevent cascading EPS bus ripple by decoupling suspect payload branch."
            ]
        }
    },
    "Excessive power consumption": {
        "Warning": {
            "summary": "Total satellite power demand exceeds planned power budget.",
            "actions": [
                "Audit active subsystem operational modes against solar array generation forecast.",
                "Confirm transmitter RF output amplifier power level.",
                "Evaluate secondary instrument power consumption timeline.",
                "Schedule delayed downlink passes if battery depth-of-discharge (DoD) margin narrows."
            ]
        },
        "Critical": {
            "summary": "Severe power overload: Energy depletion threatens mission sustainability.",
            "actions": [
                "Trigger immediate power audit advisory for flight control team.",
                "Execute non-essential payload load-shed sequence.",
                "Confirm solar array maximum power point tracking (MPPT) converter status.",
                "Verify solar array drive assembly (SADA) alignment with solar vector.",
                "Ensure command uplink receiver remains powered on emergency bus."
            ]
        }
    },
    "Sensor value drift": {
        "Warning": {
            "summary": "Gradual uncharacteristic drift detected across sensor telemetry telemetry stream.",
            "actions": [
                "Conduct cross-sensor consistency validation against redundant sensor telemetry.",
                "Review historical baseline drift rate over last 10 orbital passes.",
                "Perform software-based sensor calibration offset recalculation.",
                "Flag telemetry channel as degraded in mission control displays."
            ]
        },
        "Critical": {
            "summary": "Severe telemetry divergence or telemetry sensor failure suspected.",
            "actions": [
                "Isolate faulty sensor channel from autonomous on-board control loops.",
                "Switch attitude/subsystem determination software to redundant sensor suite B.",
                "Validate telemetry frame checksums to exclude ground station decoding anomalies.",
                "Request engineering review of sensor aging and thermal hysteresis history."
            ]
        }
    },
    "Communication signal drop": {
        "Warning": {
            "summary": "Ground-to-space RF link margin degradation (RSSI falling below nominal).",
            "actions": [
                "Check ground antenna tracking azimuth and elevation pointing accuracy.",
                "Examine satellite attitude determination and control system (ADCS) pointing vector.",
                "Verify RF power amplifier output stage telemetry.",
                "Evaluate atmospheric weather attenuation along ground station line-of-sight."
            ]
        },
        "Critical": {
            "summary": "Critical communication blackout imminent or loss of primary carrier lock.",
            "actions": [
                "Advise ground station to switch to high-power uplink transmission mode.",
                "Prepare satellite autonomous recovery timer protocol for omnidirectional antenna failover.",
                "Cross-check secondary ground network station handoff schedule.",
                "Review Doppler shift correction parameters on SDR receiver."
            ]
        }
    },
    "Multiple simultaneous abnormalities": {
        "Warning": {
            "summary": "Concurrent parameter deviations observed across multiple subsystem boundaries.",
            "actions": [
                "Activate cross-subsystem correlation diagnostic protocol.",
                "Check space weather advisories for geomagnetic storm or solar energetic particle (SEP) influx.",
                "Verify common EPS and thermal bus coupling points.",
                "Elevate mission monitoring alert level to Tier-2 Watch."
            ]
        },
        "Critical": {
            "summary": "Compound multi-subsystem critical anomaly: High probability of cascading failure or severe space environment event.",
            "actions": [
                "Urgent advisory: Recommend flight director convene anomaly anomaly resolution board.",
                "Initiate spacecraft Safe-Hold mode preparation protocol.",
                "Prioritize telemetry beacon transmission over science downlink.",
                "Inspect radiation dosage telemetry for deep dielectric charging or heavy-ion upset.",
                "Isolate scientific instruments; preserve core satellite bus commandability."
            ]
        }
    }
}
