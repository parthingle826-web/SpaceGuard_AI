"""
SpaceGuard AI - Dataset Generator
Generates a realistic, physics-grounded synthetic satellite telemetry dataset for LEO orbit.

DISCLAIMER:
THIS IS AN ACADEMIC SIMULATION DATASET.
IT IS NOT REAL NASA OR OPERATIONAL MISSION DATA.
All telemetry values and anomaly injections are synthetically generated for academic research
and machine learning performance evaluation.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# Ensure parent directory is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from backend.config import RAW_DATASET_PATH, DATA_DIR, TELEMETRY_FEATURES

def generate_satellite_telemetry(num_records: int = 7200, random_seed: int = 42) -> pd.DataFrame:
    """
    Generates time-series satellite telemetry spanning multiple LEO orbits (~90 min per orbit).
    Includes normal orbital cycles (eclipse vs. sunlit passes) and injected anomalies
    representing 7 distinct spacecraft subsystem failure modes.
    """
    np.random.seed(random_seed)
    
    # 7200 records at 10-second intervals = 20 hours (approx 13.3 orbits)
    start_time = datetime(2026, 3, 15, 0, 0, 0)
    timestamps = [start_time + timedelta(seconds=i * 10) for i in range(num_records)]
    
    # Orbital period in seconds (~5400 seconds = 90 mins)
    orbital_period = 5400
    t_seconds = np.array([i * 10 for i in range(num_records)])
    orbit_phase = 2 * np.pi * (t_seconds % orbital_period) / orbital_period
    
    # Solar illumination factor: in sunlight (~60% of orbit), eclipse (~40% of orbit)
    # Sunlight when sin(orbit_phase) > -0.2 with smooth transition
    sunlight_raw = np.sin(orbit_phase)
    is_sunlit = 1 / (1 + np.exp(-10 * (sunlight_raw + 0.15)))  # Sigmoid transition
    
    # Base Nominal Features with Physics Correlations
    # 1. Solar panel voltage: ~48V in sunlight, ~0V in eclipse
    solar_v_nominal = 48.0 * is_sunlit + np.random.normal(0, 1.2, num_records)
    solar_v_nominal = np.clip(solar_v_nominal, 0.0, 56.0)
    
    # 2. Solar panel current: ~6.5A in sunlight, ~0A in eclipse
    solar_i_nominal = 6.5 * is_sunlit + np.random.normal(0, 0.35, num_records)
    solar_i_nominal = np.clip(solar_i_nominal, 0.0, 9.8)
    
    # 3. Battery voltage: charges to ~29.5V in sunlight, discharges to ~25.5V in eclipse
    battery_v_nominal = 27.5 + 2.0 * np.sin(orbit_phase) + np.random.normal(0, 0.3, num_records)
    battery_v_nominal = np.clip(battery_v_nominal, 24.5, 30.5)
    
    # 4. Battery current: net charge (+ current into battery) in sunlight, discharge in eclipse
    battery_i_nominal = 3.5 + 1.8 * (1 - is_sunlit) + np.random.normal(0, 0.25, num_records)
    battery_i_nominal = np.clip(battery_i_nominal, 1.2, 7.5)
    
    # 5. Base thermal: colder in eclipse (~12°C), warmer in sunlight (~28°C)
    temperature_nominal = 20.0 + 8.0 * np.sin(orbit_phase - np.pi / 4) + np.random.normal(0, 1.5, num_records)
    temperature_nominal = np.clip(temperature_nominal, 5.0, 38.0)
    
    # 6. CPU / Subsystem temperature: runs slightly warmer than ambient structure
    cpu_temp_nominal = temperature_nominal + 14.0 + np.random.normal(0, 1.2, num_records)
    cpu_temp_nominal = np.clip(cpu_temp_nominal, 22.0, 52.0)
    
    # 7. Pressurized avionics compartment: stable ~101.3 kPa with minor barometric fluctuations
    pressure_nominal = 101.3 + np.random.normal(0, 0.4, num_records)
    pressure_nominal = np.clip(pressure_nominal, 99.5, 103.0)
    
    # 8. Base Power consumption: 180W nominal, slightly higher when transmitters active in sunlight
    power_nominal = 190.0 + 35.0 * is_sunlit + np.random.normal(0, 8.0, num_records)
    power_nominal = np.clip(power_nominal, 140.0, 260.0)
    
    # 9. Signal strength: nominal ground contact passes (-75 dBm with elevation modulation)
    signal_strength_nominal = -75.0 + 10.0 * np.sin(2 * np.pi * t_seconds / 2700) + np.random.normal(0, 3.0, num_records)
    signal_strength_nominal = np.clip(signal_strength_nominal, -92.0, -58.0)
    
    # 10. Comm status: nominal 1 (connected)
    comm_status_nominal = np.ones(num_records, dtype=int)
    
    # 11. Background space radiation: low nominal baseline in LEO
    radiation_nominal = 0.04 + np.random.exponential(0.015, num_records)
    radiation_nominal = np.clip(radiation_nominal, 0.01, 0.12)
    
    # Construct base DataFrame
    df = pd.DataFrame({
        "timestamp": [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in timestamps],
        "battery_voltage": np.round(battery_v_nominal, 3),
        "battery_current": np.round(battery_i_nominal, 3),
        "temperature": np.round(temperature_nominal, 2),
        "pressure": np.round(pressure_nominal, 2),
        "solar_panel_voltage": np.round(solar_v_nominal, 2),
        "solar_panel_current": np.round(solar_i_nominal, 3),
        "power_consumption": np.round(power_nominal, 2),
        "cpu_subsystem_temp": np.round(cpu_temp_nominal, 2),
        "signal_strength": np.round(signal_strength_nominal, 2),
        "comm_status": comm_status_nominal,
        "radiation_level": np.round(radiation_nominal, 4),
        "is_anomaly": 0,
        "anomaly_type": "Normal",
        "severity": "Normal"
    })
    
    # =========================================================================
    # INJECT REALISTIC ANOMALIES
    # =========================================================================
    # Target ~18% total anomaly rate across 7 distinct categories with realistic durations
    # (spans of 15 to 45 consecutive records = 2.5 to 7.5 minutes each)
    
    anomaly_definitions = [
        # (type, severity, duration, count)
        ("Temperature spike", "Warning", 24, 7),
        ("Temperature spike", "Critical", 28, 5),
        ("Battery voltage drop", "Warning", 25, 7),
        ("Battery voltage drop", "Critical", 30, 5),
        ("Current surge", "Warning", 15, 8),
        ("Current surge", "Critical", 20, 6),
        ("Excessive power consumption", "Warning", 26, 7),
        ("Excessive power consumption", "Critical", 30, 5),
        ("Sensor value drift", "Warning", 35, 6),
        ("Sensor value drift", "Critical", 40, 5),
        ("Communication signal drop", "Warning", 20, 7),
        ("Communication signal drop", "Critical", 25, 5),
        ("Multiple simultaneous abnormalities", "Warning", 28, 6),
        ("Multiple simultaneous abnormalities", "Critical", 35, 5),
    ]
    
    occupied_indices = set()
    
    for anom_type, severity, duration, count in anomaly_definitions:
        for _ in range(count):
            # Find an unoccupied window
            for attempt in range(150):
                start_idx = np.random.randint(150, num_records - duration - 100)
                window_indices = set(range(start_idx, start_idx + duration))
                if not window_indices.intersection(occupied_indices):
                    occupied_indices.update(window_indices)
                    break
            else:
                continue
                
            indices = list(range(start_idx, start_idx + duration))
            df.loc[indices, "is_anomaly"] = 1
            df.loc[indices, "anomaly_type"] = anom_type
            df.loc[indices, "severity"] = severity
            
            # Injection logic tailored by anomaly type and severity
            if anom_type == "Temperature spike":
                if severity == "Warning":
                    df.loc[indices, "temperature"] += np.random.uniform(18.0, 28.0, duration)
                    df.loc[indices, "cpu_subsystem_temp"] += np.random.uniform(14.0, 22.0, duration)
                else:  # Critical
                    df.loc[indices, "temperature"] += np.random.uniform(34.0, 48.0, duration)
                    df.loc[indices, "cpu_subsystem_temp"] += np.random.uniform(28.0, 42.0, duration)
                    df.loc[indices, "power_consumption"] += np.random.uniform(25.0, 60.0, duration)
                    
            elif anom_type == "Battery voltage drop":
                if severity == "Warning":
                    df.loc[indices, "battery_voltage"] -= np.random.uniform(3.5, 5.0, duration)
                    df.loc[indices, "battery_current"] += np.random.uniform(1.5, 2.8, duration)
                else:  # Critical
                    df.loc[indices, "battery_voltage"] -= np.random.uniform(6.5, 9.5, duration)
                    df.loc[indices, "battery_current"] += np.random.uniform(3.5, 5.5, duration)
                    df.loc[indices, "power_consumption"] += np.random.uniform(40.0, 80.0, duration)
                    
            elif anom_type == "Current surge":
                if severity == "Warning":
                    df.loc[indices, "battery_current"] += np.random.uniform(3.2, 5.0, duration)
                    df.loc[indices, "power_consumption"] += np.random.uniform(50.0, 90.0, duration)
                else:  # Critical
                    df.loc[indices, "battery_current"] += np.random.uniform(6.5, 10.5, duration)
                    df.loc[indices, "power_consumption"] += np.random.uniform(110.0, 190.0, duration)
                    df.loc[indices, "cpu_subsystem_temp"] += np.random.uniform(8.0, 16.0, duration)
                    
            elif anom_type == "Excessive power consumption":
                if severity == "Warning":
                    df.loc[indices, "power_consumption"] += np.random.uniform(80.0, 130.0, duration)
                    df.loc[indices, "battery_voltage"] -= np.random.uniform(1.8, 3.2, duration)
                else:  # Critical
                    df.loc[indices, "power_consumption"] += np.random.uniform(160.0, 260.0, duration)
                    df.loc[indices, "battery_voltage"] -= np.random.uniform(3.8, 6.5, duration)
                    df.loc[indices, "temperature"] += np.random.uniform(10.0, 20.0, duration)
                    
            elif anom_type == "Sensor value drift":
                # Linear ramp drift simulating aging / uncalibrated transducer
                drift_ramp = np.linspace(0, 1, duration)
                if severity == "Warning":
                    df.loc[indices, "pressure"] -= drift_ramp * np.random.uniform(12.0, 18.0)
                else:  # Critical
                    df.loc[indices, "pressure"] -= drift_ramp * np.random.uniform(25.0, 38.0)
                    df.loc[indices, "temperature"] += drift_ramp * np.random.uniform(15.0, 25.0)
                    
            elif anom_type == "Communication signal drop":
                if severity == "Warning":
                    df.loc[indices, "signal_strength"] -= np.random.uniform(18.0, 28.0, duration)
                else:  # Critical
                    df.loc[indices, "signal_strength"] -= np.random.uniform(32.0, 50.0, duration)
                    # Dropped carrier lock: comm_status drops to 0
                    df.loc[indices, "comm_status"] = 0
                    
            elif anom_type == "Multiple simultaneous abnormalities":
                # Space weather / solar particle event causing multiple subsystem stress
                if severity == "Warning":
                    df.loc[indices, "radiation_level"] += np.random.uniform(0.4, 0.9, duration)
                    df.loc[indices, "temperature"] += np.random.uniform(12.0, 20.0, duration)
                    df.loc[indices, "signal_strength"] -= np.random.uniform(15.0, 25.0, duration)
                else:  # Critical
                    df.loc[indices, "radiation_level"] += np.random.uniform(1.2, 2.8, duration)
                    df.loc[indices, "temperature"] += np.random.uniform(25.0, 42.0, duration)
                    df.loc[indices, "cpu_subsystem_temp"] += np.random.uniform(22.0, 36.0, duration)
                    df.loc[indices, "battery_voltage"] -= np.random.uniform(4.0, 7.5, duration)
                    df.loc[indices, "signal_strength"] -= np.random.uniform(30.0, 48.0, duration)
                    df.loc[indices, "comm_status"] = np.random.choice([0, 1], size=duration, p=[0.75, 0.25])

    # Round all numerical columns appropriately
    for col in TELEMETRY_FEATURES:
        if col == "comm_status":
            df[col] = df[col].astype(int)
        elif col in ["radiation_level"]:
            df[col] = df[col].round(4)
        elif col in ["battery_voltage", "battery_current", "solar_panel_current"]:
            df[col] = df[col].round(3)
        else:
            df[col] = df[col].round(2)

    return df

if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating synthetic satellite telemetry dataset...")
    df = generate_satellite_telemetry(num_records=7200, random_seed=42)
    
    # Save raw synthetic dataset
    df.to_csv(RAW_DATASET_PATH, index=False)
    print(f"Dataset generated successfully at: {RAW_DATASET_PATH}")
    print(f"Total rows: {len(df)}")
    print(f"Normal records: {(df['is_anomaly'] == 0).sum()} ({(df['is_anomaly'] == 0).mean():.1%})")
    print(f"Anomaly records: {(df['is_anomaly'] == 1).sum()} ({(df['is_anomaly'] == 1).mean():.1%})")
    print("\nAnomaly Breakdown by Category:")
    print(df["anomaly_type"].value_counts())
    print("\nSeverity Breakdown:")
    print(df["severity"].value_counts())
