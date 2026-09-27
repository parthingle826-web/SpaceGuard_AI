# SpaceGuard AI – Windows Setup & Execution Guide

Follow these step-by-step instructions to run the **SpaceGuard AI** application locally on Windows.

> **ACADEMIC SIMULATION DISCLAIMER:**  
> SpaceGuard AI is an educational simulation system. All telemetry data is synthetic and **NOT real NASA or operational spacecraft mission data**.

---

## Prerequisites
- **Operating System:** Windows 10 or Windows 11
- **Python:** Python 3.10, 3.11, 3.12, 3.13, or 3.14 (Ensure "Add Python to PATH" was checked during installation)
- **Web Browser:** Google Chrome, Microsoft Edge, or Mozilla Firefox

---

## Step 1: Open PowerShell or Command Prompt
Open PowerShell or Command Prompt in the project root directory:
```powershell
cd "d:\AIML_Mini_Project ( SpaceGuard AI)"
```

---

## Step 2: Create and Activate a Virtual Environment (Recommended)
Create a Python virtual environment:
```powershell
python -m venv venv
```

Activate the virtual environment:
```powershell
.\venv\Scripts\Activate.ps1
```
*(If on Command Prompt `cmd.exe`, run: `venv\Scripts\activate.bat`)*

*(If PowerShell displays an Execution Policy restriction, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` first).*

---

## Step 3: Install Required Dependencies
Install the verified dependencies from `backend/requirements.txt`:
```powershell
pip install -r backend/requirements.txt
```

Verify that all packages are installed:
```powershell
python -c "import flask, pandas, sklearn, xgboost, joblib; print('All packages installed successfully!')"
```

---

## Step 4: Generate Dataset & Train Machine Learning Models
Generate the 7,200-row synthetic telemetry dataset:
```powershell
python scripts/generate_dataset.py
```

Run the preprocessing pipeline:
```powershell
python backend/ml/preprocess.py
```

Train all three models (Decision Tree, SVM, XGBoost):
```powershell
python backend/ml/train_models.py
```

Evaluate all models and produce `metrics.json` and the comparison table:
```powershell
python backend/ml/evaluate.py
```

---

## Step 5: Start the Flask Backend Server
Launch the Flask REST backend and real-time simulation engine:
```powershell
python backend/app.py
```

You should see output similar to:
```
======================================================================
SpaceGuard AI - Academic Satellite Telemetry Anomaly Detection
Mission Control Dashboard running at: http://127.0.0.1:5000
DISCLAIMER: Academic simulation only - Not real NASA/mission data.
======================================================================
 * Running on http://127.0.0.1:5000
```

---

## Step 6: Access the Application in the Browser
Open your browser and navigate to:
```
http://127.0.0.1:5000/
```

### Application Page Directory:
| Page | URL | Purpose |
| :--- | :--- | :--- |
| **Mission Overview** | `http://127.0.0.1:5000/` | System landing page, architecture overview, quick stats |
| **Mission Control** | `http://127.0.0.1:5000/dashboard` | Live telemetry graphs, health gauge, simulation controls, failure injection |
| **Telemetry Explorer** | `http://127.0.0.1:5000/telemetry` | Time-series channel inspection, CSV dataset ingestion, table view |
| **Anomaly Lab** | `http://127.0.0.1:5000/anomaly` | Interactive parameter vector test form, scenario presets, decision support |
| **ML Benchmarks** | `http://127.0.0.1:5000/models` | 3-Model comparison, accuracy charts, confusion matrices, latency |
| **Incident History** | `http://127.0.0.1:5000/history` | Filterable SQLite anomaly audit log, detailed inspection modal, CSV export |
| **Architecture & Docs** | `http://127.0.0.1:5000/about` | Academic methodology, parameter specs, SVG diagrams, IEEE references |

---

## Troubleshooting on Windows
- **Port 5000 already in use?**
  Set a custom port before running:
  ```powershell
  $env:PORT="5050"; python backend/app.py
  ```
  Then access at `http://127.0.0.1:5050/`.
- **Missing packages?**
  Run `pip install flask flask-cors pandas numpy scikit-learn xgboost joblib python-dotenv`.
