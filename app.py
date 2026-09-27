"""
SpaceGuard AI - Root Application Entrypoint
Exposes the Flask 'app' instance for Vercel Serverless Functions and local execution.
NOTE: Academic simulation only - Not real NASA/spacecraft mission data.
"""

import os
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from backend.app import app, initialize_application
from backend.config import HOST, PORT, DEBUG, IS_VERCEL



if IS_VERCEL:
    try:
        initialize_application()
    except Exception as e:
        import logging
        logging.getLogger("SpaceGuardAI").warning(f"Vercel warm-up initialization warning: {e}")

if __name__ == "__main__":

    initialize_application()
    print("=" * 70)
    print("SpaceGuard AI - Academic Satellite Telemetry Anomaly Detection")
    print(f"Mission Control Dashboard running at: http://{HOST}:{PORT}")
    print("DISCLAIMER: Academic simulation only - Not real NASA/mission data.")
    print("=" * 70)
    app.run(host=HOST, port=PORT, debug=DEBUG, use_reloader=False)
