/**
 * SpaceGuard AI - API Client Library
 * Handles asynchronous communication with the Flask REST Backend.
 * NOTE: Academic simulation only - Not real NASA/spacecraft mission data.
 */

const API_BASE = window.location.origin.includes('5000') 
  ? window.location.origin 
  : 'http://127.0.0.1:5000';

const SpaceGuardAPI = {
  /**
   * Fetches real-time satellite health status and current telemetry snapshot.
   */
  async getHealth() {
    try {
      const res = await fetch(`${API_BASE}/api/health`);
      if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
      return await res.json();
    } catch (err) {
      console.error('getHealth error:', err);
      return null;
    }
  },

  /**
   * Fetches high-level dashboard KPIs, anomaly rate, and subsystem breakdown.
   */
  async getDashboardStats() {
    try {
      const res = await fetch(`${API_BASE}/api/dashboard/stats`);
      if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
      return await res.json();
    } catch (err) {
      console.error('getDashboardStats error:', err);
      return null;
    }
  },

  /**
   * Fetches historical telemetry stream observations.
   */
  async getTelemetry(params = {}) {
    const query = new URLSearchParams(params).toString();
    try {
      const res = await fetch(`${API_BASE}/api/telemetry?${query}`);
      if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
      return await res.json();
    } catch (err) {
      console.error('getTelemetry error:', err);
      return { status: 'error', records: [] };
    }
  },

  /**
   * Runs anomaly inference on single telemetry record.
   */
  async predictSingle(telemetry, modelName = 'XGBoost', logToHistory = true) {
    try {
      const res = await fetch(`${API_BASE}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          telemetry,
          model_name: modelName,
          log_to_history: logToHistory
        })
      });
      return await res.json();
    } catch (err) {
      console.error('predictSingle error:', err);
      return { status: 'error', message: err.message };
    }
  },

  /**
   * Runs batch prediction on uploaded CSV file.
   */
  async predictBatch(file, modelName = 'XGBoost') {
    const formData = new FormData();
    formData.append('file', file);
    try {
      const res = await fetch(`${API_BASE}/api/predict?model=${encodeURIComponent(modelName)}`, {
        method: 'POST',
        body: formData
      });
      return await res.json();
    } catch (err) {
      console.error('predictBatch error:', err);
      return { status: 'error', message: err.message };
    }
  },

  /**
   * Fetches ML benchmark metrics, confusion matrices, and feature importances.
   */
  async getModelPerformance() {
    try {
      const res = await fetch(`${API_BASE}/api/models/performance`);
      if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
      return await res.json();
    } catch (err) {
      console.error('getModelPerformance error:', err);
      return null;
    }
  },

  /**
   * Fetches logged anomaly history from SQLite database.
   */
  async getHistory(params = {}) {
    const query = new URLSearchParams(params).toString();
    try {
      const res = await fetch(`${API_BASE}/api/history?${query}`);
      if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
      return await res.json();
    } catch (err) {
      console.error('getHistory error:', err);
      return { status: 'error', data: { records: [], total: 0 } };
    }
  },

  /**
   * Clears anomaly history log in database.
   */
  async clearHistory() {
    try {
      const res = await fetch(`${API_BASE}/api/history/clear`, { method: 'POST' });
      return await res.json();
    } catch (err) {
      console.error('clearHistory error:', err);
      return { status: 'error' };
    }
  },

  /**
   * Uploads raw CSV telemetry to ingest into stream.
   */
  async uploadDataset(file) {
    const formData = new FormData();
    formData.append('file', file);
    try {
      const res = await fetch(`${API_BASE}/api/dataset/upload`, {
        method: 'POST',
        body: formData
      });
      return await res.json();
    } catch (err) {
      console.error('uploadDataset error:', err);
      return { status: 'error', message: err.message };
    }
  },

  /**
   * Starts real-time simulation background loop.
   */
  async startSimulation() {
    try {
      const res = await fetch(`${API_BASE}/api/simulation/start`, { method: 'POST' });
      return await res.json();
    } catch (err) {
      console.error('startSimulation error:', err);
      return { status: 'error' };
    }
  },

  /**
   * Stops real-time simulation background loop.
   */
  async stopSimulation() {
    try {
      const res = await fetch(`${API_BASE}/api/simulation/stop`, { method: 'POST' });
      return await res.json();
    } catch (err) {
      console.error('stopSimulation error:', err);
      return { status: 'error' };
    }
  },

  /**
   * Injects an anomaly mode into the live simulation tick.
   */
  async injectAnomaly(anomalyType) {
    try {
      const res = await fetch(`${API_BASE}/api/simulation/inject_anomaly`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ anomaly_type: anomalyType })
      });
      return await res.json();
    } catch (err) {
      console.error('injectAnomaly error:', err);
      return { status: 'error' };
    }
  },

  /**
   * Sets active ML model for live simulation inferences.
   */
  async setActiveModel(modelName) {
    try {
      const res = await fetch(`${API_BASE}/api/models/set_active`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_name: modelName })
      });
      return await res.json();
    } catch (err) {
      console.error('setActiveModel error:', err);
      return { status: 'error' };
    }
  }
};
