

const API_BASE = window.location.origin;

const SpaceGuardAPI = {
 
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

  
  async clearHistory() {
    try {
      const res = await fetch(`${API_BASE}/api/history/clear`, { method: 'POST' });
      return await res.json();
    } catch (err) {
      console.error('clearHistory error:', err);
      return { status: 'error' };
    }
  },

  
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


  async startSimulation() {
    try {
      const res = await fetch(`${API_BASE}/api/simulation/start`, { method: 'POST' });
      return await res.json();
    } catch (err) {
      console.error('startSimulation error:', err);
      return { status: 'error' };
    }
  },

  
  async stopSimulation() {
    try {
      const res = await fetch(`${API_BASE}/api/simulation/stop`, { method: 'POST' });
      return await res.json();
    } catch (err) {
      console.error('stopSimulation error:', err);
      return { status: 'error' };
    }
  },

  
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
