/**
 * SpaceGuard AI - Chart.js Visualization Helpers
 * Configures dark space-themed Chart.js telemetry graphs and comparative benchmarks.
 * NOTE: Academic simulation only - Not real NASA/spacecraft mission data.
 */

const SpaceGuardCharts = {
  // Chart Colors
  colors: {
    cyan: '#00f2fe',
    cyanDim: 'rgba(0, 242, 254, 0.15)',
    blue: '#38bdf8',
    emerald: '#10b981',
    emeraldDim: 'rgba(16, 185, 129, 0.15)',
    amber: '#f59e0b',
    amberDim: 'rgba(245, 158, 11, 0.15)',
    rose: '#ef4444',
    roseDim: 'rgba(239, 68, 68, 0.25)',
    purple: '#8b5cf6',
    grid: 'rgba(255, 255, 255, 0.06)',
    text: '#94a3b8'
  },

  /**
   * Returns common dark mode Chart.js options.
   */
  getDefaultOptions(title = '', showLegend = false) {
    return {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: showLegend,
          labels: { color: '#cbd5e1', font: { family: 'Inter', size: 12 } }
        },
        tooltip: {
          backgroundColor: 'rgba(9, 13, 26, 0.95)',
          titleColor: '#00f2fe',
          bodyColor: '#f1f5f9',
          borderColor: 'rgba(0, 242, 254, 0.3)',
          borderWidth: 1,
          padding: 10,
          cornerRadius: 6,
          titleFont: { family: 'Orbitron', size: 12 },
          bodyFont: { family: 'JetBrains Mono', size: 12 }
        }
      },
      scales: {
        x: {
          grid: { color: this.colors.grid },
          ticks: { color: this.colors.text, font: { family: 'JetBrains Mono', size: 10 } }
        },
        y: {
          grid: { color: this.colors.grid },
          ticks: { color: this.colors.text, font: { family: 'JetBrains Mono', size: 10 } }
        }
      }
    };
  },

  /**
   * Initializes a live telemetry time-series chart.
   */
  createTelemetryChart(canvasId, label, color, fillColor, unit = '') {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;

    const options = this.getDefaultOptions(label);
    options.scales.y.ticks.callback = (val) => `${val} ${unit}`;

    return new Chart(ctx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [{
          label: `${label} (${unit})`,
          data: [],
          borderColor: color,
          backgroundColor: fillColor,
          borderWidth: 2,
          fill: true,
          tension: 0.35,
          pointRadius: 2,
          pointHoverRadius: 6,
          pointHoverBackgroundColor: '#ffffff'
        }]
      },
      options: options
    });
  },

  /**
   * Updates an existing telemetry chart with fresh records.
   */
  updateTelemetryChart(chart, records, fieldName, maxPoints = 25) {
    if (!chart || !records || records.length === 0) return;

    const slice = records.slice(-maxPoints);
    const labels = slice.map(r => {
      const ts = r.timestamp || '';
      return ts.includes(' ') ? ts.split(' ')[1] : ts;
    });
    const values = slice.map(r => r[fieldName]);

    chart.data.labels = labels;
    chart.data.datasets[0].data = values;
    chart.update('none'); // silent instant update without heavy re-animation
  },

  /**
   * Initializes multi-model performance comparison bar chart.
   */
  createModelComparisonChart(canvasId, comparisonTable) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;

    const models = comparisonTable.map(m => m.model);
    const accuracy = comparisonTable.map(m => m.accuracy);
    const precision = comparisonTable.map(m => m.precision);
    const recall = comparisonTable.map(m => m.recall);
    const f1 = comparisonTable.map(m => m.f1_score);

    return new Chart(ctx, {
      type: 'bar',
      data: {
        labels: models,
        datasets: [
          {
            label: 'Accuracy (%)',
            data: accuracy,
            backgroundColor: 'rgba(0, 242, 254, 0.75)',
            borderColor: '#00f2fe',
            borderWidth: 1,
            borderRadius: 4
          },
          {
            label: 'Precision (%)',
            data: precision,
            backgroundColor: 'rgba(56, 189, 248, 0.75)',
            borderColor: '#38bdf8',
            borderWidth: 1,
            borderRadius: 4
          },
          {
            label: 'Recall (%)',
            data: recall,
            backgroundColor: 'rgba(139, 92, 246, 0.75)',
            borderColor: '#8b5cf6',
            borderWidth: 1,
            borderRadius: 4
          },
          {
            label: 'F1-Score (%)',
            data: f1,
            backgroundColor: 'rgba(16, 185, 129, 0.75)',
            borderColor: '#10b981',
            borderWidth: 1,
            borderRadius: 4
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: true,
            position: 'top',
            labels: { color: '#cbd5e1', font: { family: 'Inter', size: 12 } }
          },
          tooltip: {
            backgroundColor: 'rgba(9, 13, 26, 0.95)',
            borderColor: 'rgba(0, 242, 254, 0.3)',
            borderWidth: 1,
            titleFont: { family: 'Orbitron' },
            bodyFont: { family: 'JetBrains Mono' }
          }
        },
        scales: {
          x: {
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#ffffff', font: { family: 'Orbitron', size: 11, weight: 'bold' } }
          },
          y: {
            min: 90,
            max: 100,
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: {
              color: '#94a3b8',
              font: { family: 'JetBrains Mono', size: 10 },
              callback: (val) => `${val}%`
            }
          }
        }
      }
    });
  },

  /**
   * Initializes latency comparison horizontal bar chart.
   */
  createLatencyChart(canvasId, modelsData) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;

    const names = Object.keys(modelsData);
    const trainTimes = names.map(n => modelsData[n].train_time_sec);
    const predictLatencies = names.map(n => modelsData[n].predict_time_ms);

    return new Chart(ctx, {
      type: 'bar',
      data: {
        labels: names,
        datasets: [{
          label: 'Inference Latency (ms / sample)',
          data: predictLatencies,
          backgroundColor: [
            'rgba(0, 242, 254, 0.7)',
            'rgba(245, 158, 11, 0.7)',
            'rgba(16, 185, 129, 0.7)'
          ],
          borderColor: ['#00f2fe', '#f59e0b', '#10b981'],
          borderWidth: 1,
          borderRadius: 6
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => ` Latency: ${ctx.parsed.x.toFixed(4)} ms / sample`
            }
          }
        },
        scales: {
          x: {
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#94a3b8', font: { family: 'JetBrains Mono', size: 10 } }
          },
          y: {
            grid: { display: false },
            ticks: { color: '#ffffff', font: { family: 'Orbitron', size: 11, weight: 'bold' } }
          }
        }
      }
    });
  },

  /**
   * Renders an interactive confusion matrix heatmap table into a container.
   */
  renderConfusionMatrixHeatmap(containerId, cmMatrix, classLabels) {
    const container = document.getElementById(containerId);
    if (!container) return;

    // Find maximum count for color scaling
    let maxVal = 1;
    for (let r = 0; r < cmMatrix.length; r++) {
      for (let c = 0; c < cmMatrix[r].length; c++) {
        if (cmMatrix[r][c] > maxVal) maxVal = cmMatrix[r][c];
      }
    }

    let html = `<div class="table-responsive"><table class="cm-table"><thead><tr><th></th>`;
    classLabels.forEach(lbl => {
      const short = lbl.length > 9 ? lbl.substring(0, 8) + '…' : lbl;
      html += `<th title="${lbl}" style="color: #94a3b8; font-size: 0.68rem; padding: 4px; text-align: center;">${short}</th>`;
    });
    html += `</tr></thead><tbody>`;

    cmMatrix.forEach((row, i) => {
      const rowLabel = classLabels[i] || `Class ${i}`;
      const shortRow = rowLabel.length > 9 ? rowLabel.substring(0, 8) + '…' : rowLabel;
      html += `<tr><td title="${rowLabel}" style="color: #94a3b8; font-size: 0.68rem; padding: 4px; text-align: right; font-weight: 600;">${shortRow}</td>`;
      row.forEach((val, j) => {
        const isDiagonal = (i === j);
        const intensity = Math.min(1, val / maxVal);
        let bg = 'rgba(255, 255, 255, 0.03)';
        let color = '#64748b';

        if (val > 0) {
          if (isDiagonal) {
            bg = `rgba(0, 242, 254, ${Math.max(0.2, intensity * 0.85)})`;
            color = intensity > 0.4 ? '#050811' : '#00f2fe';
          } else {
            bg = `rgba(239, 68, 68, ${Math.max(0.25, intensity * 0.85)})`;
            color = '#ffffff';
          }
        }

        html += `<td class="cm-cell" style="background: ${bg}; color: ${color};" title="Actual: ${rowLabel} | Predicted: ${classLabels[j]} | Count: ${val}">${val}</td>`;
      });
      html += `</tr>`;
    });
    html += `</tbody></table></div>`;

    container.innerHTML = html;
  }
};
