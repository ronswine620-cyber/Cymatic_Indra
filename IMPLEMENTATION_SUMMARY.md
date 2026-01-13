# 🎉 Cymatic Indra - Hybrid Data Export System Complete!

## What Was Implemented

You now have a **complete hybrid JSON + CSV data export system** for the Cymatic Indra simulation engine.

---

## 📦 New Files Created

### 1. **`simulation_runner.py`** (9.8 KB)
Enhanced simulation runner with data export capabilities.

**Features:**
- Runs complete simulations with automatic data export
- Exports metadata to JSON
- Exports time-series to CSV
- Generates text summary reports
- Command-line interface to list completed runs

**Usage:**
```bash
python3 simulation_runner.py
```

**Output Structure:**
```
results/
└── run_20260113_124804/
    ├── simulation.json      # Metadata & statistics
    ├── timeseries.csv       # Time-series data (300 steps)
    ├── config.json          # Engine configuration
    └── analysis_report.txt  # Text summary
```

---

### 2. **`analyze_results.ipynb`** (Jupyter Notebook)
Interactive analysis notebook for exploring simulation data.

**Features:**
- Load JSON metadata and CSV time-series
- Statistical summaries (max, avg, min coherence)
- Time evolution plots
- Phase dynamics visualization
- Manifold geometry analysis
- Crystallization event tracking
- Multi-run comparison tools
- Export analysis reports

**Usage:**
```bash
pip install jupyter
jupyter notebook analyze_results.ipynb
```

---

### 3. **Updated `README.md`**
Comprehensive documentation covering:
- Quick start guide (3 options)
- Data export explanation
- Jupyter notebook walkthrough
- Configuration guide
- Project structure
- Performance metrics
- Mathematical foundations

---

## 📊 Data Export Format

### JSON (`simulation.json`)
Complete reproducibility metadata:
```json
{
  "metadata": {
    "timestamp": "2026-01-13T12:48:04",
    "n_agents": 20,
    "dimensions": 12,
    "num_steps": 300,
    "version": "1.0"
  },
  "statistics": {
    "max_coherence": 0.615,
    "avg_coherence": 0.337,
    "min_coherence": 0.081,
    "crystallization_events": 0,
    "coherence_history": [0.207, 0.292, ...]
  }
}
```

### CSV (`timeseries.csv`)
Time-series data for analysis:
```
step,coherence,crystallization_events,phase_0,manifold_energy,max_phase,min_phase
0,0.207,0,1.234,2.456,2.891,0.123
1,0.292,0,0.891,2.123,2.654,0.456
...
```

---

## 🔄 Workflow

### Step 1: Run Simulation
```bash
python3 simulation_runner.py
```
✓ Creates `results/run_*/` directory
✓ Exports JSON, CSV, and config files

### Step 2: Analyze Results
```bash
jupyter notebook analyze_results.ipynb
```
✓ Load the latest run
✓ Generate visualizations
✓ Compare multiple runs
✓ Export analysis report

### Step 3: Export Data
```python
import pandas as pd
df = pd.read_csv('results/run_*/timeseries.csv')
# Now use with matplotlib, seaborn, scikit-learn, etc.
```

---

## 🎯 Key Features

### ✅ Reproducibility
- Full metadata stored in JSON
- Configuration exported with each run
- Timestamps for tracking
- Complete parameter capture

### ✅ Analysis-Ready
- CSV format works with Excel, Python, R, MATLAB
- Time-series indexed by step number
- Normalized coherence values (0-1)
- Multiple time-series metrics per step

### ✅ Scalable
- Run multiple experiments in a loop
- Automatically organize results
- Compare runs side-by-side
- Track experiment history

### ✅ Jupyter Integration
- Pre-built analysis notebook
- Publication-quality plots
- Statistical summaries
- Multi-run comparison tools

---

## 📁 Project Structure

```
Cymatic_Indra/
├── cymatic_indra/
│   ├── oscillator.py       # The Jewels
│   ├── manifold.py         # The Plate
│   ├── core.py             # The Spider
│   └── requirements.txt
├── visualize_indra_v2.py    # Interactive visualization
├── simulation_runner.py      # ⭐ Enhanced runner with export
├── analyze_results.ipynb     # ⭐ Jupyter analysis notebook
├── results/                  # ⭐ Auto-created output directory
│   └── run_YYYYMMDD_HHMMSS/
│       ├── simulation.json
│       ├── timeseries.csv
│       ├── config.json
│       └── analysis_report.txt
├── .gitignore               # Updated to ignore results/
└── README.md                # Complete documentation
```

---

## 🚀 Quick Start Commands

### Run a single simulation with data export:
```bash
cd /Users/randy/Documents/GitHub/projects/Cymatic_Indra
python3 simulation_runner.py
```

### Run multiple experiments:
```python
# Edit simulation_runner.py
runner = SimulationRunner(n_agents=20, dimensions=12, num_steps=300)
for i in range(5):
    runner.run_simulation(run_name=f"experiment_{i+1}")
```

### Analyze in Jupyter:
```bash
jupyter notebook analyze_results.ipynb
```

### List all completed runs:
```bash
python3 -c "from simulation_runner import SimulationRunner; SimulationRunner().list_runs()"
```

---

## 📊 Analysis Capabilities

The Jupyter notebook provides:

1. **Statistical Summary**
   - Max/Min/Avg coherence
   - Standard deviation
   - Crystallization event count

2. **Visualizations**
   - Coherence evolution plot
   - Phase dynamics of individual oscillators
   - Manifold energy over time
   - Phase space spread
   - Coherence distribution histogram

3. **Multi-Run Comparison**
   - Compare max coherence across runs
   - Compare average coherence
   - Show coherence ranges
   - Track crystallization events

4. **Data Export**
   - Generate text summary reports
   - Export plots as PNG/SVG
   - Aggregate statistics from multiple runs

---

## 🔧 Configuration

### Simulation Parameters
Edit in `simulation_runner.py`:
```python
runner = SimulationRunner(
    n_agents=20,      # Number of oscillators
    dimensions=12,    # Dimensionality
    num_steps=300     # Simulation steps
)
```

### Engine Configuration
Edit in `simulation_runner.py`:
```python
self.config = {
    'window_size': 32,
    'damping_factor': 0.6,
    'phase_wrap_thresh': 2.0,
    'coupling_strength': 0.1,
    'plasticity_rate': 0.05
}
```

---

## ✨ Why This Approach?

### ✅ JSON Benefits
- Human-readable
- Self-documenting
- Stores complete metadata
- Language-independent
- Version control friendly
- No size limitations

### ✅ CSV Benefits
- Widely supported (Excel, R, Python, etc.)
- Optimized for time-series data
- Easy statistical analysis
- Lightweight and portable
- Compatible with all plotting libraries

### ✅ Combined Power
- JSON for experiment tracking
- CSV for data analysis
- Both for reproducibility
- Flexible workflow
- Future-proof

---

## 📈 Next Steps

### Suggested Enhancements

1. **Run Parameter Sweeps**
   ```python
   for n_agents in [10, 20, 30]:
       for dims in [8, 12, 16]:
           runner = SimulationRunner(n_agents=n_agents, dimensions=dims)
           runner.run_simulation()
   ```

2. **Statistical Analysis**
   - Compare coherence across parameter ranges
   - Identify optimal configurations
   - Analyze phase transition behavior

3. **Visualization Gallery**
   - Create publication-ready plots
   - Generate comparison figures
   - Export to PDF reports

4. **Machine Learning**
   - Use CSV data to train models
   - Predict coherence evolution
   - Classify system states

---

## 🎓 Summary

You now have:

| Component | Status | Purpose |
|-----------|--------|---------|
| Core Engine | ✅ Working | Runs simulations |
| Visualization | ✅ Working | Real-time display |
| Data Export | ✅ **NEW** | JSON + CSV output |
| Analysis Notebook | ✅ **NEW** | Jupyter analysis |
| Documentation | ✅ **UPDATED** | Complete README |
| Version Control | ✅ **UPDATED** | .gitignore ready |

**Everything is ready to use! 🚀**

---

## 📝 Generated

January 13, 2026 - Cymatic Indra Hybrid Data Export System v1.0
