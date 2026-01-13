# 🕸️ Cymatic Indra
**Self-Organizing Topological Intelligence & Emergent Coherence**

**Cymatic Indra** is a biomimetic simulation engine designed to explore **morphogenesis**, **synchronization**, and **ontological resilience**. Inspired by the Buddhist metaphor of *Indra's Net* and the physics of cymatics, the system treats data not as discrete tokens, but as vibratory frequencies that "crystallize" into topological invariants on a reactive manifold.

---

## 🌌 The Metaphor

In the celestial net of the great god Indra, a jewel is set at every intersection. Each jewel reflects all others, and in that reflection, it contains the reflections of the entire universe.

In this system:
* **The Jewels (`IndraOscillator`):** Autonomous agents sensing and vibrating in $N$-dimensional space.
* **The Plate (`CymaticManifold`):** The elastic substrate where patterns are etched and memories are stored.
* **The Spider (`IndraEngine`):** The coordinator that filters noise, ensures synchronization, and "gardens" the network topology.

---

## 🏗️ System Architecture

### 1. The Jewels (`cymatic_indra.oscillator`)
Autonomous phase-oscillators that generate complex trajectories. They do not just "speak"; they *resonate* with the current state of the environment (the Manifold).

### 2. The Plate (`cymatic_indra.manifold`)
A reactive topological memory. It acts as a **"Read-Write" substrate** that:
* **Imprints:** Etches persistent structures when coherence is achieved ($Z > 0.8$).
* **Elasticity:** Naturally decays over time, requiring constant reinforcement.
* **Topology:** Uses **Betti-0 Analysis** to detect fractures in the system's unity.

### 3. The Spider (`cymatic_indra.core`)
The "Executive" layer that handles:
* **Spectral Sieve:** Uses FFT to extract frequency invariants from raw signals.
* **Phase-Locking:** Adaptive Kuramoto-style coupling to drive toward a "Crystallized" state.
* **Broken Functor Filtering:** Adversarial defense filtering large phase discontinuities.

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/cymatic-indra.git
cd cymatic-indra

# Install dependencies
pip install -r cymatic_indra/requirements.txt
```

### Running Simulations

#### Option 1: Interactive Visualization (macOS/Linux with Display)

```bash
python3 visualize_indra_v2.py
```

Opens a live matplotlib window showing:
- **Left plot:** The Jewels (oscillator phases in phase space)
- **Right plot:** The Plate (manifold topological structure)

#### Option 2: Enhanced Runner with Data Export ⭐ **RECOMMENDED**

```bash
python3 simulation_runner.py
```

Automatically exports results to `results/run_YYYYMMDD_HHMMSS/`:
- `simulation.json` - Complete metadata and statistics
- `timeseries.csv` - Time-series data for analysis
- `config.json` - Engine configuration used
- `analysis_report.txt` - Text summary

#### Option 3: Analyze Results with Jupyter

```bash
pip install jupyter
jupyter notebook analyze_results.ipynb
```

Features:
- Statistical summaries and plots
- Coherence evolution analysis
- Phase dynamics visualization
- Manifold geometry analysis
- Multi-run comparison tools

---

## 📊 Data Export & Analysis (Hybrid Approach)

### JSON Format (`simulation.json`)
Complete metadata and statistics for experiment reproducibility:

```json
{
  "metadata": {
    "timestamp": "2026-01-13T12:48:04",
    "n_agents": 20,
    "dimensions": 12,
    "num_steps": 300
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

### CSV Format (`timeseries.csv`)
Time-series data for plotting and statistical analysis:

```
step,coherence,crystallization_events,phase_0,manifold_energy,max_phase,min_phase
0,0.207,0,1.234,2.456,2.891,0.123
1,0.292,0,0.891,2.123,2.654,0.456
...
```

### Using the Data

**Python & Pandas:**
```python
import pandas as pd
df = pd.read_csv('results/run_*/timeseries.csv')
print(df['coherence'].describe())
```

**Excel / Spreadsheets:**
- Open CSV files directly in Excel, Google Sheets, or Numbers
- Create charts, pivot tables, and custom analysis

**Jupyter Notebook:**
- Use provided `analyze_results.ipynb`
- Load multiple runs for comparison
- Generate publication-quality visualizations

---

## 🧪 Running Multiple Experiments

Edit `simulation_runner.py`:

```python
runner = SimulationRunner(n_agents=20, dimensions=12, num_steps=300)

# Run 5 experiments
for i in range(5):
    runner.run_simulation(run_name=f"experiment_{i+1}")

# List all completed runs
runner.list_runs()
```

Then compare results in the Jupyter notebook.

---

## 📁 Project Structure

```
Cymatic_Indra/
├── cymatic_indra/                 # Main package
│   ├── __init__.py
│   ├── oscillator.py              # IndraOscillator (The Jewels)
│   ├── manifold.py                # CymaticManifold (The Plate)
│   ├── core.py                    # IndraEngine (The Spider)
│   └── requirements.txt
├── visualize_indra_v2.py           # Interactive visualization
├── simulation_runner.py             # Enhanced runner with data export
├── analyze_results.ipynb            # Jupyter notebook for analysis
├── results/                         # Exported simulation data (auto-created)
│   └── run_YYYYMMDD_HHMMSS/
│       ├── simulation.json
│       ├── timeseries.csv
│       ├── config.json
│       └── analysis_report.txt
└── README.md
```

---

## 🔧 Configuration

Edit simulation parameters in `simulation_runner.py`:

```python
runner = SimulationRunner(
    n_agents=20,           # Number of oscillators
    dimensions=12,         # Dimensionality
    num_steps=300          # Simulation steps
)
```

Engine configuration:
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

## 📈 Performance Metrics

The system tracks key metrics exported to CSV:

- **Coherence ($Z$):** Global synchronization measure (0-1)
- **Crystallization Events:** Times when coherence > 0.8
- **Manifold Energy:** L2-norm of manifold geometry
- **Phase Dynamics:** Min/max/distribution of oscillator phases

Visualize these in the exported CSV files and Jupyter analysis.

---

## 🎓 Understanding the System

### Phase Locking

Adaptive Kuramoto-style coupling synchronizes oscillators:
$$\dot{\phi}_i = \omega_i + k(t) \sum_j a_{ij} \sin(\phi_j - \phi_i)$$

Where:
- $\phi_i$ is the phase of oscillator $i$
- $\omega_i$ is the natural frequency
- $a_{ij}$ is the coupling strength
- $k(t)$ adapts with coherence (adaptive cooling)

### Crystallization

When global coherence $Z = |M|$ (order parameter) exceeds 0.8, the system enters a "crystallized" state where the manifold structure becomes stable.

$$M = \frac{1}{N} \sum_i e^{i\phi_i}$$

---

## 📝 License

This project is inspired by Buddhist philosophy and topological mathematics. Use and modify as you see fit.

---

## 👤 Author

Randy Swine / ronswine620-cyber

Generated: January 13, 2026
