#!/usr/bin/env python3
"""
Cymatic Indra - Enhanced Simulation Runner
Updated for Cybernetic Control Theory (DMD + LQR)
"""

import numpy as np
import json
import csv
import os
import sys
from datetime import datetime
from pathlib import Path

# --- Internal Imports ---
# We assume the project structure is:
# /Cymatic_Indra
#    /cymatic_indra (package)
#       core.py (Contains CyberneticIndraEngine)
#       oscillator.py
#       manifold.py
try:
    from cymatic_indra.oscillator import IndraOscillator
    from cymatic_indra.manifold import CymaticManifold
    # This now imports the Refactored Cybernetic Engine
    from cymatic_indra.core import IndraEngine 
except ImportError:
    # Fallback for running directly from root without package install
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from cymatic_indra.oscillator import IndraOscillator
        from cymatic_indra.manifold import CymaticManifold
        from cymatic_indra.core import IndraEngine
    except ImportError:
        print("CRITICAL ERROR: Could not import 'cymatic_indra' package.")
        sys.exit(1)


class SimulationRunner:
    """
    Manages simulation execution, data export, and stability analysis.
    """
    
    def __init__(self, n_agents=20, dimensions=12, num_steps=300, profile="LIQUID"):
        self.n_agents = n_agents
        self.dimensions = dimensions
        self.num_steps = num_steps
        self.results_dir = Path("results")
        self.profiles_path = Path("profiles.json")
        
        # Load configuration profile
        self.tuning_profiles = self._load_profiles()
        base_profile = self.tuning_profiles.get(profile, self.tuning_profiles.get("LIQUID", {}))
        self.config = base_profile.copy()
        self.profile_name = profile
        
        # Create results directory if needed
        self.results_dir.mkdir(exist_ok=True)
    
    def _load_profiles(self):
        if self.profiles_path.exists():
            with open(self.profiles_path, 'r') as f:
                return json.load(f)
        return {}
        
    def run_simulation(self, run_name=None, verbose=True):
        """
        Main execution loop for the Cybernetic Indra simulation.
        """
        # 1. Setup Run Metadata
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = run_name or f"run_{timestamp}"
        run_dir = self.results_dir / run_name
        run_dir.mkdir(exist_ok=True)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"CYBERNETIC INDRA SIMULATION (DMD + LQR)")
            print(f"{'='*60}")
            print(f"Run ID:    {run_name}")
            print(f"Profile:   {self.profile_name}")
            print(f"Agents:    {self.n_agents} | Dimensions: {self.dimensions}")
            print(f"Steps:     {self.num_steps}")
            print(f"{'='*60}\n")
        
        # 2. Initialize System Components
        # The Jewels (Agents)
        jewels = [IndraOscillator(dim=self.dimensions, seed=i) for i in range(self.n_agents)]
        
        # The Plate (Environment/Medium)
        plate = CymaticManifold(dim=self.dimensions, elasticity=0.05)
        
        # The Governor (Cybernetic Engine)
        spider = IndraEngine(jewels, config=self.config)
        
        # 3. Prepare Data Storage
        timeseries_data = {
            'step': [],
            'coherence': [],
            'manifold_energy': [],
            'max_eigenvalue': [],   # NEW: System Stability Metric
            'narrative_status': []  # NEW: 'DECAYING', 'STABLE', 'EXPLODING'
        }
        
        stats = {
            'max_coherence': 0.0,
            'avg_coherence': 0.0,
            'avg_eigenvalue': 0.0,
            'stability_score': 0.0 # Percentage of time system was stable
        }
        
        stable_steps = 0
        
        # 4. Main Physics Loop
        for step in range(self.num_steps):
            
            # A. Environmental Feedback
            # Agents sense the current vibration of the manifold
            plate_resonance = plate.get_geometry()
            
            # B. Agent Reaction
            # Agents output a signal based on their internal state + environment
            raw_signals = [j.resonate(plate_resonance) for j in jewels]
            
            # C. Cybernetic Control Step (DMD + LQR)
            # The Engine analyzes the signals and calculates optimal feedback
            results = spider.step(raw_signals)
            
            # D. Actuation
            # Apply the LQR feedback vector to the manifold
            feedback_vector = results['feedback']
            if np.any(feedback_vector):
                plate.imprint(feedback_vector, force=0.15)
            
            # E. Metrics Extraction
            coherence = results.get('coherence', 0.0)
            max_eig = results.get('max_eigenvalue', 0.0)
            status = results.get('status', 'INIT')
            manifold_energy = np.linalg.norm(plate_resonance)
            
            # Update Accumulators
            stats['avg_coherence'] += coherence
            stats['avg_eigenvalue'] += max_eig
            if coherence > stats['max_coherence']:
                stats['max_coherence'] = coherence
            if 0.95 <= max_eig <= 1.05:
                stable_steps += 1
            
            # Store Time Series
            timeseries_data['step'].append(step)
            timeseries_data['coherence'].append(float(coherence))
            timeseries_data['manifold_energy'].append(float(manifold_energy))
            timeseries_data['max_eigenvalue'].append(float(max_eig))
            timeseries_data['narrative_status'].append(status)
            
            # Real-time Logging
            if verbose and (step + 1) % 50 == 0:
                print(f"  Step {step+1:03d}: "
                      f"Coherence={coherence:.3f} | "
                      f"Stability (λ)={max_eig:.3f} [{status}]")
        
        # 5. Finalize Statistics
        stats['avg_coherence'] /= self.num_steps
        stats['avg_eigenvalue'] /= self.num_steps
        stats['stability_score'] = stable_steps / self.num_steps
        
        if verbose:
            print(f"\n{'='*60}")
            print("SIMULATION COMPLETE")
            print(f"Max Coherence:      {stats['max_coherence']:.4f}")
            print(f"Avg Stability (λ):  {stats['avg_eigenvalue']:.4f}")
            print(f"Stability Score:    {stats['stability_score']*100:.1f}%")
            print(f"{'='*60}\n")
        
        # 6. Export Data
        self._export_json(run_dir, stats)
        self._export_csv(run_dir, timeseries_data)
        
        print(f"Results saved to: {run_dir}")
        return run_dir, stats, None

    def _export_json(self, run_dir, stats):
        """Saves high-level statistics to JSON."""
        export_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'profile': self.profile_name,
                'engine_type': 'CYBERNETIC_DMD_LQR'
            },
            'config': self.config,
            'statistics': stats
        }
        with open(run_dir / 'simulation.json', 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def _export_csv(self, run_dir, timeseries_data):
        """Saves step-by-step telemetry to CSV."""
        csv_path = run_dir / 'timeseries.csv'
        if not timeseries_data['step']: return
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            header = list(timeseries_data.keys())
            writer.writerow(header)
            # Transpose rows
            for i in range(len(timeseries_data['step'])):
                writer.writerow([timeseries_data[key][i] for key in header])

if __name__ == '__main__':
    # Simple CLI Test
    import argparse
    parser = argparse.ArgumentParser(description='Cymatic Indra Simulation Runner')
    parser.add_argument('--steps', type=int, default=300, help='Number of simulation steps')
    parser.add_argument('--agents', type=int, default=20, help='Number of agents')
    parser.add_argument('--profile', type=str, default='LIQUID', help='Configuration profile')
    
    args = parser.parse_args()
    
    runner = SimulationRunner(
        n_agents=args.agents, 
        num_steps=args.steps, 
        profile=args.profile
    )
    runner.run_simulation()
