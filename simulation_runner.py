#!/usr/bin/env python3
"""
Cymatic Indra - Enhanced Simulation Runner with Data Export
Runs simulations and exports results as JSON + CSV for analysis
"""

import numpy as np
import json
import csv
import os
import sys
from datetime import datetime
from pathlib import Path

# --- Internal Imports ---
try:
    from cymatic_indra.oscillator import IndraOscillator
    from cymatic_indra.manifold import CymaticManifold
    from cymatic_indra.core import IndraEngine
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from cymatic_indra.oscillator import IndraOscillator
        from cymatic_indra.manifold import CymaticManifold
        from cymatic_indra.core import IndraEngine
    except ImportError:
        print("CRITICAL ERROR: Could not import 'cymatic_indra'.")
        sys.exit(1)


TUNING_PROFILES = {
    "LIQUID": {  # Your current "breathing" state
        'window_size': 32,
        'damping_factor': 0.6,
        'phase_wrap_thresh': 2.0,
        'coupling_strength': 0.1,
        'plasticity_rate': 0.05,
        'gardening_interval': 10,
        'prune_thresh': 0.1,
        'grow_prob': 0.01
    },
    "CRYSTAL": {  # Forces rapid consensus (Hive Mind)
        'window_size': 16,        # Smaller window = faster reaction
        'damping_factor': 0.8,    # High damping = strong phase forcing
        'phase_wrap_thresh': 3.0, # Very permissible (listens to everyone)
        'coupling_strength': 0.8, # High coupling = forced sync
        'plasticity_rate': 0.2,   # Manifold hardens quickly
        'gardening_interval': 5,  # Aggressive gardening
        'prune_thresh': 0.2,      # Strict pruning
        'grow_prob': 0.05         # Fast growth
    },
    "CHAOS": {  # High noise, zero memory
        'window_size': 64,        # Huge window = slow to react
        'damping_factor': 0.9,    # High damping = signals die instantly
        'phase_wrap_thresh': 0.5, # Strict = ignores almost everyone
        'coupling_strength': 0.01,# Tiny coupling = almost no influence
        'plasticity_rate': 0.0,   # Manifold never learns
        'gardening_interval': 20, # Slow gardening
        'prune_thresh': 0.05,     # Loose pruning
        'grow_prob': 0.001        # Rare growth
    }
}


class SimulationRunner:
    """Manages simulation execution and data export"""
    
    def __init__(self, n_agents=20, dimensions=12, num_steps=300, profile="LIQUID"):
        """
        Initialize the runner
        
        Args:
            n_agents: Number of oscillators
            dimensions: Dimensionality of oscillator space
            num_steps: Number of simulation steps to run
            profile: Name of the tuning profile to use
        """
        self.n_agents = n_agents
        self.dimensions = dimensions
        self.num_steps = num_steps
        self.results_dir = Path("results")
        
        # Load the selected profile, default to LIQUID if not found
        self.config = TUNING_PROFILES.get(profile, TUNING_PROFILES["LIQUID"])
        self.profile_name = profile
        
        # Create results directory if needed
        self.results_dir.mkdir(exist_ok=True)
        
    def run_simulation(self, run_name=None, verbose=True):
        """
        Run a complete simulation and export data
        
        Args:
            run_name: Optional custom name for this run
            verbose: Print progress information
            
        Returns:
            Tuple of (run_dir, stats_dict)
        """
        # Generate run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = run_name or f"run_{timestamp}"
        run_dir = self.results_dir / run_name
        run_dir.mkdir(exist_ok=True)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"CYMATIC INDRA SIMULATION")
            print(f"{'='*60}")
            print(f"Run: {run_name}")
            print(f"Agents: {self.n_agents} | Dimensions: {self.dimensions} | Steps: {self.num_steps}")
            print(f"Output: {run_dir}")
            print(f"{'='*60}\n")
        
        # Initialize system
        jewels = [IndraOscillator(dim=self.dimensions, seed=i) for i in range(self.n_agents)]
        plate = CymaticManifold(dim=self.dimensions, elasticity=0.05)
        spider = IndraEngine(jewels, config=self.config)
        
        if verbose:
            print("✓ Initialized 20 Oscillators")
            print("✓ Created Manifold")
            print("✓ Created Engine\n")
            print("Running simulation...\n")
        
        # Storage for time-series data
        timeseries_data = {
            'step': [],
            'coherence': [],
            'crystallization_events': [],
            'phase_0': [],  # Track first oscillator's phase
            'manifold_energy': [],
            'max_phase': [],
            'min_phase': []
        }
        
        # Run simulation
        stats = {
            'max_coherence': 0.0,
            'avg_coherence': 0.0,
            'min_coherence': 1.0,
            'crystallization_events': 0,
            'coherence_history': []
        }
        
        for step in range(self.num_steps):
            # Simulation step
            plate_resonance = plate.get_geometry()
            raw_signals = [j.resonate(plate_resonance) for j in jewels]
            results = spider.step(raw_signals)
            
            # Close the loop! (Apply phase corrections)
            corrections = results.get('phase_corrections', np.zeros(self.n_agents))
            for i, jewel in enumerate(jewels):
                jewel.sync_phase(corrections[i])
            
            # Update manifold
            if np.any(results['feedback']):
                plate.imprint(results['feedback'], force=0.15)
            
            # Track statistics
            coherence = results['coherence']
            phases = np.array(results['phases'], dtype=float)
            
            stats['coherence_history'].append(coherence)
            stats['avg_coherence'] += coherence
            
            if coherence > stats['max_coherence']:
                stats['max_coherence'] = coherence
            if coherence < stats['min_coherence']:
                stats['min_coherence'] = coherence
            
            if coherence > 0.8:
                stats['crystallization_events'] += 1
            
            # Store time-series data
            timeseries_data['step'].append(step)
            timeseries_data['coherence'].append(float(coherence))
            timeseries_data['crystallization_events'].append(stats['crystallization_events'])
            timeseries_data['phase_0'].append(float(phases[0]) if len(phases) > 0 else 0.0)
            timeseries_data['manifold_energy'].append(float(np.linalg.norm(plate_resonance)))
            timeseries_data['max_phase'].append(float(np.max(phases)) if len(phases) > 0 else 0.0)
            timeseries_data['min_phase'].append(float(np.min(phases)) if len(phases) > 0 else 0.0)
            
            # Progress output
            if verbose and (step + 1) % 50 == 0:
                params = results.get('debug_params', {})
                coup = params.get('coupling', self.config['coupling_strength'])
                damp = params.get('damping', self.config['damping_factor'])
                print(f"  Step {step+1}/{self.num_steps}: coherence={coherence:.3f}, "
                      f"crystallizations={stats['crystallization_events']}, "
                      f"Coup={coup:.3f}, Damp={damp:.3f}")
        
        stats['avg_coherence'] /= self.num_steps
        
        if verbose:
            print(f"\n{'='*60}")
            print("SIMULATION COMPLETE")
            print(f"{'='*60}")
            print(f"Max Coherence: {stats['max_coherence']:.3f}")
            print(f"Avg Coherence: {stats['avg_coherence']:.3f}")
            print(f"Min Coherence: {stats['min_coherence']:.3f}")
            print(f"Crystallization Events: {stats['crystallization_events']}")
            print(f"{'='*60}\n")
        
        # Export data
        self._export_json(run_dir, stats)
        self._export_csv(run_dir, timeseries_data)
        self._export_config(run_dir)
        
        if verbose:
            print("✓ Exported simulation.json")
            print("✓ Exported timeseries.csv")
            print("✓ Exported config.json")
            print(f"\nData saved to: {run_dir}\n")
        
        return run_dir, stats
    
    def _export_json(self, run_dir, stats):
        """Export complete simulation metadata and statistics as JSON"""
        export_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'profile': self.profile_name,
                'n_agents': self.n_agents,
                'dimensions': self.dimensions,
                'num_steps': self.num_steps,
                'version': '1.0'
            },
            'statistics': {
                'max_coherence': float(stats['max_coherence']),
                'avg_coherence': float(stats['avg_coherence']),
                'min_coherence': float(stats['min_coherence']),
                'crystallization_events': int(stats['crystallization_events']),
                'coherence_history': [float(x) for x in stats['coherence_history']]
            }
        }
        
        json_path = run_dir / 'simulation.json'
        with open(json_path, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def _export_csv(self, run_dir, timeseries_data):
        """Export time-series data as CSV for analysis"""
        csv_path = run_dir / 'timeseries.csv'
        
        if not timeseries_data['step']:
            return
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            header = list(timeseries_data.keys())
            writer.writerow(header)
            
            # Write data rows
            num_rows = len(timeseries_data['step'])
            for i in range(num_rows):
                row = [timeseries_data[key][i] for key in header]
                writer.writerow(row)
    
    def _export_config(self, run_dir):
        """Export simulation configuration as JSON"""
        config_data = {
            'engine_config': self.config,
            'parameters': {
                'n_agents': self.n_agents,
                'dimensions': self.dimensions,
                'num_steps': self.num_steps
            }
        }
        
        config_path = run_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    def list_runs(self):
        """List all completed simulation runs"""
        if not self.results_dir.exists():
            print("No results directory found.")
            return []
        
        runs = sorted([d for d in self.results_dir.iterdir() if d.is_dir()])
        
        print(f"\nCompleted Simulation Runs ({len(runs)}):")
        print("=" * 60)
        
        for run_dir in runs:
            sim_file = run_dir / 'simulation.json'
            if sim_file.exists():
                with open(sim_file, 'r') as f:
                    data = json.load(f)
                    stats = data.get('statistics', {})
                    meta = data.get('metadata', {})
                    print(f"  {run_dir.name}")
                    print(f"    Profile: {meta.get('profile', 'N/A')}")
                    print(f"    Max Coherence: {stats.get('max_coherence', 'N/A'):.3f}")
                    print(f"    Avg Coherence: {stats.get('avg_coherence', 'N/A'):.3f}")
                    print(f"    Crystallizations: {stats.get('crystallization_events', 'N/A')}")
                    print()
        
        return runs


if __name__ == '__main__':
    # Compare all 3 states
    for profile_name in ["LIQUID", "CRYSTAL", "CHAOS"]:
        print(f"\n--- Running Profile: {profile_name} ---")
        runner = SimulationRunner(
            n_agents=20, 
            dimensions=12, 
            num_steps=300, 
            profile=profile_name
        )
        runner.run_simulation(run_name=f"experiment_{profile_name}")
    
    # List all completed runs
    SimulationRunner().list_runs()
