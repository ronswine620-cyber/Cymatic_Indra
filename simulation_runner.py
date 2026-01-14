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


TUNING_PROFILES = {}

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
        self.profiles_path = Path("profiles.json")
        
        # Load profiles
        self.tuning_profiles = self._load_profiles()
        
        # Load the selected profile, default to LIQUID if not found
        # IMPORTANT: Use copy() so we don't mutate the master profile in memory before the run starts
        base_profile = self.tuning_profiles.get(profile, self.tuning_profiles.get("LIQUID", {}))
        self.config = base_profile.copy()
        self.profile_name = profile
        self.start_config = base_profile.copy() # Keep a reference to check for evolution
        
        # Create results directory if needed
        self.results_dir.mkdir(exist_ok=True)
    
    def _load_profiles(self):
        if self.profiles_path.exists():
            with open(self.profiles_path, 'r') as f:
                return json.load(f)
        return {}
        
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
                # Handle list config (Tribes)
                # Handle list config (Tribes)
                # Look at the LIVE engine state, not the start config
                if hasattr(spider, 'agent_configs'):
                    # Grab Agent 0 as representative
                    base_coup = spider.agent_configs[0].get('coupling_strength', 0.0)
                    base_damp = spider.agent_configs[0].get('damping_factor', 0.0)
                else:
                     # Fallback
                    base_coup = self.config.get('coupling_strength', 0.0)
                    base_damp = self.config.get('damping_factor', 0.0)
                    
                coup = params.get('coupling', base_coup)
                damp = params.get('damping', base_damp)
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
        
        # Attempt Evolution
        evolved_name = self._check_for_evolution(run_dir, stats)
        
        if verbose:
            print("✓ Exported simulation.json")
            print("✓ Exported timeseries.csv")
            print("✓ Exported config.json")
            print(f"\nData saved to: {run_dir}\n")
        
        return run_dir, stats, evolved_name
    
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

    def _check_for_evolution(self, run_dir, stats):
        """
        Lamarckian Evolution: Checks if homeostasis changed the parameters significantly.
        If so, saves the evolved configuration as a new profile.
        """
        # Handle Heterogeneous Evolution separately? 
        # For now, if it's a list, we just check the AVERAGE drift or the First Agent.
        # Let's check the Mean Parameters.
        
        if isinstance(self.config, list):
            # Extract average end parameters
            end_coups = [c.get('coupling_strength', 0.1) for c in self.config]
            end_damps = [c.get('damping_factor', 0.6) for c in self.config]
            end_coup = sum(end_coups) / len(end_coups)
            end_damp = sum(end_damps) / len(end_damps)
            
            # Start config might be list or dict
            if isinstance(self.start_config, list):
                 start_coups = [c.get('coupling_strength', 0.1) for c in self.start_config]
                 start_damps = [c.get('damping_factor', 0.6) for c in self.start_config]
                 start_coup = sum(start_coups) / len(start_coups)
                 start_damp = sum(start_damps) / len(start_damps)
            else:
                 start_coup = self.start_config.get('coupling_strength', 0.1)
                 start_damp = self.start_config.get('damping_factor', 0.6)
                 
        else:
            # Classic Global Config
            start_coup = self.start_config.get('coupling_strength', 0.1)
            end_coup = self.config['coupling_strength']
            
            start_damp = self.start_config.get('damping_factor', 0.6)
            end_damp = self.config['damping_factor']
        
        # Threshold for importance (e.g., > 10% change)
        drift_coup = abs(end_coup - start_coup) / (start_coup + 1e-6)
        drift_damp = abs(end_damp - start_damp) / (start_damp + 1e-6)
        
        if drift_coup > 0.1 or drift_damp > 0.1:
            # Significant adaptation occurred -> Evolution!
            timestamp = datetime.now().strftime("%H%M%S")
            evolved_name = f"{self.profile_name}_EVO_{timestamp}"
            
            # Save to profiles.json
            self.tuning_profiles[evolved_name] = self.config.copy()
            with open(self.profiles_path, 'w') as f:
                json.dump(self.tuning_profiles, f, indent=2)
                
            print(f"\n🧬 EVOLUTION DETECTED! Created new profile: {evolved_name}")
            print(f"   Coupling: {start_coup:.3f} -> {end_coup:.3f}")
            print(f"   Damping:  {start_damp:.3f} -> {end_damp:.3f}")
            
            return evolved_name
        return None
    
    def run_generational_evolution(self, generations=1):
        """
        Runs evolutionary generations for all lineages.
        """
        lineages = ["LIQUID", "CRYSTAL", "CHAOS"]
        
        for gen in range(generations):
            print(f"\n🌀 GENERATION {gen+1}/{generations}")
            
            # 1. Identify latest profiles for each lineage
            current_profiles = self._load_profiles()
            latest_profiles = {}
            
            for lineage in lineages:
                # Find all keys starting with lineage name
                candidates = [k for k in current_profiles.keys() if k.startswith(lineage)]
                # Sort by length (longer = more timestamps = later) or just alphabetical timestamp
                # Naming convention: NAME_EVO_TIMESTAMP.
                # String sort works for standard timestamps
                candidates.sort() # Newest last
                if candidates:
                    latest = candidates[-1]
                    latest_profiles[lineage] = latest
            
            # 2. Run Simulations
            for lineage, profile_name in latest_profiles.items():
                print(f"\n--- Running Lineage: {lineage} (Profile: {profile_name}) ---")
                runner = SimulationRunner(
                    n_agents=self.n_agents, 
                    dimensions=self.dimensions, 
                    num_steps=self.num_steps, 
                    profile=profile_name
                )
                runner.run_simulation(run_name=f"gen_{gen+1}_{profile_name}")

    def run_tribal_simulation(self, tribes, run_name="tribal_conflict"):
        """
        Runs a simulation with heterogeneous agents partitioned into tribes.
        
        Args:
            tribes: Dict mapping ProfileName -> Count. 
                    e.g. {'LIQUID': 50, 'CHAOS': 50}
                    Sum of counts must equal self.n_agents.
        """
        # Validate counts
        total_requested = sum(tribes.values())
        if total_requested != self.n_agents:
            print(f"⚠️ Warning: Tribe counts ({total_requested}) do not match n_agents ({self.n_agents}). Adjusting n_agents.")
            self.n_agents = total_requested
            
        # Build Heterogeneous Agent Config List
        agent_configs = []
        tribe_map = [] # For debugging/logging who is who
        
        print("\n🏰 Assembling Tribes:")
        for profile_name, count in tribes.items():
            # Get profile config
            # Use 'latest evolved' version? For now, exact name or base name.
            # Let's support finding the latest if a base name is given?
            # For simplicity, precise lookup first.
            
            # Retrieve profile, default to LIQUID if missing
            profile_data = self.tuning_profiles.get(profile_name, self.tuning_profiles.get("LIQUID", {})).copy()
            
            # If profile_data is a list (from previous heterogeneous run), take the representative (e.g., mean or first)
            # Or better, if we are passing a list of configs to 'agent_configs', we'd need to expand it.
            # But here we are assigning 'count' agents to this profile.
            # If the profile itself is a list of N agents, that doesn't match 'count'.
            # So we should collapse the list to a single representative config for this new tribe.
            
            if isinstance(profile_data, list):
                # Take the first one as representative for now
                profile_data = profile_data[0]
            
            print(f"  - {profile_name}: {count} Agents (Coup={profile_data.get('coupling_strength'):.2f})")
            
            for _ in range(count):
                agent_configs.append(profile_data.copy())
                tribe_map.append(profile_name)
                
        # Override self.config with this list for the run initialization
        # The Engine is smart enough to handle the list.
        self.config = agent_configs 
        self.profile_name = "HETEROGENEOUS_TRIBES" # Special marker
        
        # Run!
        return self.run_simulation(run_name=run_name)

    def run_spatially_clustered_simulation(self, cluster_map, run_name="spatial_clusters"):
        """
        Runs a simulation where tribes are assigned to specific index ranges.
        Args:
            cluster_map: List of tuples (Start, End, ProfileName)
                         e.g. [(0, 33, 'CHAOS'), (34, 66, 'LIQUID'), (67, 100, 'CHAOS')]
        """
        agent_configs = [None] * self.n_agents
        
        print("\n🗺️ Assembling Spatial Clusters:")
        for start, end, profile_name in cluster_map:
            # Get profile
            if profile_name in self.tuning_profiles:
                p_cfg = self.tuning_profiles[profile_name]
            elif profile_name == "CHAOS":
                # Fallback default
                p_cfg = {"coupling_strength": 0.01, "damping_factor": 0.95, "window_size": 32, "phase_wrap_thresh": 2.0, "plasticity_rate": 0.05, "gardening_interval": 10, "prune_thresh": 0.1, "grow_prob": 0.01}
            else:
                p_cfg = self.tuning_profiles.get("LIQUID", {}) # Fallback
            
            # Handle heterogeneous sources
            if isinstance(p_cfg, list):
                # If source is a list, just take the first one or average? Let's take mean.
                p_cfg = p_cfg[0] 
                
            count = end - start
            print(f"  - [{start}-{end}] = {profile_name} ({count} Agents)")
            
            for i in range(start, end):
                if i < self.n_agents:
                    agent_configs[i] = p_cfg.copy()
                    
        # Fill any holes with Chaos
        for i in range(self.n_agents):
            if agent_configs[i] is None:
                agent_configs[i] = {"coupling_strength": 0.01, "damping_factor": 0.95, "window_size": 32, "phase_wrap_thresh": 2.0, "plasticity_rate": 0.05, "gardening_interval": 10, "prune_thresh": 0.1, "grow_prob": 0.01}
        
        self.config = agent_configs
        return self.run_simulation(run_name=run_name)

    def run_evolutionary_ladder(self, start_profile="CHAOS", start_n=100, max_generations=10):
        """
        Runs an infinite ascent where successful crystallization triggers a scale-up.
        Logic:
        1. Run Sim.
        2. If Coherence > 0.8 (Success):
           - New N = N * 1.5
           - Seed = Evolved Profile
           - Repeat.
        3. If Coherence < 0.8 (Fail):
           - Terminate.
           - "Natural Selection has filtered you out."
        """
        current_profile = start_profile
        current_n = start_n
        
        print(f"\n🪜 STARTING EVOLUTIONARY LADDER (Seed: {current_profile}, N={current_n})")
        
        for gen in range(max_generations):
            run_id = f"ladder_gen_{gen+1}_{current_profile}"
            print(f"\n🧗 CLIMBING RUNG {gen+1}: N={current_n}, Seed={current_profile}")
            
            # Instantiate with current scale
            # We assume dimensions stay fixed for now (24D)
            runner = SimulationRunner(n_agents=current_n, dimensions=24, num_steps=1000)
            
            # Force Heterogeneous execution (Tribal Path) to ensure parameter mutable list
            tribes = {current_profile: current_n}
            _, stats, evolved_name = runner.run_tribal_simulation(tribes, run_name=run_id)
            
            # Decision Gate
            # Dynamic Ladder Difficulty:
            # Gen 0 (Rung 1): Threshold 0.40
            # Gen 9 (Rung 10): Threshold 0.58
            survival_threshold = 0.40 + (gen * 0.02)
            
            max_z = stats.get('max_coherence', 0.0)
            
            print(f"📊 Ladder Goal: Max Z > {survival_threshold:.2f}")
            
            if max_z > survival_threshold:
                print(f"✅ PASSED SELECTION! (Max Z {max_z:.3f} > {survival_threshold:.2f})")
                
                if evolved_name:
                    current_profile = evolved_name
                else:
                    # If it passed but didn't evolve significant drift (stable), use same profile.
                    print("   (Stable profile maintained)")
                    
                # Scale Up!
                current_n = int(current_n * 1.5)
                
            else:
                print(f"❌ EXTINCTION EVENT. (Max Z {max_z:.3f} < {survival_threshold:.2f})")
                print("   The swarm failed to organize at this scale.")
                break

if __name__ == '__main__':
    # Run EVOLUTIONARY LADDER
    # Start with Thermal Emergence as the seed strategy (CHAOS constraint)
    # But wait, Thermal Emergence requires NO profile (just chaos + thermal logic).
    # The 'profile' argument loads configs.
    # We need to make sure the Thermal Logic is active.
    # Thermal logic is hardcoded in IndraEngine step.
    # So valid profiles are just parameter sets.
    
    # We start with CHAOS profile.
    
    SimulationRunner().run_evolutionary_ladder(start_profile="CHAOS", start_n=100, max_generations=10)
    
    # List all completed runs
    # SimulationRunner().list_runs()
