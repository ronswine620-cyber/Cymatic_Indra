import numpy as np
import matplotlib.pyplot as plt
from cymatic_indra.oscillator import IndraOscillator
from cymatic_indra.manifold import CymaticManifold
from cymatic_indra.core import IndraEngine

def main():
    print("Initializing Cymatic Indra System...")
    print(">> Weaving Net...")
    print(">> Tuning Oscillators...")
    
    # --- Configuration ---
    N_AGENTS = 12
    DIMENSIONS = 8
    STEPS = 200
    
    # --- Instantiation ---
    # 1. The Jewels (Agents)
    jewels = [IndraOscillator(dim=DIMENSIONS, base_freq=0.05 + i*0.005, seed=i) for i in range(N_AGENTS)]
    
    # 2. The Plate (Substrate)
    plate = CymaticManifold(dim=DIMENSIONS, elasticity=0.1)
    
    # 3. The Spider (Engine)
    spider = IndraEngine(jewels, config={
        'window_size': 16,
        'damping_factor': 0.6,    # Balanced at Edge of Chaos
        'phase_wrap_thresh': 1.5, # Strict filtering of Broken Functors
        'coupling_strength': 0.15,
        'plasticity_rate': 0.05
    })

    # --- Metric Storage ---
    history_coherence = []
    history_clusters = []

    print("-" * 75)
    print(f"{'Step':<6} | {'Coherence':<12} | {'Topology (Clusters)':<20} | {'State'}")
    print("-" * 75)

    # --- Simulation Loop ---
    for t in range(STEPS):
        
        # A. Read Manifold
        # Jewels sense the current shape of the plate
        plate_resonance = plate.get_geometry()
        
        # B. Oscillate
        # Jewels generate trajectories based on internal state + plate resonance
        raw_signals = [j.resonate(plate_resonance) for j in jewels]
        
        # C. Engine Compute
        # The spider processes the signals, attempts to phase-lock
        results = spider.step(raw_signals)
        
        # D. Imprint Manifold
        # If a strong invariant was found, etch it into the plate
        if np.any(results['feedback']):
            plate.imprint(results['feedback'], force=0.1)

        # E. Analysis (Maintenance)
        # Use TDA to check if the Net is fragmented
        n_clusters = plate.analyze_topology(results['phases'])
        
        # Logging
        history_coherence.append(results['coherence'])
        history_clusters.append(n_clusters)
        
        status = "Morphogenesis"
        if results['coherence'] > 0.8: status = "Crystallized"
        elif n_clusters > N_AGENTS // 2: status = "Fragmented / Noise"
        
        if t % 10 == 0:
            print(f"{t:<6} | {results['coherence']:<12.3f} | {n_clusters:<20} | {status}")

    # --- Visualization ---
    plt.figure(figsize=(10, 6))
    plt.suptitle('Cymatic Indra: System Telemetry', fontsize=16)
    
    plt.subplot(2, 1, 1)
    plt.plot(history_coherence, color='indigo', linewidth=2)
    plt.title('Coherence (Order Parameter)')
    plt.ylabel('Sync Level')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    plt.plot(history_clusters, color='teal', drawstyle='steps-post', linewidth=2)
    plt.title('Topological Integrity (Connected Components)')
    plt.ylabel('Clusters')
    plt.xlabel('Time Step')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()