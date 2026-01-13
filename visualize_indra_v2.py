#!/usr/bin/env python3
"""
Cymatic Indra - Visualization Script
Displays the self-organizing topological intelligence in real-time
"""

import numpy as np
import matplotlib
# Force macOS native backend
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import sys

# --- Internal Imports ---
try:
    from cymatic_indra.oscillator import IndraOscillator
    from cymatic_indra.manifold import CymaticManifold
    from cymatic_indra.core import IndraEngine
except ImportError:
    # Fallback to ensure users run from root
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from cymatic_indra.oscillator import IndraOscillator
        from cymatic_indra.manifold import CymaticManifold
        from cymatic_indra.core import IndraEngine
    except ImportError:
        print("CRITICAL ERROR: Could not import 'cymatic_indra'.")
        print("Ensure you are running this script from the project root.")
        sys.exit(1)

# --- Configuration ---
N_AGENTS = 20
DIMENSIONS = 12
# Force interactive mode for local Mac
HEADLESS = False 

CONFIG = {
    'window_size': 32,
    'damping_factor': 0.6,
    'phase_wrap_thresh': 2.0,
    'coupling_strength': 0.1,
    'plasticity_rate': 0.05
}

print(f"\nInitializing Cymatic Indra ({'HEADLESS' if HEADLESS else 'DISPLAY'} Mode)...")

# --- Initialize System ---
jewels = [IndraOscillator(dim=DIMENSIONS, seed=i) for i in range(N_AGENTS)]
plate = CymaticManifold(dim=DIMENSIONS, elasticity=0.05)
spider = IndraEngine(jewels, config=CONFIG)

print(f"✓ Created {N_AGENTS} Oscillators")
print(f"✓ Created Manifold (dim={DIMENSIONS})")
print(f"✓ Created Engine\n")

# --- Run Simulation Without Visualization ---
if HEADLESS:
    print("Running simulation (headless mode)...\n")
    stats = {
        'max_coherence': 0.0,
        'avg_coherence': 0.0,
        'crystallization_events': 0,
        'coherence_history': []
    }
    
    NUM_STEPS = 300
    for step in range(NUM_STEPS):
        plate_resonance = plate.get_geometry()
        raw_signals = [j.resonate(plate_resonance) for j in jewels]
        results = spider.step(raw_signals)
        
        stats['coherence_history'].append(results['coherence'])
        stats['avg_coherence'] += results['coherence']
        
        if results['coherence'] > stats['max_coherence']:
            stats['max_coherence'] = results['coherence']
        
        if np.any(results['feedback']):
            plate.imprint(results['feedback'], force=0.15)
            if results['coherence'] > 0.8:
                stats['crystallization_events'] += 1
        
        if (step + 1) % 50 == 0:
            print(f"  Step {step+1}/{NUM_STEPS}: coherence={results['coherence']:.3f}, "
                  f"crystallizations={stats['crystallization_events']}")
    
    stats['avg_coherence'] /= NUM_STEPS
    
    print(f"\n=== SIMULATION COMPLETE ===")
    print(f"Max Coherence: {stats['max_coherence']:.3f}")
    print(f"Avg Coherence: {stats['avg_coherence']:.3f}")
    print(f"Crystallization Events: {stats['crystallization_events']}")
    print(f"Final Manifold Geometry (first 5 dims): {plate.get_geometry()[:5]}\n")

else:
    # --- Interactive Visualization (if display available) ---
    print("Setting up interactive visualization...\n")
    
    fig = plt.figure(figsize=(14, 6), facecolor='#0b0d17')
    fig.patch.set_facecolor('#0b0d17')
    
    # 1. THE JEWELS (Phase Space)
    ax_phase = fig.add_subplot(121, projection='polar', facecolor='#0b0d17')
    ax_phase.set_title("The Jewels (Phase Space)", color='#a0a0ff', pad=20)
    ax_phase.set_ylim(0, 1.2)
    ax_phase.grid(True, alpha=0.1, color='white')
    ax_phase.set_yticklabels([])
    
    scat = ax_phase.scatter([], [], c=[], cmap='hsv', alpha=0.8, s=120, edgecolors='white', linewidth=0.5)
    
    # 2. THE PLATE (Manifold Topology)
    ax_struct = fig.add_subplot(122, projection='polar', facecolor='#0b0d17')
    ax_struct.set_title("The Plate (Manifold Topology)", color='#39ff14', pad=20)
    ax_struct.set_ylim(-1.1, 1.1)
    ax_struct.grid(True, alpha=0.1, color='white')
    ax_struct.set_yticklabels([])
    
    theta = np.linspace(0, 2 * np.pi, DIMENSIONS, endpoint=False)
    theta = np.concatenate((theta, [theta[0]]))
    line, = ax_struct.plot([], [], color='#00ced1', linewidth=3, alpha=0.8)
    
    poly_container = [None]  # Mutable container to store polygon patch
    
    def init():
        scat.set_offsets(np.empty((0, 2)))
        line.set_data([], [])
        return scat, line
    
    def update(frame):
        plate_resonance = plate.get_geometry()
        raw_signals = [j.resonate(plate_resonance) for j in jewels]
        results = spider.step(raw_signals)
        
        if np.any(results['feedback']):
            plate.imprint(results['feedback'], force=0.15)
        
        # Update jewels
        phases = np.array(results['phases'], dtype=float)
        if len(phases) == 0:
            phases = np.zeros(N_AGENTS)
        
        radii = np.ones(N_AGENTS) + np.random.uniform(-0.02, 0.02, N_AGENTS)
        offsets = np.column_stack((phases, radii))
        scat.set_offsets(offsets)
        scat.set_array(phases)
        
        # Update manifold - amplify the geometry for visibility
        raw_geom = plate.get_geometry()
        
        # Amplify small values for better visualization
        raw_geom_amplified = raw_geom * 3.0
        raw_geom_amplified = np.clip(raw_geom_amplified, -1.1, 1.1)
        
        max_val = np.max(np.abs(raw_geom_amplified))
        norm_factor = max_val if max_val > 0.1 else 1.0
        geom_disp = raw_geom_amplified / norm_factor
        geom_disp = np.concatenate((geom_disp, [geom_disp[0]]))
        
        line.set_data(theta, geom_disp)
        
        # Update or create the filled polygon
        if poly_container[0] is not None:
            poly_container[0].remove()
        
        # Convert polar to cartesian for polygon
        xs = geom_disp * np.cos(theta)
        ys = geom_disp * np.sin(theta)
        poly = plt.Polygon(list(zip(xs, ys)), closed=True, color='#00ced1', alpha=0.2)
        ax_struct.add_patch(poly)
        poly_container[0] = poly
        
        # Status
        status = "MORPHOGENESIS"
        state_color = "white"
        if results['coherence'] > 0.8:
            status = "CRYSTALLIZED"
            state_color = "#39ff14"
        elif results['coherence'] < 0.3:
            status = "FRAGMENTED"
            state_color = "#ff3131"
        
        fig.suptitle(f"CYMATIC INDRA | COHERENCE: {results['coherence']:.3f} | STATE: {status}", 
                     color=state_color, fontsize=16, fontweight='bold')
        
        return scat, line
    
    ani = FuncAnimation(fig, update, init_func=init, frames=300, interval=50, blit=False)
    plt.tight_layout(pad=3.0)
    plt.show()

print("Done!")
