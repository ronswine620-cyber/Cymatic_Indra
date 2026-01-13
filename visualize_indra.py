import numpy as np
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
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
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
# Auto-detect if running in a headless environment (Cloud/Server)
HEADLESS = os.environ.get('DISPLAY') is None 

CONFIG = {
    'window_size': 16,
    'damping_factor': 0.6,
    'phase_wrap_thresh': 2.0,
    'coupling_strength': 0.2, 
    'plasticity_rate': 0.05
}

# --- Initialization ---
# Create the Jewels (Agents)
jewels = [IndraOscillator(dim=DIMENSIONS, base_freq=0.05 + i*0.002, seed=i) for i in range(N_AGENTS)]
# Create the Plate (Substrate)
plate = CymaticManifold(dim=DIMENSIONS, elasticity=0.1)
# Create the Spider (Engine)
spider = IndraEngine(jewels, config=CONFIG)

# --- Visualization Setup ---
# Dark Theme Setup
plt.style.use('dark_background')
fig = plt.figure(figsize=(12, 7), facecolor='#0b0d17')
fig.suptitle("CYMATIC INDRA: REAL-TIME MORPHOGENESIS", color='white', fontsize=18, fontweight='bold')

# 1. THE JEWELS (Phase Synchronization)
# Polar plot showing agent alignment
ax_phase = fig.add_subplot(121, projection='polar', facecolor='#0b0d17')
ax_phase.set_title("The Jewels (Phase Space)", color='#a0a0ff', pad=20)
ax_phase.set_ylim(0, 1.2)
ax_phase.grid(True, alpha=0.1, color='white')
ax_phase.set_yticklabels([]) # Hide radius labels for cleaner look

# Scatter plot: Angle = Phase, Radius = ~1.0
scat = ax_phase.scatter([], [], c=[], cmap='hsv', alpha=0.8, s=120, edgecolors='white', linewidth=0.5)

# 2. THE PLATE (Manifold Topology)
# Radar chart showing the geometric shape of the thought 

ax_struct = fig.add_subplot(122, projection='polar', facecolor='#0b0d17')
ax_struct.set_title("The Plate (Manifold Topology)", color='#39ff14', pad=20)
ax_struct.set_ylim(-1.1, 1.1)
ax_struct.grid(True, alpha=0.1, color='white')
ax_struct.set_yticklabels([])

# Prepare angles for radar chart
theta = np.linspace(0, 2 * np.pi, DIMENSIONS, endpoint=False)
theta = np.concatenate((theta, [theta[0]])) # Close the loop
line, = ax_struct.plot([], [], color='#00ced1', linewidth=3, alpha=0.8)

def init():
    """Initialize empty plot data."""
    scat.set_offsets(np.empty((0, 2)))
    line.set_data([], [])
    return scat, line

def update(frame):
    """Main simulation loop called every frame."""
    # --- A. Simulation Step ---
    plate_resonance = plate.get_geometry()
    raw_signals = [j.resonate(plate_resonance) for j in jewels]
    results = spider.step(raw_signals)
    
    # Write back to manifold if feedback exists
    if np.any(results['feedback']):
        plate.imprint(results['feedback'], force=0.15)
        
    # --- B. Visual Update 1: Jewels ---
    phases = np.array(results['phases'], dtype=float)
    # Ensure we have valid data
    if len(phases) == 0:
        phases = np.zeros(N_AGENTS)
    
    # Jitter radius slightly for "living" organic feel
    radii = np.ones(N_AGENTS) + np.random.uniform(-0.02, 0.02, N_AGENTS)
    
    # Update positions (Theta, R) and colors
    offsets = np.column_stack((phases, radii))
    scat.set_offsets(offsets)
    scat.set_array(phases)
    
    # --- C. Visual Update 2: Plate ---
    raw_geom = plate.get_geometry()
    
    # Normalize for visual clarity (Dynamic Scaling)
    max_val = np.max(np.abs(raw_geom))
    norm_factor = max_val if max_val > 0.1 else 1.0
    geom_disp = raw_geom / norm_factor
    geom_disp = np.concatenate((geom_disp, [geom_disp[0]])) # Close loop
    
    line.set_data(theta, geom_disp)
    
    # Update fill (Clear previous fill to avoid overlap artifacts)
    ax_struct.collections.clear()
    ax_struct.fill(theta, geom_disp, color='#00ced1', alpha=0.3)
    
    # --- D. Status Text ---
    status = "MORPHOGENESIS"
    state_color = "white"
    if results['coherence'] > 0.8:
        status = "CRYSTALLIZED"
        state_color = "#39ff14" # Green
    elif results['coherence'] < 0.3:
        status = "FRAGMENTED"
        state_color = "#ff3131" # Red

    fig.suptitle(f"CYMATIC INDRA | COHERENCE: {results['coherence']:.3f} | STATE: {status}", 
                 color=state_color, fontsize=16, fontweight='bold')
    
    return scat, line

# --- Execution ---
print(f"Initializing Cymatic Indra Visualization ({'HEADLESS' if HEADLESS else 'DISPLAY'} Mode)...")
ani = FuncAnimation(fig, update, init_func=init, frames=300, interval=50, blit=False)

if HEADLESS:
    output_path = 'indra_morphogenesis.gif'
    print(f"Rendering simulation to {output_path}...")
    try:
        # Use pillow for GIF (more portable than ffmpeg)
        ani.save(output_path, writer='pillow', fps=20, dpi=80)
        print("Render complete.")
    except Exception as e:
        print(f"\nError saving animation: {e}")
        print("Tip: Try installing ffmpeg for MP4 output")
else:
    plt.tight_layout(pad=3.0)
    plt.show()