#!/usr/bin/env python3
"""Test script to verify imports and basic simulation work."""

import sys
import os
import numpy as np

# Add root to path
sys.path.insert(0, '/Users/randy/Documents/GitHub/projects/Cymatic_Indra')

print("Testing imports...")

try:
    from cymatic_indra import IndraOscillator, CymaticManifold, IndraEngine
    print("✓ Successfully imported all modules")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test basic instantiation
print("\nTesting basic instantiation...")

N_AGENTS = 5
DIMENSIONS = 12

try:
    jewels = [IndraOscillator(dim=DIMENSIONS, seed=i) for i in range(N_AGENTS)]
    print(f"✓ Created {N_AGENTS} oscillators")
except Exception as e:
    print(f"✗ Failed to create oscillators: {e}")
    sys.exit(1)

try:
    plate = CymaticManifold(dim=DIMENSIONS)
    print("✓ Created manifold")
except Exception as e:
    print(f"✗ Failed to create manifold: {e}")
    sys.exit(1)

try:
    spider = IndraEngine(jewels, config={
        'window_size': 8,
        'damping_factor': 0.6,
        'phase_wrap_thresh': 2.0,
        'coupling_strength': 0.1,
        'plasticity_rate': 0.05
    })
    print("✓ Created engine")
except Exception as e:
    print(f"✗ Failed to create engine: {e}")
    sys.exit(1)

# Test simulation step
print("\nTesting simulation...")

try:
    for step in range(5):
        plate_resonance = plate.get_geometry()
        raw_signals = [j.resonate(plate_resonance) for j in jewels]
        results = spider.step(raw_signals)
        
        print(f"  Step {step+1}: coherence={results['coherence']:.3f}, phases shape={np.array(results['phases']).shape}")
        
        if np.any(results['feedback']):
            plate.imprint(results['feedback'], force=0.15)
    
    print("✓ Simulation ran successfully for 5 steps")
except Exception as e:
    print(f"✗ Simulation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ All tests passed!")
