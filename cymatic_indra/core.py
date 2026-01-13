import numpy as np
from collections import deque

class IndraEngine:
    """
    The 'Spider' of the Net. Handles:
    1. Spectral Sieve (CLT)
    2. Cymatic Resonance (Phase Locking)
    3. Net Maintenance (Gardening)
    """
    def __init__(self, oscillators, config=None):
        self.oscillators = oscillators
        self.N = len(oscillators)
        
        self.config = config or {
            'window_size': 32,
            'damping_factor': 0.6,    # The "Edge of Chaos" viscosity
            'phase_wrap_thresh': 2.0, # Filters "Broken Functors" (Adversarial Noise)
            'coupling_strength': 0.1,
            'plasticity_rate': 0.05
        }

        self.buffers = [deque(maxlen=self.config['window_size']) for _ in range(self.N)]
        
        # Build Indra's Net (The Spatial Hierarchy)
        self.net_topology = self._weave_net()
        
        # System health metrics
        self.coherence = 0.0
        self.invariants_detected = 0

    def _weave_net(self):
        """Weaves the static neighborhood connections."""
        # Simple ring lattice + random shortcuts for small-world property
        adj = np.eye(self.N) * 0.5
        for i in range(self.N):
            adj[i, (i+1)%self.N] = 0.5
            adj[i, (i-1)%self.N] = 0.5
        return adj

    # --- Layer 1: CLT / Interface (Spectral Sieve) ---
    def _extract_frequency(self, trajectory):
        """
        Simulates the 'Functorial Mapping' from signal to frequency invariant.
        Returns dominant frequency if stable, else None.
        """
        if len(trajectory) < self.config['window_size']:
            return None
        
        # Convert trajectory to array of first dimensions for FFT
        traj_data = np.array([np.asarray(t)[0] if hasattr(t, '__len__') else t for t in trajectory])
        
        # Simple FFT
        spectrum = np.abs(np.fft.fft(traj_data))
        freqs = np.fft.fftfreq(len(spectrum))
        
        # Find dominant peak in positive spectrum (avoid DC component)
        half_len = len(spectrum) // 2
        if half_len > 1:
            idx = np.argmax(spectrum[1:half_len]) + 1
            idx = min(idx, len(freqs) - 1)  # Safety bound check
        else:
            idx = 1
        
        return freqs[idx]

    # --- Layer 2: Substrate Logic (Resonance) ---
    def _phase_lock(self, phases):
        """
        The core thinking process. 
        Adjusts phases based on neighbors, filtering out 'Adversarial Noise'.
        """
        new_phases = phases.copy()
        
        # Calculate Global Order Parameter (Coherence)
        z = np.mean(np.exp(1j * phases))
        self.coherence = np.abs(z)
        
        # Adaptive cooling: As coherence increases, coupling decreases
        adaptive_coup = self.config['coupling_strength'] * (1.1 - self.coherence)

        for i in range(self.N):
            influence = 0.0
            for j in range(self.N):
                weight = self.net_topology[i, j]
                if i == j or weight == 0: continue
                
                # Phase difference
                diff = phases[j] - phases[i]
                # Wrap to [-pi, pi]
                diff = np.angle(np.exp(1j * diff))
                
                # --- ADVERSARIAL FILTER (The "Semantic Sieve") ---
                # If phase difference is too massive, it's treated as a "Broken Functor"
                # (Adversarial input) and ignored.
                if np.abs(diff) > self.config['phase_wrap_thresh']:
                    continue 

                influence += weight * np.sin(diff)

            # Apply Damping (Viscosity)
            update = adaptive_coup * influence * self.config['damping_factor']
            new_phases[i] = (phases[i] + update) % (2*np.pi)
            
        return new_phases

    # --- Master Cycle ---
    def step(self, inputs):
        """
        Runs one tick of the Cymatic Indra system.
        inputs: List of current raw embeddings from IndraOscillators.
        """
        # 1. Buffer Inputs
        freq_invariants = []
        for i, val in enumerate(inputs):
            self.buffers[i].append(val)
            # Extract Invariant (Frequency)
            inv = self._extract_frequency(np.array(self.buffers[i]))
            if inv is not None:
                freq_invariants.append(inv)

        # 2. Compute Phases (Current state of the idea)
        # Simplified: Phase is proxied by the first dimension of the embedding
        current_phases = np.array([b[-1][0] * np.pi for b in self.buffers]) 
        
        # 3. Crystallize (Phase Lock)
        next_phases = self._phase_lock(current_phases)
        
        # 4. Feedback Vector (What gets written back to the Manifold)
        # If coherence is high, we reinforce this structure
        feedback = np.zeros_like(inputs[0])
        if self.coherence > 0.8:
            # "Crystallization Event"
            feedback = np.mean(inputs, axis=0)
            self.invariants_detected += 1
            
        return {
            'phases': next_phases,
            'coherence': self.coherence,
            'feedback': feedback
        }
