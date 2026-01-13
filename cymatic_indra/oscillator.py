import numpy as np

class IndraOscillator:
    """
    The 'Jewel' of the Net.
    A discrete agent that does not output tokens, but emits a 'Cymatic Trajectory'
    (oscillatory embedding) representing a semantic wavefront.
    """
    def __init__(self, dim=16, base_freq=0.02, drift=0.0005, seed=None):
        self.dim = dim
        self.t = 0
        self.base_freq = base_freq
        self.drift = drift
        self.rng = np.random.default_rng(seed)

        # Initial conditions (The "Seed" of the thought)
        self.phase = self.rng.uniform(0, 2*np.pi, size=dim)
        self.detune = self.rng.normal(0, 0.002, size=dim)

    def resonate(self, net_influence=None):
        """
        Generates the next step in the cymatic trajectory.
        net_influence: A perturbation from the Cymatic Manifold (the 'Web').
        """
        self.t += 1
        
        # Calculate current frequency based on base + drift
        freqs = self.base_freq + self.detune + self.drift * self.t
        
        # Fundamental oscillation
        signal = np.sin(self.phase + 2*np.pi*freqs*self.t)
        
        # Apply topological perturbation (The Net pushing back)
        if net_influence is not None:
            # Subtle coupling: the manifold bends the light of the jewel
            signal += 0.05 * net_influence

        # Add stochastic noise (simulating "temperature" / plasticity)
        signal += 0.05 * self.rng.normal(size=self.dim)
        
        return signal.astype(np.float64)

    def sync_phase(self, correction):
        """
        Adjusts the internal phase based on the Engine's feedback.
        correction: Scalar phase shift (radians)
        """
        # Apply correction to all dimensions to maintain relative harmonic structure
        self.phase = (self.phase + correction) % (2*np.pi)