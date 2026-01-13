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
            'plasticity_rate': 0.05,
            'gardening_interval': 10, # How often to prune/grow connections
            'prune_thresh': 0.1,      # Connection health below this dies
            'grow_prob': 0.01         # Probability of spontaneous new connection
        }

        self.buffers = [deque(maxlen=self.config['window_size']) for _ in range(self.N)]
        
        # Build Indra's Net (The Spatial Hierarchy)
        self.net_topology = self._weave_net()
        
        # System health metrics
        self.coherence = 0.0
        self.invariants_detected = 0
        self.step_counter = 0
        
        # Connection Health (Plasticity Memory)
        # Tracks how well neighbors resonate. 1.0 = Perfect sync, 0.0 = Dissonance
        self.connection_health = np.zeros((self.N, self.N))
        # Initialize health for existing connections
        self.connection_health[self.net_topology > 0] = 0.5

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
        # Adaptive cooling: As coherence increases, coupling decreases
        adaptive_coup = self.config['coupling_strength'] * (1.1 - self.coherence)

        updates = np.zeros_like(phases)

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
                    # Penalty for adversarial noise
                    self.connection_health[i, j] *= 0.95
                    continue 

                influence += weight * np.sin(diff)
                
                # --- PLASTICITY UPDATE (Hebbian Learning) ---
                # "Cells that fire together, wire together"
                # Agreement: 1.0 if diff=0, 0.0 if diff=pi
                agreement = (1.0 + np.cos(diff)) * 0.5
                rate = self.config['plasticity_rate']
                self.connection_health[i, j] = (1 - rate) * self.connection_health[i, j] + rate * agreement

            # Apply Damping (Viscosity)
            update = adaptive_coup * influence * self.config['damping_factor']
            updates[i] = update
            new_phases[i] = (phases[i] + update) % (2*np.pi)
            
        return new_phases, updates

    def _garden_topology(self):
        """
        Structural Plasticity: Prunes weak connections and grows new ones.
        """
        # 1. Prune
        # Find connections that are active but have low health
        mask_prune = (self.net_topology > 0) & (self.connection_health < self.config['prune_thresh'])
        if np.any(mask_prune):
            self.net_topology[mask_prune] = 0.0
            self.connection_health[mask_prune] = 0.0
            
        # 2. Grow
        # Randomly attempting to add connections
        # For small N, we can just iterate a few random pairs
        num_attempts = int(self.N * 0.5) # Heuristic
        for _ in range(num_attempts):
            i, j = np.random.choice(self.N, 2, replace=False)
            if self.net_topology[i, j] == 0:
                if np.random.random() < self.config['grow_prob']:
                    # New connection!
                    self.net_topology[i, j] = 0.5
                    self.net_topology[j, i] = 0.5 # Undirected
                    self.connection_health[i, j] = 0.5
                    self.connection_health[j, i] = 0.5

    def _regulate_homeostasis(self):
        """
        Homeostatic Regulation: Auto-tunes parameters to maintain 'Edge of Chaos'.
        Target Coherence: 0.3 - 0.7
        """
        # "Hormonal Response"
        if self.coherence < 0.3:
            # Too Chaotic -> Harden the system
            self.config['coupling_strength'] = min(1.2, self.config['coupling_strength'] + 0.005)
            self.config['damping_factor'] = min(0.95, self.config['damping_factor'] + 0.005)
        elif self.coherence > 0.7:
            # Too Rigid -> Soften the system
            self.config['coupling_strength'] = max(0.01, self.config['coupling_strength'] - 0.005)
            self.config['damping_factor'] = max(0.1, self.config['damping_factor'] - 0.005)

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
        next_phases, corrections = self._phase_lock(current_phases)
        
        # 4. Feedback Vector (What gets written back to the Manifold)
        # If coherence is high, we reinforce this structure
        feedback = np.zeros_like(inputs[0])
        if self.coherence > 0.8:
            # "Crystallization Event"
            feedback = np.mean(inputs, axis=0)
            self.invariants_detected += 1

        # 5. Maintenance (Gardening & Homeostasis)
        self.step_counter += 1
        if self.step_counter % self.config['gardening_interval'] == 0:
            self._garden_topology()
            self._regulate_homeostasis()
            
        return {
            'phases': next_phases,
            'phase_corrections': corrections,
            'coherence': self.coherence,
            'feedback': feedback,
            'debug_params': { # Return current params for observation
                'coupling': self.config['coupling_strength'],
                'damping': self.config['damping_factor']
            }
        }
