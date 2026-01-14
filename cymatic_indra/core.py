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
        
        # Heterogeneity Support
        # self.agent_configs is a list of configs, one per agent.
        # If 'config' is passed as a dict, it's replicated for all.
        # If passed as a list, it's assigned directly.
        if isinstance(config, list):
            self.agent_configs = config
            if len(self.agent_configs) != self.N:
                raise ValueError("Agent config list must match number of oscillators")
            self.config = self.agent_configs[0] # Fallback for global access if needed
        else:
            self.config = config or {
                'window_size': 32,
                'damping_factor': 0.6,
                'phase_wrap_thresh': 2.0,
                'coupling_strength': 0.1,
                'plasticity_rate': 0.05,
                'gardening_interval': 10,
                'prune_thresh': 0.1,
                'grow_prob': 0.01
            }
            self.agent_configs = [self.config.copy() for _ in range(self.N)]

        self.buffers = [deque(maxlen=self.agent_configs[i]['window_size']) for i in range(self.N)]
        
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
        updates = np.zeros_like(phases)
        
        # Calculate Coherence first for regulation
        z = np.mean(np.exp(1j * phases))
        self.coherence = np.abs(z)

        # Calculate Local Coherence for each agent
        # Local Z = | Sum(exp(i * theta_neighbor)) / Degree |
        self.local_coherences = np.zeros(self.N)

        for i in range(self.N):
            # Heterogeneous parameters for Agent i
            cfg = self.agent_configs[i]
            coupling = cfg.get('coupling_strength', 0.1)
            damping = cfg.get('damping_factor', 0.6)
            thresh = cfg.get('phase_wrap_thresh', 2.0)
            plasticity = cfg.get('plasticity_rate', 0.05)
            
            # Adaptive cooling per agent (Global version, maybe remove?)
            # adaptive_coup = coupling * (1.1 - self.coherence)
            adaptive_coup = coupling # Let's trust the autonomy logic instead of interfering
            
            influence = 0.0
            neighbor_phases = []
            
            for j in range(self.N):
                weight = self.net_topology[i, j]
                if i == j or weight == 0: continue
                
                neighbor_phases.append(phases[j])

                # Phase difference
                diff = phases[j] - phases[i]
                # Wrap to [-pi, pi]
                diff = np.angle(np.exp(1j * diff))
                
                # --- ADVERSARIAL FILTER (The "Semantic Sieve") ---
                if np.abs(diff) > thresh:
                    self.connection_health[i, j] *= 0.95
                    continue 

                # Retrieve Sender Power (Neighbor's Coupling Strength)
                # Influence = Weight * Sender_Power * sin(diff)
                sender_coupling = self.agent_configs[j].get('coupling_strength', 0.1)
                
                # We can normalize sender coupling or just use raw.
                # If sender_coup > 1.0, it's LOUD.
                influence += weight * sender_coupling * np.sin(diff)
                
                # --- PLASTICITY UPDATE ---
                agreement = (1.0 + np.cos(diff)) * 0.5
                rate = plasticity
                self.connection_health[i, j] = (1 - rate) * self.connection_health[i, j] + rate * agreement

            # Calculate Local Coherence
            if neighbor_phases:
                # Add self phase for local cluster coherence
                cluster_phases = neighbor_phases + [phases[i]]
                local_z = np.abs(np.mean(np.exp(1j * np.array(cluster_phases))))
                self.local_coherences[i] = local_z
            else:
                self.local_coherences[i] = 1.0 # Solipsistic perfection

            # Apply Damping (Viscosity)
            update = adaptive_coup * influence * damping
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

    def _apply_thermal_dynamics(self):
        """
        Constraint-Based Homeostasis (The "Thermal" Model).
        
        Instead of dictating a mechanism ("If X, do Y"), we apply a systemic constraint:
        **"Entropy generates Heat."**
        
        1. Calculate Local Temperature T proportional to Local Disorder (1 - Z).
        2. Apply stochastic thermal noise to parameters based on T.
        
        Dynamics:
        - High Disorder (Low Z) -> High Temp -> High Jitter -> Active Search (Melting).
        - High Order (High Z) -> Low Temp -> Low Jitter -> Structural Lock (Freezing).
        
        This allows profiles to emerge as 'Free Energy Minima' rather than scripted states.
        """
        # Thermal Scale (How much heat is generated by entropy)
        kb = 0.15 # Higher thermal vibration to explore higher coupling
        
        for i, cfg in enumerate(self.agent_configs):
            local_z = self.local_coherences[i]
            
            # Constraint: Temperature T = (1.0 - Z)^2  (Squaring punishes low Z more)
            # Actually, let's keep it linear or slightly non-linear.
            # If Z=1.0, T=0.0 (Absolute Zero).
            # If Z=0.0, T=1.0 (Boiling).
            T = (1.0 - local_z) 
            
            curr_coup = cfg.get('coupling_strength', 0.1)
            curr_damp = cfg.get('damping_factor', 0.6)
            
            # Stochastic Drift (Brownian Motion in Parameter Space)
            # Noise is centered at 0, standard deviation T.
            # We want them to explore the WHOLE space if hot.
            
            # We also add a small 'energy cost' drift?
            # Constraint: Maintaining high coupling costs energy, so weak natural decay.
            decay = 0.001 # Reduced decay to allow parameters to stick at higher levels
            
            # Update Coupling
            noise_coup = np.random.normal(0, 1) * T * kb
            # Apply Noise + Natural Decay (Gravity)
            new_coup = curr_coup + noise_coup - (decay * curr_coup)
            cfg['coupling_strength'] = np.clip(new_coup, 0.01, 5.0)
            
            # Update Damping
            noise_damp = np.random.normal(0, 1) * T * kb
            # Damping doesn't necessarily decay, maybe it just jitters?
            new_damp = curr_damp + noise_damp
            cfg['damping_factor'] = np.clip(new_damp, 0.01, 0.2) # Max 0.2 to prevent overshoot instability

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
        
        # Apply Thermal Dynamics (CONSTANT VIBRATION)
        self._apply_thermal_dynamics()

        if self.step_counter % self.config['gardening_interval'] == 0:
            self._garden_topology()
            
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
