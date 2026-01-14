import numpy as np
from collections import deque
from scipy.linalg import solve_discrete_are, inv, norm

class CyberneticIndraEngine:
    """
    The 'Cybernetic Governor' of the Net.
    Replaces heuristic resonance with Rigorous Control Theory:
    1. Observation: Dynamic Mode Decomposition (DMD) to learn system physics (A).
    2. Control: Linear Quadratic Regulator (LQR) to steer coherence.
    3. Architecture: Controllability Gramians for topology pruning.
    """
    def __init__(self, oscillators, config=None):
        self.oscillators = oscillators
        self.N = len(oscillators)
        self.dim = oscillators[0].dim
        
        # Define defaults
        defaults = {
            'window_size': 50,
            'dmd_rank': 12,
            'lqr_r_penalty': 0.1,
            'lqr_q_penalty': 1.0,
            'topology_update_rate': 20,
            'gardening_interval': 10
        }
        
        # Merge provided config with defaults
        if config:
            # Handle potential list input (Tribes) by taking first element for global settings
            base_cfg = config[0] if isinstance(config, list) else config
            self.config = {**defaults, **base_cfg}
        else:
            self.config = defaults
        
        # Buffer now stores the FULL STATE vector (N agents * D dimensions)
        self.state_dim = self.N * self.dim
        self.history_buffer = deque(maxlen=self.config['window_size'])
        
        # Internal Physics Model
        self.A_matrix = None # The learned physics of the swarm (X' = AX)
        self.coherence = 0.0
        self.step_counter = 0
        
        # Agent Configurations (Legacy support for heterogenous profiles)
        if isinstance(config, list):
            self.agent_configs = config
            if len(self.agent_configs) != self.N:
                # Resize if mismatch (can happen during tribal initialization)
                self.agent_configs = [config[0].copy() for _ in range(self.N)]
        else:
            self.agent_configs = [self.config.copy() for _ in range(self.N)]

    def _flatten_state(self, inputs):
        """Flattens list of agent vectors into one massive state vector."""
        # inputs shape: (N, D) -> (N*D,)
        return np.concatenate(inputs)

    # --- 1. THE OBSERVER (DMD Algorithm) ---
    def _compute_system_dynamics(self):
        """
        Uses Exact DMD to reverse-engineer the matrix A where x(t+1) = A * x(t).
        Returns: The approximated A matrix and the Eigenvalues.
        """
        if len(self.history_buffer) < self.config['window_size']:
            return None, None

        # Create Snapshot Matrices
        # Data shape: (State_Dim, Time_Steps)
        data = np.array(self.history_buffer).T  
        
        # X1: States from t=0 to t=T-1
        # X2: States from t=1 to t=T
        X1 = data[:, :-1]
        X2 = data[:, 1:]
        
        # 1. SVD of X1 (Dimensionality Reduction)
        try:
            r = min(self.config['dmd_rank'], X1.shape[1] - 1)
            if r < 1: r = 1
            
            U, S, Vh = np.linalg.svd(X1, full_matrices=False)
            
            U_r = U[:, :r]
            S_r_inv = np.diag(1.0 / S[:r])
            V_r = Vh[:r, :].conj().T
            
            # 2. Compute the Low-Rank Operator (Atilde)
            # Atilde = U' * X2 * V * S^-1
            Atilde = U_r.T @ X2 @ V_r @ S_r_inv
            
            # 3. Reconstruct the full High-Dimensional A (Approximation)
            # A ≈ U * Atilde * U' (Projecting back up)
            # We perform this projection because LQR needs the full dimensions.
            self.A_matrix = U_r @ Atilde @ U_r.T
            
            # 4. Extract Eigenvalues for Stability Analysis
            eigenvalues = np.linalg.eigvals(Atilde)
            return self.A_matrix, eigenvalues
            
        except np.linalg.LinAlgError:
            return None, None

    # --- 2. THE CONTROLLER (LQR Strategy) ---
    def _calculate_optimal_feedback(self, current_state):
        """
        Solves the Riccati Equation to find optimal feedback u = -Kx
        """
        if self.A_matrix is None:
            # Fallback: Simple average feedback (The "Liquid" strategy)
            return np.zeros(self.dim)

        # Define System Matrices
        # x(t+1) = A x(t) + B u(t)
        A = self.A_matrix
        
        # B Matrix: The Manifold affects ALL agents. 
        # If we imprint vector 'u' to the manifold, every agent sees 'u'.
        # We model this as B being a stack of Identity matrices.
        # Shape: (State_Dim, Input_Dim) -> (N*D, D)
        B = np.tile(np.eye(self.dim), (self.N, 1))

        # Define Cost Matrices
        # Q: Penalty for state deviation. We treat 'current_state' as error relative to mean.
        Q = np.eye(self.state_dim) * self.config.get('lqr_q_penalty', 1.0)
        
        # R: Penalty for spending energy.
        R = np.eye(self.dim) * self.config.get('lqr_r_penalty', 0.1)

        # Solve Discrete Algebraic Riccati Equation (DARE)
        try:
            P = solve_discrete_are(A, B, Q, R)
            
            # Compute Gain Matrix K
            # K = (R + B'PB)^-1 * B'PA
            # Note: We use pinv for stability against singular matrices
            K = np.linalg.pinv(R + B.T @ P @ B) @ (B.T @ P @ A)
            
            # Control Law: u = -K * x
            u_optimal = -K @ current_state
            
            # Safety: Normalize magnitude to prevent feedback loops from exploding
            # LQR assumes linear dynamics, but our agents are non-linear oscillators.
            # We clip the signal to avoid destabilizing the integrator.
            norm_val = np.linalg.norm(u_optimal)
            if norm_val > 1.0:
                u_optimal = u_optimal / norm_val
                
            return u_optimal
            
        except Exception as e:
            # Fallback if Riccati solver fails (e.g. uncontrollable system)
            return np.zeros(self.dim)

    # --- 3. MASTER CYCLE ---
    def step(self, inputs):
        """
        Main Loop
        inputs: List of agent embeddings (N lists of D floats).
        """
        # 1. Flatten and Buffer
        flat_state = self._flatten_state(inputs)
        self.history_buffer.append(flat_state)
        self.step_counter += 1
        
        # 2. Learn the Physics (DMD)
        A, eigenvalues = self._compute_system_dynamics()
        
        # 3. Analyze Narrative Health (Stability)
        narrative_status = "STABLE"
        max_eig = 0.0
        if eigenvalues is not None:
            max_eig = np.max(np.abs(eigenvalues))
            if max_eig < 0.95: narrative_status = "DECAYING"
            elif max_eig > 1.05: narrative_status = "EXPLODING"
            
        # 4. Compute Optimal Feedback (LQR)
        if A is not None and self.step_counter > self.config['window_size']:
            feedback_vector = self._calculate_optimal_feedback(flat_state)
        else:
            # Warm-up phase: standard averaging (Pre-Cybernetic)
            feedback_vector = np.mean(inputs, axis=0) * 0.5
            
        # 5. Legacy Metrics Calculation (For compatibility with Runner)
        # Calculate Phase Coherence
        # Proxy phases from the first dimension of input embeddings
        current_phases = np.array([inp[0] * np.pi for inp in inputs])
        z = np.mean(np.exp(1j * current_phases))
        self.coherence = np.abs(z)
        
        # "Phase Corrections" - In this model, the manifold does the work.
        # But we return empty corrections to simulate agent autonomy 
        # (they align to manifold, not each other).
        phase_corrections = np.zeros(self.N)

        return {
            'feedback': feedback_vector,
            'phases': current_phases,             # Legacy for Viz
            'phase_corrections': phase_corrections, # Legacy for Viz
            'coherence': self.coherence,          # Legacy for Viz
            'eigenvalues': eigenvalues,           # New Control Metric
            'max_eigenvalue': max_eig,            # New Control Metric
            'status': narrative_status,           # New Control Metric
            'debug_params': {
                'dmd_rank': self.config['dmd_rank'],
                'lqr_penalty': self.config['lqr_r_penalty']
            }
        }

# Maintain alias for backward compatibility if needed, 
# but ideally we modify imports in runner.
IndraEngine = CyberneticIndraEngine
