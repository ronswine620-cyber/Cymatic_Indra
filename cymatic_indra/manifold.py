import numpy as np
from collections import deque

# --- Geometric Utilities ---
def pairwise_distances(X):
    X = np.asarray(X, dtype=float)
    diffs = X[:, None, :] - X[None, :, :]
    return np.sqrt(np.sum(diffs * diffs, axis=-1))

class UnionFind:
    """Helper for Topological Data Analysis (TDA)"""
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0]*n
        self.components = n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return False
        if self.rank[ra] < self.rank[rb]: self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]: self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1
        self.components -= 1
        return True

# --- The Cymatic Manifold ---
class CymaticManifold:
    """
    The 'Plate'.
    A read-write topological memory that stores Structural Invariants.
    It takes vibrational input and 'crystallizes' it into geometry.
    """
    def __init__(self, dim=16, elasticity=0.05):
        self.geometry = np.zeros(dim, dtype=float)
        self.elasticity = elasticity  # How fast it returns to baseline (Memory decay)

    def get_geometry(self):
        """Read-Only access for oscillators."""
        return self.geometry.copy()

    def imprint(self, feedback_vector, force=0.05):
        """
        Writes to the manifold. 
        'force' determines how deep the cymatic pattern is etched.
        """
        # Elastic snapping back to previous invariant state + new deformation
        self.geometry = (1 - self.elasticity) * self.geometry + force * np.asarray(feedback_vector)

    def analyze_topology(self, phases):
        """
        Topological Data Analysis (Betti-0) to check for fractures in the Net.
        Returns the number of connected components (clusters).
        """
        # Map phases to unit circle coords for distance calculation
        X = np.stack([np.cos(phases), np.sin(phases)], axis=1)
        
        # Simple persistence check at a fixed epsilon
        epsilon = 0.5 
        n = X.shape[0]
        uf = UnionFind(n)
        dists = pairwise_distances(X)
        
        # Connect components that are close (phase-locked)
        for i in range(n):
            for j in range(i + 1, n):
                if dists[i, j] < epsilon:
                    uf.union(i, j)
        
        return uf.components