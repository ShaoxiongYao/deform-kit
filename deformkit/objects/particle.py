import numpy as np

from deformkit.collision.geometry import Points
from deformkit.core.object import SimObject


class ParticleSystem(SimObject):
    """Independent particles with no internal forces; each collides as a sphere."""

    def __init__(self, positions, radius=0.01, mass=1.0, velocities=None, device=None):
        positions = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
        n = len(positions)
        mass = np.broadcast_to(np.asarray(mass, dtype=np.float32), (n,))
        super().__init__(positions, velocities, mass, Points(np.arange(n), radius), device)
