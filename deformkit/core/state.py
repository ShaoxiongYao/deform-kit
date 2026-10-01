from dataclasses import dataclass

import numpy as np
import warp as wp

from deformkit.core.backend import to_float, to_vec3


@dataclass
class State:
    """Nodal state shared by every dynamic object.

    x:    [N] vec3 positions
    v:    [N] vec3 velocities
    mass: [N] nodal masses (0: node ignores forces and keeps its velocity)
    """

    x: wp.array
    v: wp.array
    mass: wp.array

    @classmethod
    def from_numpy(cls, x, v=None, mass=None, device=None):
        x = np.asarray(x, dtype=np.float32).reshape(-1, 3)
        v = np.zeros_like(x) if v is None else v
        mass = np.ones(len(x), dtype=np.float32) if mass is None else mass
        return cls(to_vec3(x, device), to_vec3(v, device), to_float(mass, device))

    @property
    def num_nodes(self):
        return self.x.shape[0]

    @property
    def device(self):
        return self.x.device

    def view(self, start, count):
        """A State whose arrays alias the node range [start, start + count)."""
        end = start + count
        return State(self.x[start:end], self.v[start:end], self.mass[start:end])
