from deformkit.core.backend import get_device
from deformkit.core.state import State


class SimObject:
    """Base class for dynamic objects.

    An object owns its internal physics (springs, elements, ...) and a collision
    shape, and nothing else. It never handles contact. Once added to a World its
    `state` arrays become views into the world's global node arrays, so objects
    always read and write their own nodes with local indices 0..vertex_count-1.
    """

    def __init__(self, x, v=None, mass=None, collision_shape=None, device=None):
        self.device = get_device(device)
        self.state = State.from_numpy(x, v, mass, device=self.device)
        self.collision_shape = collision_shape
        self.vertex_start = 0
        self.vertex_count = self.state.num_nodes
        self.object_id = -1

    def bind(self, state, vertex_start, object_id):
        """Called by World: alias `state` (a view into the global arrays)."""
        self.state = state
        self.vertex_start = vertex_start
        self.object_id = object_id

    def compute_internal_forces(self, force):
        """Accumulate internal forces into `force` ([vertex_count] vec3, local indexing)."""
