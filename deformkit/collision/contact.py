"""Contact records and the contact-model interface.

A contact couples up to four nodes through signed weights: the relative
position of the two sides is  sum_k w_k x_k,  with positive weights on side A
and negative weights on side B (index -1 marks an unused slot, e.g. a static
obstacle). Every primitive fits this form:

    vertex-obstacle:  nodes (p, -1, -1, -1)   weights (1, 0, 0, 0)
    vertex-triangle:  nodes (p, a, b, c)      weights (1, -wa, -wb, -wc)
    edge-edge:        nodes (a0, a1, b0, b1)  weights (1-s, s, -(1-t), -t)

`normal` is the unit direction in which side A is pushed, and `gap` is the
signed separation along it after subtracting radii (negative when overlapping).
A contact model maps each record to a force  f  and applies  w_k f  to node k,
which conserves linear momentum because the weights sum to zero.
"""

import warp as wp


@wp.struct
class ContactData:
    count: wp.array(dtype=wp.int32)
    nodes: wp.array(dtype=wp.vec4i)
    weights: wp.array(dtype=wp.vec4)
    normal: wp.array(dtype=wp.vec3)
    gap: wp.array(dtype=wp.float32)


@wp.func
def add_contact(c: ContactData, nodes: wp.vec4i, weights: wp.vec4, normal: wp.vec3, gap: float):
    k = wp.atomic_add(c.count, 0, 1)
    if k < c.nodes.shape[0]:
        c.nodes[k] = nodes
        c.weights[k] = weights
        c.normal[k] = normal
        c.gap[k] = gap


@wp.func
def static_contact(c: ContactData, vertex: int, normal: wp.vec3, gap: float):
    add_contact(c, wp.vec4i(vertex, -1, -1, -1), wp.vec4(1.0, 0.0, 0.0, 0.0), normal, gap)


class Contacts:
    """Fixed-capacity GPU contact list. CollisionSystem grows it when detection overflows."""

    def __init__(self, capacity, device):
        self.capacity = max(int(capacity), 1)
        d = ContactData()
        d.count = wp.zeros(1, dtype=wp.int32, device=device)
        d.nodes = wp.zeros(self.capacity, dtype=wp.vec4i, device=device)
        d.weights = wp.zeros(self.capacity, dtype=wp.vec4, device=device)
        d.normal = wp.zeros(self.capacity, dtype=wp.vec3, device=device)
        d.gap = wp.zeros(self.capacity, dtype=wp.float32, device=device)
        self.data = d

    def clear(self):
        self.data.count.zero_()

    def num_detected(self):
        """Contacts reported by detection, possibly more than were stored (synchronizes)."""
        return int(self.data.count.numpy()[0])

    def count(self):
        """Number of stored contacts (synchronizes). Raises if contacts were dropped."""
        n = self.num_detected()
        if n > self.capacity:
            raise RuntimeError(f"contact buffer overflow: {n} contacts, capacity {self.capacity}")
        return n

    def numpy(self):
        """Stored contacts as a dict of NumPy arrays, for inspection and tests."""
        n = self.count()
        d = self.data
        return {
            "nodes": d.nodes.numpy()[:n],
            "weights": d.weights.numpy()[:n],
            "normal": d.normal.numpy()[:n],
            "gap": d.gap.numpy()[:n],
        }


class ContactModel:
    """Turns detected contacts into a response. See PenaltyContact."""

    # Pairs closer than this are reported by detection.
    margin = 0.0

    def apply(self, world, contacts):
        raise NotImplementedError
