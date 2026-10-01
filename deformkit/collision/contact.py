"""Contact buffer and contact response models.

Detection fills a `Contacts` buffer. A contact pairs one node `vertex` with an
"other side" given as up to three nodes with barycentric `weight`s (index -1
marks a static obstacle). `normal` points from the other side toward the vertex,
and `gap` is the signed separation after subtracting radii (negative when the
two sides overlap). A contact model turns these records into forces or
corrections. Neither side needs to know which object a node belongs to.
"""

import warp as wp


@wp.struct
class CollisionVertices:
    """Nodes that take part in collision (gathered from all objects' shapes)."""

    index: wp.array(dtype=wp.int32)  # global node index
    radius: wp.array(dtype=wp.float32)
    owner: wp.array(dtype=wp.int32)  # object id, used to skip self-contact


@wp.struct
class ContactData:
    count: wp.array(dtype=wp.int32)
    vertex: wp.array(dtype=wp.int32)
    other: wp.array(dtype=wp.vec3i)
    weight: wp.array(dtype=wp.vec3)
    normal: wp.array(dtype=wp.vec3)
    gap: wp.array(dtype=wp.float32)


@wp.func
def add_contact(c: ContactData, vertex: int, other: wp.vec3i, weight: wp.vec3, normal: wp.vec3, gap: float):
    k = wp.atomic_add(c.count, 0, 1)
    if k < c.vertex.shape[0]:
        c.vertex[k] = vertex
        c.other[k] = other
        c.weight[k] = weight
        c.normal[k] = normal
        c.gap[k] = gap


class Contacts:
    """Fixed-capacity GPU contact list."""

    def __init__(self, capacity, device):
        self.capacity = max(int(capacity), 1)
        d = ContactData()
        d.count = wp.zeros(1, dtype=wp.int32, device=device)
        d.vertex = wp.zeros(self.capacity, dtype=wp.int32, device=device)
        d.other = wp.zeros(self.capacity, dtype=wp.vec3i, device=device)
        d.weight = wp.zeros(self.capacity, dtype=wp.vec3, device=device)
        d.normal = wp.zeros(self.capacity, dtype=wp.vec3, device=device)
        d.gap = wp.zeros(self.capacity, dtype=wp.float32, device=device)
        self.data = d

    def clear(self):
        self.data.count.zero_()

    def count(self):
        """Number of detected contacts (synchronizes with the device).

        Raises if detection produced more contacts than fit; the extra ones were dropped.
        """
        n = int(self.data.count.numpy()[0])
        if n > self.capacity:
            raise RuntimeError(f"contact buffer overflow: {n} contacts, capacity {self.capacity}")
        return n


class ContactModel:
    """Turns detected contacts into a response. Subclasses: PenaltyContact, ..."""

    # Pairs closer than this are reported by detection.
    margin = 0.0

    def apply(self, world, contacts):
        raise NotImplementedError


@wp.kernel
def penalty_contact_kernel(
    c: ContactData,
    v: wp.array(dtype=wp.vec3),
    stiffness: float,
    damping: float,
    thickness: float,
    force: wp.array(dtype=wp.vec3),
):
    k = wp.tid()
    if k >= wp.min(c.count[0], c.vertex.shape[0]):
        return

    depth = thickness - c.gap[k]
    if depth <= 0.0:
        return

    i = c.vertex[k]
    o = c.other[k]
    w = c.weight[k]
    n = c.normal[k]

    # relative normal velocity of the vertex w.r.t. the other side
    v_other = wp.vec3(0.0)
    for j in range(3):
        if o[j] >= 0:
            v_other += w[j] * v[o[j]]
    vn = wp.dot(v[i] - v_other, n)

    f = wp.max(stiffness * depth - damping * vn, 0.0) * n
    wp.atomic_add(force, i, f)
    for j in range(3):
        if o[j] >= 0:
            wp.atomic_sub(force, o[j], w[j] * f)


class PenaltyContact(ContactModel):
    """f_n = max(0, k d - c v_n) n with penetration depth d = thickness - gap.

    v_n is the vertex velocity relative to the other side along n, so it is
    negative while approaching (damping adds repulsion) and positive while
    separating (damping reduces repulsion, but the clamp keeps it non-attractive).
    """

    def __init__(self, stiffness=1e4, damping=10.0, thickness=1e-3):
        self.stiffness = stiffness
        self.damping = damping
        self.thickness = thickness

    @property
    def margin(self):
        return self.thickness

    def apply(self, world, contacts):
        wp.launch(
            penalty_contact_kernel,
            dim=contacts.capacity,
            inputs=[contacts.data, world.state.v, self.stiffness, self.damping, self.thickness],
            outputs=[world.force],
            device=world.device,
        )
