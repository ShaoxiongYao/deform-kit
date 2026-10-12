"""Penalty contact with optional Coulomb friction."""

import warp as wp

from deformkit.collision.contact import ContactData, ContactModel


@wp.func
def penalty_normal_force(
    c: ContactData, k: int, v: wp.array(dtype=wp.vec3), stiffness: float, damping: float, thickness: float
):
    """f_n = max(0, k d - c v_n), with d = thickness - gap and v_n < 0 while approaching."""
    depth = thickness - c.gap[k]
    if depth <= 0.0:
        return 0.0
    nodes = c.nodes[k]
    w = c.weights[k]
    v_rel = wp.vec3(0.0)
    for j in range(4):
        if nodes[j] >= 0:
            v_rel += w[j] * v[nodes[j]]
    return wp.max(stiffness * depth - damping * wp.dot(v_rel, c.normal[k]), 0.0)


@wp.kernel
def penalty_normal_kernel(
    c: ContactData,
    v: wp.array(dtype=wp.vec3),
    stiffness: float,
    damping: float,
    thickness: float,
    force: wp.array(dtype=wp.vec3),
    share: wp.array(dtype=wp.float32),
):
    k = wp.tid()
    if k >= wp.min(c.count[0], c.nodes.shape[0]):
        return
    fn = penalty_normal_force(c, k, v, stiffness, damping, thickness)
    if fn <= 0.0:
        return
    nodes = c.nodes[k]
    w = c.weights[k]
    f = fn * c.normal[k]
    for j in range(4):
        if nodes[j] >= 0:
            wp.atomic_add(force, nodes[j], w[j] * f)
            wp.atomic_add(share, nodes[j], wp.abs(w[j]))  # how many active contacts touch the node


@wp.kernel
def penalty_friction_kernel(
    c: ContactData,
    v: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=wp.float32),
    force: wp.array(dtype=wp.vec3),
    share: wp.array(dtype=wp.float32),
    stiffness: float,
    damping: float,
    thickness: float,
    mu: float,
    dt: float,
    friction: wp.array(dtype=wp.vec3),
):
    k = wp.tid()
    if k >= wp.min(c.count[0], c.nodes.shape[0]):
        return
    fn = penalty_normal_force(c, k, v, stiffness, damping, thickness)
    if fn <= 0.0:
        return

    # relative velocity after this step if friction were absent, and the
    # contact's effective inverse mass (each node's mass split among its contacts)
    nodes = c.nodes[k]
    w = c.weights[k]
    v_rel = wp.vec3(0.0)
    inv_mass = float(0.0)
    for j in range(4):
        if nodes[j] >= 0:
            vj = v[nodes[j]]
            m = mass[nodes[j]]
            if m > 0.0:
                vj = vj + force[nodes[j]] * (dt / m)
                inv_mass += w[j] * w[j] * share[nodes[j]] / m
            v_rel += w[j] * vj

    n = c.normal[k]
    v_t = v_rel - wp.dot(v_rel, n) * n
    speed = wp.length(v_t)
    if speed < 1.0e-9 or inv_mass <= 0.0:
        return

    # Coulomb cone, capped at the force that stops the tangential motion this step
    ft = wp.min(mu * fn, speed / (dt * inv_mass))
    f = -ft / speed * v_t
    for j in range(4):
        if nodes[j] >= 0:
            wp.atomic_add(friction, nodes[j], w[j] * f)


@wp.kernel
def add_kernel(a: wp.array(dtype=wp.vec3), b: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    a[i] = a[i] + b[i]


class PenaltyContact(ContactModel):
    """Penalty normal force plus optional Coulomb friction.

    Normal:   f_n = max(0, k d - c v_n),  d = thickness - gap.  v_n < 0 while
              approaching, so damping adds repulsion; the clamp keeps it from
              ever pulling the sides together.
    Friction: |f_t| <= friction * f_n, opposing the tangential relative velocity
              predicted for the end of the step. The force is capped at what
              stops that motion, so it sticks without adding a stiffness limit
              on dt. A node shared by several contacts has its mass split among
              them so their combined friction cannot overshoot.

    Friction must run after all other forces of the step are accumulated (World
    applies contacts last), since it predicts velocities from `world.force`.
    """

    def __init__(self, stiffness=1e4, damping=10.0, thickness=1e-3, friction=0.0):
        self.stiffness = stiffness
        self.damping = damping
        self.thickness = thickness
        self.friction = friction
        self._share = None
        self._friction_force = None

    @property
    def margin(self):
        return self.thickness

    def apply(self, world, contacts):
        n = world.num_nodes
        if self._share is None or self._share.shape[0] != n:
            self._share = wp.zeros(n, dtype=wp.float32, device=world.device)
            self._friction_force = wp.zeros(n, dtype=wp.vec3, device=world.device)
        self._share.zero_()

        params = [self.stiffness, self.damping, self.thickness]
        wp.launch(
            penalty_normal_kernel,
            dim=contacts.capacity,
            inputs=[contacts.data, world.state.v, *params],
            outputs=[world.force, self._share],
            device=world.device,
        )
        if self.friction <= 0.0:
            return

        self._friction_force.zero_()
        wp.launch(
            penalty_friction_kernel,
            dim=contacts.capacity,
            inputs=[
                contacts.data,
                world.state.v,
                world.state.mass,
                world.force,
                self._share,
                *params,
                self.friction,
                world.dt,
            ],
            outputs=[self._friction_force],
            device=world.device,
        )
        wp.launch(add_kernel, dim=n, inputs=[world.force, self._friction_force], device=world.device)
