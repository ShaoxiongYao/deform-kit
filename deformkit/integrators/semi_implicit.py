import warp as wp


@wp.kernel
def semi_implicit_euler_kernel(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=wp.float32),
    force: wp.array(dtype=wp.vec3),
    dt: float,
):
    i = wp.tid()
    m = mass[i]
    vi = v[i]
    if m > 0.0:  # zero-mass nodes ignore forces and keep their velocity (pinned if v = 0)
        vi = vi + force[i] * (dt / m)
    v[i] = vi
    x[i] = x[i] + vi * dt


class SemiImplicitEuler:
    """v <- v + dt f / m,  x <- x + dt v  (symplectic Euler)."""

    def step(self, world):
        wp.launch(
            semi_implicit_euler_kernel,
            dim=world.num_nodes,
            inputs=[world.state.x, world.state.v, world.state.mass, world.force, world.dt],
            device=world.device,
        )
