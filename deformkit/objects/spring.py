import numpy as np
import warp as wp

from deformkit.collision.geometry import Points, TriangleMesh
from deformkit.core.backend import to_float, to_int
from deformkit.core.object import SimObject
from deformkit.materials.spring import hookean_spring_force


@wp.kernel
def spring_force_kernel(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    springs: wp.array(dtype=wp.vec2i),
    rest_length: wp.array(dtype=wp.float32),
    stiffness: wp.array(dtype=wp.float32),
    damping: wp.array(dtype=wp.float32),
    force: wp.array(dtype=wp.vec3),
):
    s = wp.tid()
    i, j = springs[s][0], springs[s][1]
    f = hookean_spring_force(x[i], x[j], v[i], v[j], rest_length[s], stiffness[s], damping[s])
    wp.atomic_add(force, i, f)
    wp.atomic_sub(force, j, f)


class SpringMesh(SimObject):
    """Mass-spring network.

    springs:   [S, 2] node index pairs; rest lengths default to the initial lengths
    stiffness, damping: scalars or per-spring arrays
    mass:      total mass (spread evenly) or a per-node array
    triangles: optional closed outward surface used for collision; without it
               nodes collide as spheres of `radius`
    """

    def __init__(
        self,
        vertices,
        springs,
        stiffness=1e3,
        damping=1.0,
        mass=1.0,
        rest_length=None,
        triangles=None,
        radius=0.0,
        velocities=None,
        device=None,
    ):
        vertices = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
        springs = np.asarray(springs, dtype=np.int32).reshape(-1, 2)
        n, m = len(vertices), len(springs)

        mass = np.asarray(mass, dtype=np.float32)
        node_mass = np.full(n, mass / n, np.float32) if mass.ndim == 0 else mass
        if rest_length is None:
            rest_length = np.linalg.norm(vertices[springs[:, 1]] - vertices[springs[:, 0]], axis=1)

        if triangles is not None:
            shape = TriangleMesh(triangles, radius=radius)
        else:
            shape = Points(np.arange(n), radius)
        super().__init__(vertices, velocities, node_mass, shape, device)

        self.springs = to_int(springs, self.device, dtype=wp.vec2i)
        self.rest_length = to_float(np.broadcast_to(rest_length, m), self.device)
        self.stiffness = to_float(np.broadcast_to(stiffness, m), self.device)
        self.damping = to_float(np.broadcast_to(damping, m), self.device)

    def compute_internal_forces(self, force):
        wp.launch(
            spring_force_kernel,
            dim=self.springs.shape[0],
            inputs=[self.state.x, self.state.v, self.springs, self.rest_length, self.stiffness, self.damping],
            outputs=[force],
            device=self.device,
        )
