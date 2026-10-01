import functools

import numpy as np
import warp as wp

from deformkit.collision.geometry import TriangleMesh
from deformkit.core.backend import to_float, to_int
from deformkit.core.object import SimObject
from deformkit.materials.elasticity import StableNeoHookean
from deformkit.utils.mesh import boundary_triangles, orient_tets, tet_signed_volumes


@functools.cache
def fem_force_kernel(stress):
    """Build the element force kernel for one material stress function."""

    @wp.kernel(module="unique")
    def kernel(
        x: wp.array(dtype=wp.vec3),
        v: wp.array(dtype=wp.vec3),
        tets: wp.array(dtype=wp.vec4i),
        Dm_inv: wp.array(dtype=wp.mat33),
        rest_volume: wp.array(dtype=wp.float32),
        mu: float,
        lam: float,
        damping: float,
        force: wp.array(dtype=wp.vec3),
    ):
        e = wp.tid()
        t = tets[e]
        x0, v0 = x[t[0]], v[t[0]]
        Ds = wp.matrix_from_cols(x[t[1]] - x0, x[t[2]] - x0, x[t[3]] - x0)
        Ds_dot = wp.matrix_from_cols(v[t[1]] - v0, v[t[2]] - v0, v[t[3]] - v0)
        B = Dm_inv[e]
        F = Ds * B

        P = stress(F, mu, lam)
        if damping > 0.0:
            # viscous stress from the Green strain rate; zero for rigid motion
            F_dot = Ds_dot * B
            E_dot = 0.5 * (wp.transpose(F) * F_dot + wp.transpose(F_dot) * F)
            P = P + damping * F * E_dot

        # forces on nodes 1..3 are the columns of -V P Dm^-T; node 0 balances them
        H = wp.transpose(-rest_volume[e] * P * wp.transpose(B))
        f1, f2, f3 = H[0], H[1], H[2]
        wp.atomic_add(force, t[0], -(f1 + f2 + f3))
        wp.atomic_add(force, t[1], f1)
        wp.atomic_add(force, t[2], f2)
        wp.atomic_add(force, t[3], f3)

    return kernel


class TetFEM(SimObject):
    """Linear tetrahedral FEM with lumped mass.

    Tets drive the elasticity; the boundary triangles are extracted once and
    used as the collision mesh.

    damping: viscosity (Pa s) on the Green strain rate
    """

    def __init__(
        self,
        vertices,
        tets,
        material=None,
        density=1000.0,
        damping=0.0,
        velocities=None,
        radius=0.0,
        device=None,
    ):
        vertices = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
        tets = orient_tets(vertices, tets)
        self.material = material or StableNeoHookean()
        self.damping = damping

        volume = tet_signed_volumes(vertices, tets)
        if np.any(volume <= 0):
            raise ValueError("degenerate tetrahedra in rest mesh")
        mass = np.zeros(len(vertices), np.float32)
        np.add.at(mass, tets.ravel(), np.repeat(density * volume / 4.0, 4))

        p = vertices[tets].astype(np.float64)
        Dm = np.stack([p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]], axis=2)

        super().__init__(vertices, velocities, mass, TriangleMesh(boundary_triangles(tets), radius=radius), device)

        self.tets = to_int(tets, self.device, dtype=wp.vec4i)
        self.Dm_inv = wp.array(np.linalg.inv(Dm).astype(np.float32), dtype=wp.mat33, device=self.device)
        self.rest_volume = to_float(volume, self.device)

    def compute_internal_forces(self, force):
        wp.launch(
            fem_force_kernel(self.material.stress),
            dim=self.tets.shape[0],
            inputs=[
                self.state.x,
                self.state.v,
                self.tets,
                self.Dm_inv,
                self.rest_volume,
                self.material.mu,
                self.material.lam,
                self.damping,
            ],
            outputs=[force],
            device=self.device,
        )
