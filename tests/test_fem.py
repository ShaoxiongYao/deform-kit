import numpy as np
import pytest
import warp as wp

import deformkit as dk
from deformkit.utils.mesh import boundary_triangles, box_tet_mesh, tet_signed_volumes


def neo_hookean_energy(x, tets, Dm_inv, volume, mu, lam):
    p = x[tets].astype(np.float64)
    Ds = np.stack([p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]], axis=2)
    F = Ds @ Dm_inv
    Ic = np.einsum("eij,eij->e", F, F)
    J = np.linalg.det(F)
    psi = 0.5 * mu * (Ic - 3.0) - mu * (J - 1.0) + 0.5 * lam * (J - 1.0) ** 2
    return np.sum(volume * psi)


def internal_forces(body):
    force = wp.zeros(body.vertex_count, dtype=wp.vec3, device=body.device)
    body.compute_internal_forces(force)
    return force.numpy().astype(np.float64)


def make_body(x_rest, tets, x_deformed=None, damping=0.0):
    body = dk.TetFEM(x_rest, tets, dk.StableNeoHookean(1e4, 0.3), damping=damping)
    if x_deformed is not None:
        body.state.x.assign(x_deformed.astype(np.float32))
    return body


@pytest.fixture
def cube():
    return box_tet_mesh((0, 0, 0), (1, 1, 1), 2)


def test_box_mesh_is_valid(cube):
    v, t = cube
    assert np.all(tet_signed_volumes(v, t) > 0)
    assert np.isclose(tet_signed_volumes(v, t).sum(), 1.0)
    tris = boundary_triangles(t)
    assert len(tris) == 6 * 2 * 2 * 2  # 6 faces x (2x2 cells) x 2 triangles
    # outward winding: divergence theorem gives the enclosed volume
    p = v[tris]
    assert np.isclose(np.einsum("ij,ij->", p[:, 0], np.cross(p[:, 1], p[:, 2])) / 6.0, 1.0, atol=1e-5)


def test_rest_state_has_zero_force(cube):
    v, t = cube
    assert np.abs(internal_forces(make_body(v, t))).max() < 1e-3


def test_rigid_rotation_has_zero_force(cube):
    v, t = cube
    angle = 0.7
    R = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    body = make_body(v, t, v @ R.T + np.array([1.0, 2.0, 3.0]))
    assert np.abs(internal_forces(body)).max() < 1e-2  # vs ~1e3 N for a comparable strain


def test_forces_match_energy_gradient(cube):
    v, t = cube
    rng = np.random.default_rng(0)
    x = v + 0.1 * rng.standard_normal(v.shape)
    body = make_body(v, t, x)
    f = internal_forces(body)

    t_np = body.tets.numpy()
    Dm_inv = body.Dm_inv.numpy().astype(np.float64)
    vol = body.rest_volume.numpy().astype(np.float64)
    mu, lam = body.material.mu, body.material.lam

    eps = 1e-6
    grad = np.zeros_like(x)
    for i in range(len(x)):
        for d in range(3):
            xp, xm = x.copy(), x.copy()
            xp[i, d] += eps
            xm[i, d] -= eps
            e_p = neo_hookean_energy(xp, t_np, Dm_inv, vol, mu, lam)
            e_m = neo_hookean_energy(xm, t_np, Dm_inv, vol, mu, lam)
            grad[i, d] = (e_p - e_m) / (2 * eps)
    np.testing.assert_allclose(f, -grad, rtol=1e-3, atol=1e-3 * np.abs(grad).max())


def test_forces_sum_to_zero(cube):
    v, t = cube
    rng = np.random.default_rng(1)
    f = internal_forces(make_body(v, t, v + 0.1 * rng.standard_normal(v.shape)))
    assert np.abs(f.sum(axis=0)).max() < 1e-2 * np.abs(f).max()


def test_damping_ignores_rigid_rotation(cube):
    v, t = cube
    body = make_body(v, t, damping=10.0)
    omega = np.array([0.0, 0.0, 2.0])
    body.state.v.assign(np.cross(omega, v - v.mean(axis=0)).astype(np.float32))
    assert np.abs(internal_forces(body)).max() < 1e-3


def test_lumped_mass(cube):
    v, t = cube
    body = dk.TetFEM(v, t, density=500.0)
    assert np.isclose(body.state.mass.numpy().sum(), 500.0)
