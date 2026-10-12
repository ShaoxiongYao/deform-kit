import numpy as np
import pytest
from test_contact import contact_forces, overlapping_cubes

import deformkit as dk
from deformkit.utils.mesh import box_tet_mesh

G = 9.81


def particle_on_incline(angle, mu, k=1e4, m=0.1, thickness=1e-3, dt=1e-3, velocity=None):
    """A particle resting at penalty equilibrium on a plane through the origin tilted by `angle`."""
    n = np.array([-np.sin(angle), np.cos(angle), 0.0])
    world = dk.World(dt=dt, contact_model=dk.PenaltyContact(stiffness=k, damping=5.0, thickness=thickness, friction=mu))
    world.add_plane(normal=n, offset=0.0)
    x0 = n * (thickness - m * G * np.cos(angle) / k)
    p = world.add(dk.ParticleSystem([x0], radius=0.0, mass=m, velocities=None if velocity is None else [velocity]))
    return world, p, n


def test_particle_sticks_below_friction_angle():
    world, p, _ = particle_on_incline(np.radians(20), mu=0.5)  # tan 20deg = 0.36 < 0.5
    x0 = p.state.x.numpy()[0].copy()
    for _ in range(2000):
        world.step()
    assert np.linalg.norm(p.state.x.numpy()[0] - x0) < 1e-5


@pytest.mark.parametrize("mu", [0.0, 0.1, 0.3])
def test_particle_slides_with_coulomb_acceleration(mu):
    angle = np.radians(30)
    world, p, n = particle_on_incline(angle, mu)
    down = np.array([-np.cos(angle), -np.sin(angle), 0.0])  # along the slope
    t = 1.0
    for _ in range(int(t / world.dt)):
        world.step()
    speed = p.state.v.numpy()[0] @ down
    expected = G * (np.sin(angle) - mu * np.cos(angle)) * t
    assert abs(speed - expected) < 0.01 * expected


def test_sliding_particle_stops_at_coulomb_distance():
    v0, mu = 1.0, 0.5
    world, p, _ = particle_on_incline(0.0, mu, velocity=[v0, 0.0, 0.0])
    for _ in range(500):
        world.step()
    assert abs(p.state.v.numpy()[0, 0]) < 1e-6  # static friction holds it
    distance = p.state.x.numpy()[0, 0]
    assert abs(distance - v0**2 / (2 * mu * G)) < 0.02 * distance


def test_friction_conserves_momentum_between_objects():
    world, _ = overlapping_cubes()
    world.contact_model.friction = 0.8
    f, n = contact_forces(world)
    assert n > 0
    scale = np.abs(f).sum()
    np.testing.assert_allclose(f.sum(0), 0.0, atol=1e-5 * scale)


@pytest.mark.parametrize("mu", [0.0, 0.6])
def test_friction_holds_soft_cube_off_center_on_soft_cube(mu):
    """The v0.1 failure case: without friction the upper cube slides off; with it, it stays."""
    v, t = box_tet_mesh((-0.1, 0, -0.1), (0.1, 0.2, 0.1), 4)
    v2, t2 = box_tet_mesh((-0.06, 0.21, -0.06), (0.06, 0.33, 0.06), 3)
    world = dk.World(dt=2e-4, contact_model=dk.PenaltyContact(stiffness=2e4, damping=5.0, friction=mu))
    world.add_plane()
    lower = world.add(dk.TetFEM(v, t, dk.StableNeoHookean(2e5, 0.3), damping=50.0))
    upper = world.add(dk.TetFEM(v2 + [0.03, 0.0, 0.04], t2, dk.StableNeoHookean(2e5, 0.3), damping=50.0))

    def offset():
        return upper.state.x.numpy().mean(0) - lower.state.x.numpy().mean(0)

    for _ in range(5000):  # land and settle
        world.step()
    landed = offset()
    for _ in range(10000):
        world.step()
    drift = np.linalg.norm((offset() - landed)[[0, 2]])
    if mu > 0:
        assert drift < 2e-3  # residual vibration only; frictionless drifts > 5 cm
        assert abs(upper.state.x.numpy()[:, 1].min() - lower.state.x.numpy()[:, 1].max()) < 3e-3
    else:
        assert drift > 0.05
