import numpy as np
import warp as wp

import deformkit as dk
from deformkit.utils.mesh import box_tet_mesh, tet_edges


def spring_forces(obj):
    force = wp.zeros(obj.vertex_count, dtype=wp.vec3, device=obj.device)
    obj.compute_internal_forces(force)
    return force.numpy()


def test_spring_force_hookean():
    x = np.array([[0, 0, 0], [1, 0, 0]], np.float32)
    s = dk.SpringMesh(x, [[0, 1]], stiffness=100.0, damping=0.0, rest_length=0.5)
    f = spring_forces(s)
    np.testing.assert_allclose(f[0], [50.0, 0, 0], rtol=1e-6)  # pulled toward node 1
    np.testing.assert_allclose(f[1], [-50.0, 0, 0], rtol=1e-6)


def test_spring_damping_opposes_stretch_rate():
    x = np.array([[0, 0, 0], [1, 0, 0]], np.float32)
    v = np.array([[0, 0, 0], [2, 0, 0]], np.float32)
    s = dk.SpringMesh(x, [[0, 1]], stiffness=0.0, damping=3.0, velocities=v)
    np.testing.assert_allclose(spring_forces(s)[1], [-6.0, 0, 0], rtol=1e-6)


def test_oscillation_frequency():
    k, m, dt = 100.0, 1.0, 1e-4
    x = np.array([[0, 0, 0], [1.1, 0, 0]], np.float32)  # stretched by 0.1
    world = dk.World(dt=dt, gravity=(0, 0, 0))
    s = world.add(dk.SpringMesh(x, [[0, 1]], stiffness=k, damping=0.0, mass=2 * m, rest_length=1.0))

    # reduced mass m/2 -> omega = sqrt(2k/m); first return to the start after one period
    period = 2 * np.pi / np.sqrt(2 * k / m)
    lengths = []
    for _ in range(int(1.5 * period / dt)):
        world.step()
        p = s.state.x.numpy()
        lengths.append(np.linalg.norm(p[1] - p[0]))
    lengths = np.array(lengths)
    half = len(lengths) // 3  # skip the first minimum at T/2
    t_peak = (half + np.argmax(lengths[half:]) + 1) * dt
    assert abs(t_peak - period) < 0.01 * period
    assert abs(lengths.max() - 1.1) < 1e-3  # symplectic Euler does not gain energy


def test_hanging_mass_with_kinematic_anchor():
    k, m, g = 200.0, 0.5, 9.81
    x = np.array([[0, 1, 0], [0, 0.5, 0]], np.float32)
    world = dk.World(dt=1e-3, gravity=(0, -g, 0))
    s = world.add(dk.SpringMesh(x, [[0, 1]], stiffness=k, damping=2.0, mass=np.array([0.0, m])))
    for _ in range(5000):
        world.step()
    p = s.state.x.numpy()
    np.testing.assert_allclose(p[0], [0, 1, 0], atol=1e-6)  # zero mass: pinned
    assert abs((p[0, 1] - p[1, 1]) - (0.5 + m * g / k)) < 1e-3


def test_spring_cube_keeps_shape_and_momentum():
    v, t = box_tet_mesh((0, 0, 0), (1, 1, 1), 2)
    world = dk.World(dt=1e-3, gravity=(0, 0, 0))
    vel = np.tile([1.0, 0.0, 0.0], (len(v), 1))
    s = world.add(dk.SpringMesh(v, tet_edges(t), stiffness=1e3, damping=1.0, velocities=vel))
    for _ in range(500):
        world.step()
    x = s.state.x.numpy()
    np.testing.assert_allclose(x - x.mean(0), v - v.mean(0), atol=1e-4)
    np.testing.assert_allclose(s.state.v.numpy().mean(0), [1, 0, 0], atol=1e-5)
