import numpy as np
import pytest

import deformkit as dk
from deformkit.utils.mesh import box_tet_mesh, rotate


def settle(world, steps):
    for _ in range(steps):
        world.step()


def contact_forces(world):
    """Contact forces alone (no gravity or internal forces) for the current state."""
    if not world.finalized:
        world.finalize()
    world.force.zero_()
    contacts = world.collision_system.detect(world, margin=world.contact_model.margin)
    world.contact_model.apply(world, contacts)
    return world.force.numpy().astype(np.float64), contacts.count()


def overlapping_cubes(order=(0, 1), seed=0):
    """Two cubes interpenetrating at a generic angle, with random velocities, and no obstacles."""
    va, ta = box_tet_mesh((0, 0, 0), (0.1, 0.1, 0.1), 3)
    vb, tb = box_tet_mesh((0, 0, 0), (0.08, 0.08, 0.08), 2)
    vb = rotate(vb, (1, 2, 3), 0.7) + np.array([0.04, 0.09, 0.03], np.float32)
    bodies = [(va, ta), (vb, tb)]
    world = dk.World(gravity=(0, 0, 0), contact_model=dk.PenaltyContact(stiffness=1e4, damping=5.0, thickness=2e-3))
    objs = {}
    for k in order:
        v, t = bodies[k]
        vel = 0.3 * np.random.default_rng(seed + k).standard_normal(v.shape)  # per body, not per slot
        objs[k] = world.add(dk.TetFEM(v, t, velocities=vel))
    return world, objs


def test_particle_rests_on_plane_at_penalty_equilibrium():
    k, m, g, r, thickness = 1e4, 0.2, 9.81, 0.05, 1e-3
    world = dk.World(dt=1e-3, contact_model=dk.PenaltyContact(stiffness=k, damping=5.0, thickness=thickness))
    p = world.add(dk.ParticleSystem([[0, 0.3, 0]], radius=r, mass=m))
    world.add_plane(normal=(0, 1, 0), offset=0.1)
    settle(world, 3000)
    # k d = m g with d = thickness - (y - offset - r)
    expected = 0.1 + r + thickness - m * g / k
    assert abs(p.state.x.numpy()[0, 1] - expected) < 1e-5
    assert abs(p.state.v.numpy()[0, 1]) < 1e-4


def test_tilted_plane_pushes_along_normal():
    world = dk.World(dt=1e-3, gravity=(0, 0, 0))
    world.add_plane(normal=(1, 1, 0), offset=0.0)
    p = world.add(dk.ParticleSystem([[-0.01, 0.0, 0]], radius=0.0))
    world.step()
    v = p.state.v.numpy()[0]
    assert v[0] > 0 and np.isclose(v[0], v[1]) and v[2] == 0


def test_particle_rests_on_sphere():
    world = dk.World(dt=1e-3, contact_model=dk.PenaltyContact(stiffness=1e4, damping=5.0))
    world.add_sphere(center=(0, 0, 0), radius=0.5)
    p = world.add(dk.ParticleSystem([[0, 1.0, 0]], radius=0.1, mass=0.1))
    settle(world, 3000)
    assert abs(np.linalg.norm(p.state.x.numpy()[0]) - 0.6) < 1e-3


def test_particles_without_obstacles_fall_freely():
    world = dk.World(dt=1e-3)
    p = world.add(dk.ParticleSystem([[0, 0, 0], [1, 0, 0]]))
    settle(world, 100)
    assert np.all(np.abs(p.state.v.numpy()[:, 1] + 0.981) < 1e-4)


def test_vertex_triangle_detection_inside_mesh():
    v, t = box_tet_mesh((0, 0, 0), (1, 1, 1), 2)
    world = dk.World(dt=1e-3, gravity=(0, 0, 0))
    world.add(dk.TetFEM(v, t))
    world.add(dk.ParticleSystem([[0.5, 0.95, 0.5], [0.5, 1.0005, 0.5], [0.5, 2.0, 0.5]], radius=0.0))
    world.finalize()
    c = world.collision_system.detect(world, margin=1e-3)
    n = c.count()
    assert n == 2  # the far particle is ignored
    gap = c.data.gap.numpy()[:n]
    normal = c.data.normal.numpy()[:n]
    vertex = c.data.vertex.numpy()[:n]
    by_vertex = dict(zip(vertex - world.objects[1].vertex_start, zip(gap, normal)))
    np.testing.assert_allclose(by_vertex[0][0], -0.05, atol=1e-5)  # inside: negative gap
    np.testing.assert_allclose(by_vertex[1][0], 0.0005, atol=1e-5)
    for _, nrm in by_vertex.values():
        np.testing.assert_allclose(nrm, [0, 1, 0], atol=1e-5)


def test_vertex_triangle_conserves_momentum():
    v, t = box_tet_mesh((0, 0, 0), (0.2, 0.2, 0.2), 2)
    world = dk.World(
        dt=1e-4, gravity=(0, 0, 0), contact_model=dk.PenaltyContact(stiffness=1e3, damping=1.0, thickness=1e-3)
    )
    world.add(dk.TetFEM(v, t, dk.StableNeoHookean(1e5, 0.3), density=500.0))
    ball = world.add(dk.ParticleSystem([[0.07, 0.3, 0.13]], radius=0.02, mass=0.2, velocities=[[0, -2.0, 0]]))
    world.finalize()
    m = world.state.mass.numpy()[:, None]
    p0 = (m * world.state.v.numpy()).sum(0)
    settle(world, 2000)
    p1 = (m * world.state.v.numpy()).sum(0)
    np.testing.assert_allclose(p1, p0, atol=1e-4)
    assert ball.state.v.numpy()[0, 1] > 0  # bounced off the cube


def test_particle_does_not_sink_into_soft_cube():
    v, t = box_tet_mesh((-0.1, 0, -0.1), (0.1, 0.2, 0.1), 3)
    world = dk.World(dt=2e-4, contact_model=dk.PenaltyContact(stiffness=2e4, damping=20.0))
    world.add_plane()
    cube = world.add(dk.TetFEM(v, t, dk.StableNeoHookean(1e5, 0.3), damping=20.0))
    ball = world.add(dk.ParticleSystem([[0.01, 0.3, 0.02]], radius=0.02, mass=0.05))
    settle(world, 5000)
    top = cube.state.x.numpy()[:, 1].max()
    y = ball.state.x.numpy()[0, 1]
    assert top - 0.01 < y - 0.02 < top + 0.002
    assert cube.state.x.numpy()[:, 1].min() > -2e-3


def test_soft_cube_rests_on_kinematic_cube():
    v, t = box_tet_mesh((-0.1, 0, -0.1), (0.1, 0.2, 0.1), 4)
    v2, t2 = box_tet_mesh((-0.06, 0.21, -0.06), (0.06, 0.33, 0.06), 3)
    world = dk.World(dt=2e-4, contact_model=dk.PenaltyContact(stiffness=2e4, damping=5.0))
    lower = world.add(dk.TetFEM(v, t))
    upper = world.add(dk.TetFEM(v2 + [0.013, 0, 0.021], t2, dk.StableNeoHookean(2e5, 0.3), damping=50.0))
    lower.state.mass.zero_()  # zero mass: the lower cube is kinematic and stays put
    settle(world, 10000)
    up = upper.state.x.numpy()
    assert 0.2 < up[:, 1].min() < 0.2 + 2e-3
    assert np.abs(upper.state.v.numpy()).max() < 0.05
    np.testing.assert_allclose(up.mean(0)[[0, 2]], [0.013, 0.021], atol=5e-3)


def test_soft_cubes_collide_head_on():
    v, t = box_tet_mesh((0, 0, 0), (0.1, 0.1, 0.1), 2)
    world = dk.World(dt=1e-4, gravity=(0, 0, 0), contact_model=dk.PenaltyContact(stiffness=1e4, damping=1.0))
    a = world.add(dk.TetFEM(v, t, dk.StableNeoHookean(1e5, 0.3), velocities=np.tile([1.0, 0, 0], (len(v), 1))))
    b = world.add(dk.TetFEM(v + [0.15, 0.02, 0.03], t, dk.StableNeoHookean(1e5, 0.3)))
    min_gap, num_contacts = np.inf, 0
    for _ in range(1500):
        world.step()
        n = world.contacts.count()
        if n:
            num_contacts = max(num_contacts, n)
            min_gap = min(min_gap, world.contacts.data.gap.numpy()[:n].min())
    assert num_contacts > 0
    assert min_gap > -5e-3  # penetration stays small: no tunneling
    va, vb = a.state.v.numpy().mean(0), b.state.v.numpy().mean(0)
    np.testing.assert_allclose(va + vb, [1.0, 0, 0], atol=1e-3)  # equal masses: momentum conserved
    assert vb[0] > va[0]  # separating


def test_mesh_contact_conserves_linear_and_angular_momentum():
    world, _ = overlapping_cubes()
    f, n = contact_forces(world)
    assert n >= 5
    x = world.state.x.numpy().astype(np.float64)
    scale = np.abs(f).sum()
    assert scale > 1.0
    np.testing.assert_allclose(f.sum(0), 0.0, atol=1e-5 * scale)
    # about the centroid, so the check does not depend on where the origin is
    torque = np.cross(x - x.mean(0), f).sum(0)
    np.testing.assert_allclose(torque, 0.0, atol=1e-5 * scale * 0.1)


def test_mesh_contact_is_symmetric_in_object_order():
    world_ab, objs_ab = overlapping_cubes(order=(0, 1))
    world_ba, objs_ba = overlapping_cubes(order=(1, 0))
    f_ab, n_ab = contact_forces(world_ab)
    f_ba, n_ba = contact_forces(world_ba)
    assert n_ab == n_ba
    for k in (0, 1):
        a, b = objs_ab[k], objs_ba[k]
        fa = f_ab[a.vertex_start : a.vertex_start + a.vertex_count]
        fb = f_ba[b.vertex_start : b.vertex_start + b.vertex_count]
        np.testing.assert_allclose(fa, fb, rtol=1e-4, atol=1e-4 * np.abs(fa).max())


def test_both_contact_directions_are_detected():
    world, objs = overlapping_cubes()
    _, n = contact_forces(world)
    c = world.contacts if world.contacts is not None else world.collision_system.contacts
    vertex = c.data.vertex.numpy()[:n]
    owner = world.object_ids.numpy()[vertex]
    assert set(owner) == {objs[0].object_id, objs[1].object_id}  # V_A -> F_B and V_B -> F_A


def test_distant_objects_have_no_contact_forces():
    v, t = box_tet_mesh((0, 0, 0), (0.1, 0.1, 0.1), 2)
    world = dk.World(gravity=(0, 0, 0))
    world.add(dk.TetFEM(v, t))
    world.add(dk.TetFEM(v + [1.0, 0.0, 0.0], t))
    world.add(dk.ParticleSystem([[0.0, 1.0, 0.0]]))
    f, n = contact_forces(world)
    assert n == 0
    assert np.all(f == 0.0)


@pytest.mark.parametrize("damping", [0.0, 5.0])
def test_penalty_damping_is_never_attractive(damping):
    def normal_force(vy):
        world = dk.World(gravity=(0, 0, 0), contact_model=dk.PenaltyContact(1e4, damping, thickness=1e-3))
        world.add_plane()
        world.add(dk.ParticleSystem([[0.0, 0.0, 0.0]], radius=0.0, velocities=[[0.0, vy, 0.0]]))
        return contact_forces(world)[0][0, 1]  # depth d = 1 mm

    static = normal_force(0.0)
    assert np.isclose(static, 1e4 * 1e-3)
    assert normal_force(-1.0) >= static  # approaching: damping adds repulsion
    assert 0.0 <= normal_force(0.5) <= static  # separating: damping only reduces it
    if damping > 0:
        assert normal_force(100.0) == 0.0  # fast separation: clamped, never pulls back


def test_contact_buffer_overflow_raises():
    world = dk.World()
    world.add_plane()
    world.add(dk.ParticleSystem(np.zeros((5, 3))))
    world.finalize()
    world.collision_system.contacts = dk.Contacts(capacity=2, device=world.device)
    with pytest.raises(RuntimeError, match="overflow"):
        world.collision_system.detect(world, margin=1e-3).count()
