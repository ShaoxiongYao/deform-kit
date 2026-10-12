import numpy as np
import pytest

import deformkit as dk
from deformkit.utils.mesh import box_tet_mesh, remove_unused_vertices, rotate


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
    rec = c.numpy()
    vertex = rec["nodes"][:, 0]
    by_vertex = dict(zip(vertex - world.objects[1].vertex_start, zip(rec["gap"], rec["normal"])))
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
    contact_forces(world)
    rec = world.collision_system.contacts.numpy()
    vertex_triangle = (rec["weights"][:, 0] == 1.0) & (rec["nodes"][:, 1] >= 0)
    owner = world.object_ids.numpy()[rec["nodes"][vertex_triangle, 0]]
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


def test_contact_buffer_grows_instead_of_dropping():
    world = dk.World()
    world.add_plane()
    world.add(dk.ParticleSystem(np.zeros((5, 3))))
    world.finalize()
    world.collision_system.contacts = dk.Contacts(capacity=2, device=world.device)
    c = world.collision_system.detect(world, margin=1e-3)
    assert c.capacity >= 5
    assert c.count() == 5


def test_contact_count_raises_if_records_were_dropped():
    c = dk.Contacts(capacity=2, device=dk.World().device)
    c.data.count.fill_(3)
    with pytest.raises(RuntimeError, match="overflow"):
        c.count()


def crossed_bars(gap, edge_edge=True, velocity=0.0):
    """Two square bars at right angles, turned 45deg so only an edge of each faces the other.

    Their nodes sit 3 cm away from the crossing point, so vertex-triangle
    contact cannot see the crossing until the bars are deep inside each other.
    """
    half = 0.03 * np.sqrt(2)  # half-diagonal of the 6 cm square section
    va, ta = box_tet_mesh((-0.3, -0.03, -0.03), (0.3, 0.03, 0.03), (10, 1, 1))
    va = rotate(va, (1, 0, 0), np.pi / 4, center=(0, 0, 0))  # top edge along x at y = +half
    vb, tb = box_tet_mesh((-0.03, -0.03, -0.3), (0.03, 0.03, 0.3), (1, 1, 10))
    vb = rotate(vb, (0, 0, 1), np.pi / 4, center=(0, 0, 0))  # bottom edge along z at y = -half
    vb = vb + np.array([0.03, 2 * half + gap, 0.03], np.float32)

    world = dk.World(
        dt=1e-4,
        gravity=(0, 0, 0),
        contact_model=dk.PenaltyContact(stiffness=2e4, damping=2.0, thickness=2e-3),
        collision_system=dk.CollisionSystem(edge_edge=edge_edge),
    )
    material = dk.StableNeoHookean(1e6, 0.3)
    a = world.add(dk.TetFEM(va, ta, material, density=500))
    b = world.add(dk.TetFEM(vb, tb, material, density=500, velocities=np.tile([0, -velocity, 0], (len(vb), 1))))
    # the two mesh edges that cross, at (x, z) = (0.03, 0): A's from x = 0 to 0.06, B's from z = -0.03 to 0.03
    top_a = np.isclose(va[:, 1], va[:, 1].max(), atol=1e-6)
    bottom_b = np.isclose(vb[:, 1], vb[:, 1].min(), atol=1e-6)
    edge_a = np.where(top_a & (np.isclose(va[:, 0], 0.0, atol=1e-6) | np.isclose(va[:, 0], 0.06, atol=1e-6)))[0]
    edge_b = np.where(bottom_b & (np.isclose(np.abs(vb[:, 2]), 0.03, atol=1e-6)))[0]
    assert len(edge_a) == 2 and len(edge_b) == 2
    return world, a, b, edge_a, edge_b


def test_edge_edge_detection():
    world, a, b, _, _ = crossed_bars(gap=5e-4)
    world.finalize()
    rec = world.collision_system.detect(world, margin=2e-3).numpy()
    assert len(rec["gap"]) == 1  # exactly one record for the crossing pair, not one per side
    np.testing.assert_allclose(rec["gap"][0], 5e-4, atol=1e-5)
    np.testing.assert_allclose(np.abs(rec["normal"][0]), [0, 1, 0], atol=1e-4)
    np.testing.assert_allclose(rec["weights"][0].sum(), 0.0, atol=1e-6)

    world, *_ = crossed_bars(gap=5e-4, edge_edge=False)
    world.finalize()
    assert world.collision_system.detect(world, margin=2e-3).count() == 0  # invisible to vertex-triangle


def test_edge_edge_detection_after_crossing():
    world, a, b, _, _ = crossed_bars(gap=-5e-4)  # edges have passed through each other
    world.finalize()
    rec = world.collision_system.detect(world, margin=2e-3).numpy()
    assert len(rec["gap"]) == 1
    np.testing.assert_allclose(rec["gap"][0], -5e-4, atol=1e-5)  # signed: overlapping
    # the force pushes side A's edge back out of B, whichever bar side A is
    side_a = world.object_ids.numpy()[rec["nodes"][0, 0]]
    expected = [0, -1, 0] if side_a == a.object_id else [0, 1, 0]
    np.testing.assert_allclose(rec["normal"][0], expected, atol=1e-4)


@pytest.mark.parametrize("edge_edge", [True, False])
def test_edge_edge_stops_crossing_bars(edge_edge):
    # slow enough that one penalty contact stops the bar within its 2 mm thickness;
    # edge-edge only acts within that band, so faster crossings would tunnel
    world, a, b, edge_a, edge_b = crossed_bars(gap=0.01, edge_edge=edge_edge, velocity=0.3)
    separation = np.inf  # height of B's edge above A's edge at the crossing (both edge midpoints)
    for _ in range(1000):
        world.step()
        y_a = a.state.x.numpy()[edge_a, 1].mean()
        y_b = b.state.x.numpy()[edge_b, 1].mean()
        separation = min(separation, y_b - y_a)
    if edge_edge:
        assert separation > 0.0  # held apart within the contact thickness
    else:
        assert separation < -0.01  # the edges pass straight through each other


def u_shape(self_contact):
    """A soft U (two 3 cm arms on a base, 6 cm apart) with its arms thrown toward each other."""
    res = (4, 4, 1)
    v, t = box_tet_mesh((0, 0, 0), (0.12, 0.12, 0.03), res)
    cell = np.repeat(np.arange(res[0] * res[1]), 5)  # 5 tets per cell, cells ordered x-fastest
    i, j = cell % res[0], cell // res[0]
    keep = ~((i >= 1) & (i <= 2) & (j >= 1))  # carve the slot between the arms
    v, t = remove_unused_vertices(v, t[keep])

    arm = v[:, 1] > 0.035
    left, right = arm & (v[:, 0] < 0.035), arm & (v[:, 0] > 0.085)
    vel = np.zeros_like(v)
    vel[left, 0], vel[right, 0] = 1.5, -1.5
    world = dk.World(
        dt=1e-4,
        gravity=(0, 0, 0),
        contact_model=dk.PenaltyContact(stiffness=1e4, damping=1.0, thickness=2e-3),
        collision_system=dk.CollisionSystem(self_contact=self_contact),
    )
    body = world.add(dk.TetFEM(v, t, dk.StableNeoHookean(5e4, 0.3), density=500, velocities=vel))
    # inner faces of the slot: left arm's right face and right arm's left face, above the base
    inner_left = np.where(arm & np.isclose(v[:, 0], 0.03))[0]
    inner_right = np.where(arm & np.isclose(v[:, 0], 0.09))[0]
    return world, body, inner_left, inner_right


@pytest.mark.parametrize("self_contact", [True, False])
def test_self_contact_keeps_u_arms_apart(self_contact):
    world, body, inner_left, inner_right = u_shape(self_contact)
    gap = np.inf
    for _ in range(1500):
        world.step()
        x = body.state.x.numpy()
        gap = min(gap, x[inner_right, 0].min() - x[inner_left, 0].max())
    if self_contact:
        assert gap > 0.0
    else:
        assert gap < -0.005  # the arms pass through each other


def test_self_contact_has_no_false_positives_at_rest():
    v, t = box_tet_mesh((0, 0, 0), (0.1, 0.1, 0.1), 4)
    for obj in (dk.TetFEM(v, t), dk.SpringMesh(v, [[0, 1]], triangles=dk.TetFEM(v, t).collision_shape.triangles)):
        world = dk.World(collision_system=dk.CollisionSystem(self_contact=True))
        world.add(obj)
        world.finalize()
        assert world.collision_system.detect(world, margin=2e-3).count() == 0
