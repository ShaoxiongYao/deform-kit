import numpy as np
import pytest

import deformkit as dk
from deformkit.utils.mesh import box_tet_mesh


def two_object_world():
    v, t = box_tet_mesh((0, 0, 0), (1, 1, 1), 1)
    world = dk.World(gravity=(0, 0, 0))
    a = world.add(dk.TetFEM(v, t))
    b = world.add(dk.ParticleSystem([[5.0, 0, 0], [6.0, 0, 0]]))
    world.finalize()
    return world, a, b


def test_objects_occupy_consecutive_ranges():
    world, a, b = two_object_world()
    assert (a.vertex_start, b.vertex_start) == (0, a.vertex_count)
    assert world.num_nodes == a.vertex_count + b.vertex_count
    np.testing.assert_array_equal(world.object_ids.numpy(), [0] * a.vertex_count + [1] * b.vertex_count)


def test_object_state_aliases_global_arrays():
    world, a, b = two_object_world()
    for name in ("x", "v", "mass"):
        glob, local = getattr(world.state, name), getattr(b.state, name)
        assert local.ptr == glob.ptr + b.vertex_start * glob.strides[0]  # same storage, not a copy

    # writes through either side are seen by the other
    x = world.state.x.numpy()
    x[b.vertex_start] = [7.0, 8.0, 9.0]
    world.state.x.assign(x)
    np.testing.assert_array_equal(b.state.x.numpy()[0], [7.0, 8.0, 9.0])

    b.state.v.fill_(2.0)
    np.testing.assert_array_equal(world.state.v.numpy()[b.vertex_start :], 2.0)
    assert np.all(world.state.v.numpy()[: b.vertex_start] == 0.0)


def test_initial_object_state_survives_packing():
    v, t = box_tet_mesh((0, 0, 0), (1, 1, 1), 1)
    world = dk.World()
    a = world.add(dk.TetFEM(v, t))
    vel = np.tile([1.0, 2.0, 3.0], (a.vertex_count, 1)).astype(np.float32)
    a.state.v.assign(vel)  # edited before finalize, on the object's own arrays
    world.finalize()
    np.testing.assert_array_equal(world.state.v.numpy(), vel)
    np.testing.assert_allclose(world.state.x.numpy(), v)


@pytest.mark.parametrize(
    "modify",
    [
        lambda w: w.add(dk.ParticleSystem([[0.0, 0.0, 0.0]])),
        lambda w: w.add_plane(),
        lambda w: w.add_sphere((0, 0, 0), 1.0),
        lambda w: w.finalize(),
    ],
)
def test_scene_is_frozen_after_finalize(modify):
    world = dk.World()
    world.add(dk.ParticleSystem([[0.0, 1.0, 0.0]]))
    world.step()  # finalizes implicitly
    with pytest.raises(RuntimeError, match="finalize"):
        modify(world)


def test_pinned_nodes_stay_put_under_gravity_and_contact():
    v, t = box_tet_mesh((0, 0.05, 0), (0.1, 0.15, 0.1), 2)
    world = dk.World(dt=2e-4)
    world.add_plane()
    cube = world.add(dk.TetFEM(v, t))
    top = np.where(np.isclose(v[:, 1], 0.15))[0]
    mass = cube.state.mass.numpy()
    mass[top] = 0.0
    cube.state.mass.assign(mass)
    for _ in range(3000):
        world.step()
    np.testing.assert_array_equal(cube.state.x.numpy()[top], v[top])
    assert cube.state.x.numpy()[:, 1].min() < 0.05  # the rest of the cube sagged
