"""A small soft cube lands on a larger one, tips over its edge, and falls to the ground.

Contact between the cubes is vertex-triangle penalty contact in both directions.
There is no friction, so nothing holds the upper cube in place.

    uv run examples/two_soft_objects.py [--viewer] [--usd out.usd]
"""

import argparse

import numpy as np

import deformkit as dk
from deformkit.utils.mesh import box_tet_mesh, rotate
from deformkit.utils.render import run

parser = argparse.ArgumentParser()
parser.add_argument("--viewer", action="store_true", help="open an OpenGL window")
parser.add_argument("--usd", help="record to a USD file")
parser.add_argument("--frames", type=int, default=180)
args = parser.parse_args()

world = dk.World(dt=2e-4, contact_model=dk.PenaltyContact(stiffness=2e4, damping=5.0, thickness=2e-3))
world.add_plane(normal=(0, 1, 0), offset=0.0)

material = dk.StableNeoHookean(youngs_modulus=2e5, poisson_ratio=0.4)
big_v, big_t = box_tet_mesh((-0.15, 0.0, -0.15), (0.15, 0.2, 0.15), (6, 4, 6))
small_v, small_t = box_tet_mesh((0.02, 0.35, -0.05), (0.14, 0.47, 0.07), 4)
small_v = rotate(small_v, axis=(1, 0, 1), angle=0.5)

big = world.add(dk.TetFEM(big_v, big_t, material, density=500, damping=50.0))
small = world.add(dk.TetFEM(small_v, small_t, material, density=500, damping=50.0))

min_gap = np.inf


def report(world, frame):
    global min_gap
    c = world.contacts
    n = c.count()
    between = c.data.other.numpy()[:n, 0] >= 0  # contacts against a mesh, not an obstacle
    if between.any():
        min_gap = min(min_gap, c.data.gap.numpy()[:n][between].min())
    if frame % 30 == 29:
        com = small.state.x.numpy().mean(0)
        print(f"t={world.time:5.2f}s  small cube com={com.round(3)}  object-object contacts={int(between.sum())}")


run(world, args.frames, steps_per_frame=83, viewer=args.viewer, usd=args.usd, report=report)
print(f"closest approach between the cubes: {1000 * min_gap:.2f} mm (negative = overlap)")
print(f"big cube bottom: {np.min(big.state.x.numpy()[:, 1]):.4f}")
