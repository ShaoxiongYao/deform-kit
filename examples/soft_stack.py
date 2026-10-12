"""Three soft cubes dropped into an off-center tower; friction holds the stack together.

Compare with two_soft_objects.py (frictionless), where the upper cube slides off.

    uv run examples/soft_stack.py [--viewer] [--usd out.usd] [--friction 0.6]
"""

import argparse

import numpy as np

import deformkit as dk
from deformkit.utils.mesh import box_tet_mesh, rotate
from deformkit.utils.render import run

parser = argparse.ArgumentParser()
parser.add_argument("--viewer", action="store_true", help="open an OpenGL window")
parser.add_argument("--usd", help="record to a USD file")
parser.add_argument("--frames", type=int, default=240)
parser.add_argument("--friction", type=float, default=0.6)
args = parser.parse_args()

world = dk.World(
    dt=2e-4,
    contact_model=dk.PenaltyContact(stiffness=2e4, damping=5.0, thickness=2e-3, friction=args.friction),
)
world.add_plane(normal=(0, 1, 0), offset=0.0)

material = dk.StableNeoHookean(youngs_modulus=2e5, poisson_ratio=0.4)
cubes = []
for size, center, angle in [(0.24, (0.0, 0.0), 0.0), (0.16, (0.04, 0.03), 0.3), (0.10, (0.06, 0.0), -0.5)]:
    base = sum(c.state.x.numpy()[:, 1].max() for c in cubes[-1:]) + 0.02
    lo = (center[0] - size / 2, base, center[1] - size / 2)
    hi = (center[0] + size / 2, base + size, center[1] + size / 2)
    v, t = box_tet_mesh(lo, hi, 4)
    v = rotate(v, axis=(0, 1, 0), angle=angle)
    cubes.append(world.add(dk.TetFEM(v, t, material, density=500, damping=50.0)))


def report(world, frame):
    if frame % 60 == 59:
        coms = [c.state.x.numpy().mean(0) for c in cubes]
        offsets = [np.linalg.norm((coms[i + 1] - coms[i])[[0, 2]]) for i in range(len(coms) - 1)]
        print(f"t={world.time:5.2f}s  horizontal offsets between neighbours: {np.round(offsets, 4)} m")


run(world, args.frames, steps_per_frame=83, viewer=args.viewer, usd=args.usd, report=report)
