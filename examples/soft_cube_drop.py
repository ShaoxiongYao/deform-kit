"""A tilted Neo-Hookean FEM cube falls onto the ground.

Tetrahedra drive the elasticity; boundary triangles are the collision mesh.

    uv run examples/soft_cube_drop.py [--viewer] [--usd out.usd]
"""

import argparse

import deformkit as dk
from deformkit.utils.mesh import box_tet_mesh, rotate
from deformkit.utils.render import run

parser = argparse.ArgumentParser()
parser.add_argument("--viewer", action="store_true", help="open an OpenGL window")
parser.add_argument("--usd", help="record to a USD file")
parser.add_argument("--frames", type=int, default=180)
args = parser.parse_args()

world = dk.World(dt=2e-4, contact_model=dk.PenaltyContact(stiffness=2e4, damping=10.0, thickness=2e-3))
world.add_plane(normal=(0, 1, 0), offset=0.0)

vertices, tets = box_tet_mesh((-0.1, 0.3, -0.1), (0.1, 0.5, 0.1), 5)
vertices = rotate(vertices, axis=(1, 0, 0.4), angle=0.6)
cube = world.add(
    dk.TetFEM(
        vertices, tets, material=dk.StableNeoHookean(youngs_modulus=1e5, poisson_ratio=0.4), density=500, damping=100.0
    )
)


def report(world, frame):
    if frame % 30 == 29:
        x = cube.state.x.numpy()
        com, min_y = x.mean(0).round(3), x[:, 1].min()
        print(f"t={world.time:5.2f}s  com={com}  min y={min_y:.4f}  contacts={world.contacts.count()}")


run(world, args.frames, steps_per_frame=83, viewer=args.viewer, usd=args.usd, report=report)
