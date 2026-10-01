"""A tilted mass-spring cube falls corner-first onto the ground.

Springs follow the edges of a tet mesh; the boundary triangles collide.

    uv run examples/spring_cube_drop.py [--viewer] [--usd out.usd]
"""

import argparse

import deformkit as dk
from deformkit.utils.mesh import boundary_triangles, box_tet_mesh, rotate, tet_edges
from deformkit.utils.render import run

parser = argparse.ArgumentParser()
parser.add_argument("--viewer", action="store_true", help="open an OpenGL window")
parser.add_argument("--usd", help="record to a USD file")
parser.add_argument("--frames", type=int, default=180)
args = parser.parse_args()

world = dk.World(dt=2e-4, contact_model=dk.PenaltyContact(stiffness=1e4, damping=10.0, thickness=2e-3))
world.add_plane(normal=(0, 1, 0), offset=0.0)

vertices, tets = box_tet_mesh((-0.1, 0.3, -0.1), (0.1, 0.5, 0.1), 5)
vertices = rotate(vertices, axis=(1, 0, 1), angle=0.6)
cube = world.add(
    dk.SpringMesh(
        vertices,
        tet_edges(tets),
        stiffness=2e3,
        damping=1.0,
        mass=1.0,
        triangles=boundary_triangles(tets),
    )
)


def report(world, frame):
    if frame % 30 == 29:
        x = cube.state.x.numpy()
        com, min_y = x.mean(0).round(3), x[:, 1].min()
        print(f"t={world.time:5.2f}s  com={com}  min y={min_y:.4f}  contacts={world.contacts.count()}")


run(world, args.frames, steps_per_frame=83, viewer=args.viewer, usd=args.usd, report=report)
