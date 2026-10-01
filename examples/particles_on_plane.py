"""A block of particles rains onto a sphere and a ground plane.

Particles do not collide with each other (no self-contact), only with obstacles.

    uv run examples/particles_on_plane.py [--viewer] [--usd out.usd]
"""

import argparse

import numpy as np

import deformkit as dk
from deformkit.utils.mesh import grid_points
from deformkit.utils.render import run

parser = argparse.ArgumentParser()
parser.add_argument("--viewer", action="store_true", help="open an OpenGL window")
parser.add_argument("--usd", help="record to a USD file")
parser.add_argument("--frames", type=int, default=180)
args = parser.parse_args()

world = dk.World(dt=1e-3, contact_model=dk.PenaltyContact(stiffness=1e4, damping=2.0, thickness=1e-3))
world.add_plane(normal=(0, 1, 0), offset=0.0)
world.add_sphere(center=(0.0, 0.25, 0.0), radius=0.25)

rng = np.random.default_rng(0)
x = grid_points((-0.3, 0.7, -0.3), (0.3, 0.9, 0.3), (12, 4, 12))
x += 0.005 * rng.standard_normal(x.shape).astype(np.float32)
particles = world.add(dk.ParticleSystem(x, radius=0.015, mass=0.01))


def report(world, frame):
    if frame % 30 == 29:
        y = particles.state.x.numpy()[:, 1]
        print(f"t={world.time:5.2f}s  min y={y.min():.4f}  contacts={world.contacts.count()}")


run(world, args.frames, steps_per_frame=17, viewer=args.viewer, usd=args.usd, report=report)

y = particles.state.x.numpy()[:, 1]
print(f"lowest particle center {y.min():.4f} (radius 0.015, so resting ~0.015)")
