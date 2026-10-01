"""Collision geometry, independent of any physics model.

Static obstacles (`Plane`, `Sphere`) live in the World. Deformable objects carry
a shape (`Points`, `TriangleMesh`) whose indices refer to the object's own nodes.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Plane:
    """Half-space {x : dot(normal, x) >= offset}."""

    normal: tuple
    offset: float = 0.0

    def __post_init__(self):
        n = np.asarray(self.normal, dtype=np.float64)
        self.normal = tuple(n / np.linalg.norm(n))


@dataclass
class Sphere:
    """Static solid sphere."""

    center: tuple
    radius: float


class Points:
    """A set of object nodes treated as spheres of a common radius."""

    def __init__(self, vertex_indices, radius=0.0):
        self.vertex_indices = np.asarray(vertex_indices, dtype=np.int32).reshape(-1)
        self.radius = float(radius)


class TriangleMesh:
    """A closed, outward-oriented surface over a subset of an object's nodes.

    vertex_indices: nodes that collide as points (e.g. the boundary vertices)
    triangles:      [T, 3] node indices, outward winding
    """

    def __init__(self, triangles, vertex_indices=None, radius=0.0):
        self.triangles = np.asarray(triangles, dtype=np.int32).reshape(-1, 3)
        if vertex_indices is None:
            vertex_indices = np.unique(self.triangles)
        self.vertex_indices = np.asarray(vertex_indices, dtype=np.int32).reshape(-1)
        self.radius = float(radius)
