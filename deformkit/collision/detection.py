"""Global collision detection over all objects and static obstacles."""

import numpy as np
import warp as wp

from deformkit.collision.contact import CollisionVertices, Contacts
from deformkit.collision.geometry import TriangleMesh
from deformkit.collision.particle_plane import detect_particle_plane
from deformkit.collision.particle_sphere import detect_particle_sphere
from deformkit.collision.vertex_triangle import detect_vertex_triangle
from deformkit.core.backend import to_float, to_int, to_vec3


class CollisionSystem:
    """Gathers collision geometry from the world and runs every contact primitive.

    Pair policy: every collision vertex (the `vertex_indices` of each object's
    shape) is tested against all planes, all spheres, and the triangle mesh of
    every *other* object. Mesh-mesh contact is therefore symmetric (V_A -> F_B
    and V_B -> F_A). Self-contact is not supported: a surface vertex always lies
    on its own mesh, so it needs a different algorithm, not a flag.

    Cost: planes and spheres are tested densely, O(V * obstacles). Each mesh
    has a BVH (wp.Mesh, refit every step), so vertex-triangle queries are
    O(V * meshes * log F). Candidate generation lives entirely in
    vertex_triangle.py; contact models only see the resulting Contacts.

    Each (vertex, obstacle) pair yields at most one contact, so the contact
    buffer is sized exactly and cannot overflow (Contacts.count() checks anyway).

    max_penetration: deepest vertex-in-mesh penetration that is still resolved.
    """

    def __init__(self, max_penetration=0.1):
        self.max_penetration = max_penetration

    def build(self, world):
        """Rebuild GPU collision data. Called by World when its contents change."""
        device = world.device
        index, radius, owner = [np.zeros(0, np.int32)], [np.zeros(0, np.float32)], [np.zeros(0, np.int32)]
        self.meshes = []
        for obj in world.objects:
            shape = obj.collision_shape
            if shape is None:
                continue
            index.append(shape.vertex_indices + obj.vertex_start)
            radius.append(np.full(len(shape.vertex_indices), shape.radius))
            owner.append(np.full(len(shape.vertex_indices), obj.object_id))
            if isinstance(shape, TriangleMesh):
                # the BVH is built over the object's view of the global positions
                mesh = wp.Mesh(points=obj.state.x, indices=to_int(shape.triangles.ravel(), device))
                self.meshes.append((obj, mesh, to_int(shape.triangles, device, dtype=wp.vec3i)))

        self.verts = CollisionVertices()
        self.verts.index = to_int(np.concatenate(index), device)
        self.verts.radius = to_float(np.concatenate(radius), device)
        self.verts.owner = to_int(np.concatenate(owner), device)

        self.plane_normal = to_vec3([p.normal for p in world.planes], device)
        self.plane_offset = to_float([p.offset for p in world.planes], device)
        self.sphere_center = to_vec3([s.center for s in world.spheres], device)
        self.sphere_radius = to_float([s.radius for s in world.spheres], device)

        num_obstacles = len(world.planes) + len(world.spheres) + len(self.meshes)
        self.contacts = Contacts(self.verts.index.shape[0] * num_obstacles, device)

    def detect(self, world, margin=0.0):
        """Fill and return the contact buffer with all pairs whose gap is below `margin`."""
        x, verts, c = world.state.x, self.verts, self.contacts
        c.clear()
        detect_particle_plane(x, verts, self.plane_normal, self.plane_offset, margin, c)
        detect_particle_sphere(x, verts, self.sphere_center, self.sphere_radius, margin, c)
        for obj, mesh, triangles in self.meshes:
            mesh.refit()
            max_dist = margin + self.max_penetration + obj.collision_shape.radius
            detect_vertex_triangle(x, verts, obj, mesh, triangles, max_dist, margin, c)
        return c
