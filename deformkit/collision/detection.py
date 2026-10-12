"""Global collision detection over all objects and static obstacles."""

import numpy as np

from deformkit.collision.contact import Contacts
from deformkit.collision.edge_edge import detect_edge_edge
from deformkit.collision.particle_plane import detect_particle_plane
from deformkit.collision.particle_sphere import detect_particle_sphere
from deformkit.collision.surface import Surfaces
from deformkit.collision.vertex_triangle import detect_vertex_triangle
from deformkit.core.backend import to_float, to_vec3


class CollisionSystem:
    """Gathers collision geometry from the world and runs every contact primitive.

    Pairs tested:
      - every collision vertex vs. every plane and sphere
      - every collision vertex vs. every *other* object's closed mesh (signed)
      - every surface edge vs. every other object's surface edges   (edge_edge)
      - vertices and edges vs. their own object's surface           (self_contact)

    Mesh-mesh contact is symmetric: vertices and edges of A meet faces and
    edges of B, and vice versa. Self-contact is opt-in; it skips incident
    faces and edges, so it needs a margin well below the edge length, and it
    uses unsigned distances, so it can prevent crossings but not undo them.

    Cost: one launch per primitive covers all objects. A (vertex or edge,
    mesh) pair is culled by the mesh's bounding box before its BVH is queried,
    so distant objects cost a box test each.

    The contact buffer grows when detection overflows (detection then reruns),
    so contacts are never dropped. Checking costs one device sync per step.

    max_penetration: deepest vertex-in-mesh penetration that is still resolved.
    """

    def __init__(self, max_penetration=0.1, edge_edge=True, self_contact=False):
        self.max_penetration = max_penetration
        self.edge_edge = edge_edge
        self.self_contact = self_contact

    def build(self, world):
        """Rebuild GPU collision data. Called once by World.finalize()."""
        device = world.device
        self.surfaces = Surfaces(world)
        self.plane_normal = to_vec3([p.normal for p in world.planes], device)
        self.plane_offset = to_float([p.offset for p in world.planes], device)
        self.sphere_center = to_vec3([s.center for s in world.spheres], device)
        self.sphere_radius = to_float([s.radius for s in world.spheres], device)

        s = self.surfaces
        static_bound = s.num_vertices * (len(world.planes) + len(world.spheres))
        self.contacts = Contacts(max(1024, static_bound + 2 * (s.num_vertices + s.num_edges)), device)

    def detect(self, world, margin=0.0):
        """Fill and return the contact buffer with all pairs whose gap is below `margin`."""
        x = world.state.x
        self.surfaces.update(x)
        while True:
            c = self.contacts
            c.clear()
            self._run(x, margin, c)
            n = c.num_detected()
            if n <= c.capacity:
                return c
            self.contacts = Contacts(int(np.ceil(1.5 * n)), world.device)

    def _run(self, x, margin, c):
        s = self.surfaces
        detect_particle_plane(x, s.verts, self.plane_normal, self.plane_offset, margin, c)
        detect_particle_sphere(x, s.verts, self.sphere_center, self.sphere_radius, margin, c)
        detect_vertex_triangle(x, s, self.max_penetration, margin, c, self.self_contact)
        if self.edge_edge or self.self_contact:
            detect_edge_edge(x, s, margin, c, self.self_contact, other_objects=self.edge_edge)
