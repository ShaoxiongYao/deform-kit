"""GPU collision data gathered from every object's collision shape.

All indices are global node indices, so one kernel launch can cover every
object at once instead of one launch per object.
"""

import numpy as np
import warp as wp

from deformkit.collision.geometry import TriangleMesh
from deformkit.core.backend import to_float, to_int
from deformkit.utils.mesh import triangle_edges, triangle_neighbors


@wp.struct
class CollisionVertices:
    """Nodes that collide as points (the shapes' `vertex_indices`)."""

    index: wp.array(dtype=wp.int32)
    radius: wp.array(dtype=wp.float32)
    owner: wp.array(dtype=wp.int32)  # object id


@wp.struct
class SurfaceMeshes:
    """Closed triangle surfaces, one per object that has a TriangleMesh."""

    ids: wp.array(dtype=wp.uint64)  # wp.Mesh BVH over the object's own (local) nodes
    owner: wp.array(dtype=wp.int32)  # object id
    radius: wp.array(dtype=wp.float32)
    tri_start: wp.array(dtype=wp.int32)  # first entry of this mesh in `triangles`
    lower: wp.array(dtype=wp.vec3)  # bounding box, refreshed every step
    upper: wp.array(dtype=wp.vec3)
    triangles: wp.array(dtype=wp.vec3i)  # all meshes' triangles, global node indices
    tri_mesh: wp.array(dtype=wp.int32)  # mesh of each triangle
    neighbors: wp.array(dtype=wp.vec3i)  # triangle across edge k = (t[k], t[k+1]), -1 if open


@wp.struct
class SurfaceEdges:
    nodes: wp.array(dtype=wp.vec2i)  # global node indices
    faces: wp.array(dtype=wp.vec2i)  # adjacent triangles (indices into SurfaceMeshes.triangles)
    mesh: wp.array(dtype=wp.int32)


@wp.kernel
def surface_bounds_kernel(x: wp.array(dtype=wp.vec3), s: SurfaceMeshes):
    t = wp.tid()
    m = s.tri_mesh[t]
    tri = s.triangles[t]
    for k in range(3):
        p = x[tri[k]]
        wp.atomic_min(s.lower, m, p)
        wp.atomic_max(s.upper, m, p)


class Surfaces:
    """Builds the structs above from a world and keeps the BVHs up to date."""

    def __init__(self, world):
        device = world.device
        index, radius, owner = [np.zeros(0, np.int32)], [np.zeros(0, np.float32)], [np.zeros(0, np.int32)]
        self.bvhs = []  # keeps the wp.Mesh objects alive
        mesh_of_object = -np.ones(len(world.objects), np.int32)
        m_owner, m_radius, tri_start = [], [], []
        tris, tri_mesh, nbrs, edges, edge_faces, edge_mesh = [], [], [], [], [], []
        num_tris = 0

        for obj in world.objects:
            shape = obj.collision_shape
            if shape is None:
                continue
            index.append(shape.vertex_indices + obj.vertex_start)
            radius.append(np.full(len(shape.vertex_indices), shape.radius))
            owner.append(np.full(len(shape.vertex_indices), obj.object_id))
            if not isinstance(shape, TriangleMesh):
                continue

            m = len(self.bvhs)
            mesh_of_object[obj.object_id] = m
            # the BVH is built over the object's view of the global positions
            self.bvhs.append(wp.Mesh(points=obj.state.x, indices=to_int(shape.triangles.ravel(), device)))
            m_owner.append(obj.object_id)
            m_radius.append(shape.radius)
            tri_start.append(num_tris)

            local_nbr = triangle_neighbors(shape.triangles)
            e, f = triangle_edges(shape.triangles)
            tris.append(shape.triangles + obj.vertex_start)
            tri_mesh.append(np.full(len(shape.triangles), m))
            nbrs.append(np.where(local_nbr >= 0, local_nbr + num_tris, -1))
            edges.append(e + obj.vertex_start)
            edge_faces.append(np.where(f >= 0, f + num_tris, -1))
            edge_mesh.append(np.full(len(e), m))
            num_tris += len(shape.triangles)

        self.verts = CollisionVertices()
        self.verts.index = to_int(np.concatenate(index), device)
        self.verts.radius = to_float(np.concatenate(radius), device)
        self.verts.owner = to_int(np.concatenate(owner), device)
        self.mesh_of_object = to_int(mesh_of_object, device)

        cat = lambda xs, shape: np.concatenate(xs) if xs else np.zeros(shape, np.int32)  # noqa: E731
        s = SurfaceMeshes()
        s.ids = wp.array([b.id for b in self.bvhs], dtype=wp.uint64, device=device)
        s.owner = to_int(m_owner, device)
        s.radius = to_float(m_radius, device)
        s.tri_start = to_int(tri_start, device)
        s.lower = wp.zeros(len(self.bvhs), dtype=wp.vec3, device=device)
        s.upper = wp.zeros(len(self.bvhs), dtype=wp.vec3, device=device)
        s.triangles = to_int(cat(tris, (0, 3)), device, dtype=wp.vec3i)
        s.tri_mesh = to_int(cat(tri_mesh, (0,)), device)
        s.neighbors = to_int(cat(nbrs, (0, 3)), device, dtype=wp.vec3i)
        self.meshes = s

        e = SurfaceEdges()
        e.nodes = to_int(cat(edges, (0, 2)), device, dtype=wp.vec2i)
        e.faces = to_int(cat(edge_faces, (0, 2)), device, dtype=wp.vec2i)
        e.mesh = to_int(cat(edge_mesh, (0,)), device)
        self.edges = e

    @property
    def num_vertices(self):
        return self.verts.index.shape[0]

    @property
    def num_meshes(self):
        return len(self.bvhs)

    @property
    def num_edges(self):
        return self.edges.nodes.shape[0]

    def update(self, x):
        """Refit every BVH and recompute every mesh's bounding box."""
        for bvh in self.bvhs:
            bvh.refit()
        if self.num_meshes == 0:
            return
        self.meshes.lower.fill_(wp.vec3(1.0e30))
        self.meshes.upper.fill_(wp.vec3(-1.0e30))
        wp.launch(surface_bounds_kernel, dim=self.meshes.triangles.shape[0], inputs=[x, self.meshes], device=x.device)
