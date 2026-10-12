import warp as wp

from deformkit.collision.contact import ContactData, add_contact
from deformkit.collision.distance import boxes_overlap, triangle_normal
from deformkit.collision.surface import SurfaceEdges, SurfaceMeshes

# Closest points this close to an edge end are vertex-edge cases, already covered by vertex-triangle.
END_EPS = 1.0e-3


@wp.func
def edge_normal(x: wp.array(dtype=wp.vec3), s: SurfaceMeshes, faces: wp.vec2i):
    """Outward surface normal at an edge: the average of its adjacent faces' normals."""
    t = s.triangles[faces[0]]
    n = triangle_normal(x[t[0]], x[t[1]], x[t[2]])
    if faces[1] >= 0:
        t = s.triangles[faces[1]]
        n = n + triangle_normal(x[t[0]], x[t[1]], x[t[2]])
    return wp.normalize(n)


@wp.kernel
def edge_edge_kernel(
    x: wp.array(dtype=wp.vec3),
    e: SurfaceEdges,
    s: SurfaceMeshes,
    other_objects: int,
    self_contact: int,
    margin: float,
    contacts: ContactData,
):
    """Every surface edge against the edges of other meshes and/or its own mesh."""
    i, m = wp.tid()
    own = e.mesh[i] == m
    if (own and self_contact == 0) or (not own and other_objects == 0):
        return
    if e.mesh[i] > m:
        return  # an edge pair between two objects is symmetric: handle it from the lower mesh only

    a = e.nodes[i]
    a0, a1 = x[a[0]], x[a[1]]
    radius = s.radius[e.mesh[i]] + s.radius[m]
    reach = margin + radius
    lower = wp.min(a0, a1) - wp.vec3(reach)
    upper = wp.max(a0, a1) + wp.vec3(reach)
    if not boxes_overlap(lower, upper, s.lower[m], s.upper[m]):
        return

    n_a = edge_normal(x, s, e.faces[i])
    query = wp.mesh_query_aabb(s.ids[m], lower, upper)
    face = int(0)
    while wp.mesh_query_next(query, face):
        g = s.tri_start[m] + face
        t = s.triangles[g]
        nbr = s.neighbors[g]
        for k in range(3):
            b0 = t[k]
            b1 = t[(k + 1) % 3]
            # visit each edge once (from its lower-index face); in self mode skip
            # edges sharing a node and count each pair once (from the lower node)
            visit = nbr[k] < 0 or nbr[k] > g
            if own:
                shares = b0 == a[0] or b0 == a[1] or b1 == a[0] or b1 == a[1]
                visit = visit and not shares and wp.min(a[0], a[1]) < wp.min(b0, b1)
            if visit:
                st = wp.closest_point_edge_edge(a0, a1, x[b0], x[b1], 1.0e-12)
                sa, tb, dist = st[0], st[1], st[2]
                interior = sa > END_EPS and sa < 1.0 - END_EPS and tb > END_EPS and tb < 1.0 - END_EPS
                if interior and dist < reach:
                    delta = (a0 + sa * (a1 - a0)) - (x[b0] + tb * (x[b1] - x[b0]))
                    # push A out of B: along B's outward normal and against A's
                    out = edge_normal(x, s, wp.vec2i(g, nbr[k])) - n_a
                    n = wp.vec3(0.0)
                    signed_dist = dist
                    ok = True
                    if dist > 1.0e-6:
                        n = delta / dist
                        if not own and wp.dot(n, out) < 0.0:
                            n = -n  # the edges have crossed
                            signed_dist = -dist
                    elif not own and wp.length(out) > 1.0e-6:
                        n = wp.normalize(out)
                    else:
                        ok = False
                    if ok:
                        nodes = wp.vec4i(a[0], a[1], b0, b1)
                        weights = wp.vec4(1.0 - sa, sa, -(1.0 - tb), -tb)
                        add_contact(contacts, nodes, weights, n, signed_dist - radius)


def detect_edge_edge(x, surfaces, margin, contacts, self_contact=False, other_objects=True):
    if surfaces.num_edges == 0 or surfaces.num_meshes == 0:
        return
    wp.launch(
        edge_edge_kernel,
        dim=(surfaces.num_edges, surfaces.num_meshes),
        inputs=[x, surfaces.edges, surfaces.meshes, int(other_objects), int(self_contact), margin, contacts.data],
        device=x.device,
    )
