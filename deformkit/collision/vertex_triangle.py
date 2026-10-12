import warp as wp

from deformkit.collision.contact import ContactData, add_contact
from deformkit.collision.distance import boxes_overlap, closest_point_triangle, triangle_normal
from deformkit.collision.surface import CollisionVertices, SurfaceMeshes


@wp.kernel
def vertex_triangle_kernel(
    x: wp.array(dtype=wp.vec3),
    verts: CollisionVertices,
    s: SurfaceMeshes,
    max_penetration: float,
    margin: float,
    contacts: ContactData,
):
    """Every collision vertex against every *other* object's closed mesh (signed distance)."""
    i, m = wp.tid()
    if verts.owner[i] == s.owner[m]:
        return

    vi = verts.index[i]
    p = x[vi]
    radius = verts.radius[i] + s.radius[m]
    reach = wp.vec3(margin + radius + max_penetration)
    if not boxes_overlap(p - reach, p + reach, s.lower[m], s.upper[m]):
        return  # broad phase: vertex is nowhere near this object

    q = wp.mesh_query_point(s.ids[m], p, margin + radius + max_penetration)
    if not q.result:
        return

    t = s.triangles[s.tri_start[m] + q.face]
    xa, xb, xc = x[t[0]], x[t[1]], x[t[2]]
    w = wp.vec3(q.u, q.v, 1.0 - q.u - q.v)
    delta = p - (w[0] * xa + w[1] * xb + w[2] * xc)
    dist = wp.length(delta)
    inside = q.sign < 0.0  # ray-cast inside/outside test against the whole closed mesh

    n = triangle_normal(xa, xb, xc)  # outward
    if dist > 1.0e-6:  # below this p - q is float32 noise
        n = delta / dist
        if inside:
            n = -n
    signed_dist = dist
    if inside:
        signed_dist = -dist

    gap = signed_dist - radius
    if gap < margin:
        add_contact(contacts, wp.vec4i(vi, t[0], t[1], t[2]), wp.vec4(1.0, -w[0], -w[1], -w[2]), n, gap)


@wp.kernel
def vertex_triangle_self_kernel(
    x: wp.array(dtype=wp.vec3),
    verts: CollisionVertices,
    s: SurfaceMeshes,
    mesh_of_object: wp.array(dtype=wp.int32),
    margin: float,
    contacts: ContactData,
):
    """Each collision vertex against its own object's mesh (unsigned distance).

    Faces incident to the vertex are skipped, so this needs `margin` well below
    the edge length. Without a sign it can only prevent, not undo, crossings.
    """
    i = wp.tid()
    m = mesh_of_object[verts.owner[i]]
    if m < 0:
        return

    vi = verts.index[i]
    p = x[vi]
    radius = verts.radius[i] + s.radius[m]
    reach = margin + radius
    best_dist = reach
    best_face = int(-1)
    best_w = wp.vec3(0.0)

    query = wp.mesh_query_aabb(s.ids[m], p - wp.vec3(reach), p + wp.vec3(reach))
    face = int(0)
    while wp.mesh_query_next(query, face):
        t = s.triangles[s.tri_start[m] + face]
        if t[0] != vi and t[1] != vi and t[2] != vi:
            w = closest_point_triangle(p, x[t[0]], x[t[1]], x[t[2]])
            d = wp.length(p - (w[0] * x[t[0]] + w[1] * x[t[1]] + w[2] * x[t[2]]))
            if d < best_dist:
                best_dist = d
                best_face = face
                best_w = w

    if best_face < 0:
        return
    t = s.triangles[s.tri_start[m] + best_face]
    xa, xb, xc = x[t[0]], x[t[1]], x[t[2]]
    n = triangle_normal(xa, xb, xc)
    if best_dist > 1.0e-6:
        n = (p - (best_w[0] * xa + best_w[1] * xb + best_w[2] * xc)) / best_dist
    w = best_w
    add_contact(contacts, wp.vec4i(vi, t[0], t[1], t[2]), wp.vec4(1.0, -w[0], -w[1], -w[2]), n, best_dist - radius)


def detect_vertex_triangle(x, surfaces, max_penetration, margin, contacts, self_contact=False):
    """Vertex-triangle contacts between objects and, optionally, within each object."""
    if surfaces.num_vertices == 0 or surfaces.num_meshes == 0:
        return
    wp.launch(
        vertex_triangle_kernel,
        dim=(surfaces.num_vertices, surfaces.num_meshes),
        inputs=[x, surfaces.verts, surfaces.meshes, max_penetration, margin, contacts.data],
        device=x.device,
    )
    if self_contact:
        wp.launch(
            vertex_triangle_self_kernel,
            dim=surfaces.num_vertices,
            inputs=[x, surfaces.verts, surfaces.meshes, surfaces.mesh_of_object, margin, contacts.data],
            device=x.device,
        )
