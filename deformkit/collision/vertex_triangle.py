import warp as wp

from deformkit.collision.contact import CollisionVertices, ContactData, add_contact


@wp.kernel
def vertex_triangle_kernel(
    x: wp.array(dtype=wp.vec3),
    verts: CollisionVertices,
    mesh: wp.uint64,
    triangles: wp.array(dtype=wp.vec3i),
    mesh_owner: int,
    mesh_start: int,
    mesh_radius: float,
    max_dist: float,
    margin: float,
    contacts: ContactData,
):
    i = wp.tid()
    if verts.owner[i] == mesh_owner:
        return  # no self-contact

    vi = verts.index[i]
    p = x[vi]
    q = wp.mesh_query_point(mesh, p, max_dist)
    if not q.result:
        return

    delta = p - wp.mesh_eval_position(mesh, q.face, q.u, q.v)
    dist = wp.length(delta)
    inside = q.sign < 0.0

    # outward normal of the mesh at the closest point
    n = wp.mesh_eval_face_normal(mesh, q.face)
    if dist > 1.0e-6:  # below this p - q is float32 noise
        n = delta / dist
        if inside:
            n = -n
    signed_dist = dist
    if inside:
        signed_dist = -dist

    gap = signed_dist - verts.radius[i] - mesh_radius
    if gap < margin:
        t = triangles[q.face]
        other = wp.vec3i(t[0] + mesh_start, t[1] + mesh_start, t[2] + mesh_start)
        add_contact(contacts, vi, other, wp.vec3(q.u, q.v, 1.0 - q.u - q.v), n, gap)


def detect_vertex_triangle(x, verts, obj, mesh, triangles, max_dist, margin, contacts):
    """Collide every collision vertex (except `obj`'s own) against `obj`'s closed triangle mesh.

    mesh:      wp.Mesh over the object's node positions (local indexing)
    triangles: the same triangles as a [T] vec3i array, for mapping faces back to nodes
    """
    if verts.index.shape[0] == 0:
        return
    wp.launch(
        vertex_triangle_kernel,
        dim=verts.index.shape[0],
        inputs=[
            x,
            verts,
            mesh.id,
            triangles,
            obj.object_id,
            obj.vertex_start,
            obj.collision_shape.radius,
            max_dist,
            margin,
            contacts.data,
        ],
        device=x.device,
    )
