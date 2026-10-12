import warp as wp

from deformkit.collision.contact import ContactData, static_contact
from deformkit.collision.surface import CollisionVertices


@wp.kernel
def particle_plane_kernel(
    x: wp.array(dtype=wp.vec3),
    verts: CollisionVertices,
    plane_normal: wp.array(dtype=wp.vec3),
    plane_offset: wp.array(dtype=wp.float32),
    margin: float,
    contacts: ContactData,
):
    i, p = wp.tid()
    vi = verts.index[i]
    n = plane_normal[p]
    # plane n^T x = h: signed distance n^T x - h, minus the particle radius
    gap = wp.dot(n, x[vi]) - plane_offset[p] - verts.radius[i]
    if gap < margin:
        static_contact(contacts, vi, n, gap)


def detect_particle_plane(x, verts, plane_normal, plane_offset, margin, contacts):
    num_verts, num_planes = verts.index.shape[0], plane_normal.shape[0]
    if num_verts == 0 or num_planes == 0:
        return
    wp.launch(
        particle_plane_kernel,
        dim=(num_verts, num_planes),
        inputs=[x, verts, plane_normal, plane_offset, margin, contacts.data],
        device=x.device,
    )
