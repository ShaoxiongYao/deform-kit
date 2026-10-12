import warp as wp

from deformkit.collision.contact import ContactData, static_contact
from deformkit.collision.surface import CollisionVertices


@wp.kernel
def particle_sphere_kernel(
    x: wp.array(dtype=wp.vec3),
    verts: CollisionVertices,
    sphere_center: wp.array(dtype=wp.vec3),
    sphere_radius: wp.array(dtype=wp.float32),
    margin: float,
    contacts: ContactData,
):
    i, s = wp.tid()
    vi = verts.index[i]
    delta = x[vi] - sphere_center[s]
    dist = wp.length(delta)
    gap = dist - sphere_radius[s] - verts.radius[i]
    if gap < margin:
        n = wp.vec3(0.0, 1.0, 0.0)
        if dist > 1.0e-6:  # below this p - q is float32 noise
            n = delta / dist
        static_contact(contacts, vi, n, gap)


def detect_particle_sphere(x, verts, sphere_center, sphere_radius, margin, contacts):
    num_verts, num_spheres = verts.index.shape[0], sphere_center.shape[0]
    if num_verts == 0 or num_spheres == 0:
        return
    wp.launch(
        particle_sphere_kernel,
        dim=(num_verts, num_spheres),
        inputs=[x, verts, sphere_center, sphere_radius, margin, contacts.data],
        device=x.device,
    )
