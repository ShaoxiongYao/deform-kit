"""Optional visualization through warp.render (needs the `viz` extra)."""

import numpy as np
import warp as wp

from deformkit.collision.geometry import Points, TriangleMesh


def _rotation_from_up(normal):
    """Quaternion rotating +Y onto `normal`."""
    up, n = np.array([0.0, 1.0, 0.0]), np.asarray(normal, dtype=np.float64)
    axis = np.cross(up, n)
    s = np.linalg.norm(axis)
    if s < 1e-9:
        return wp.quat_identity() if n[1] > 0 else wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), np.pi)
    angle = np.arctan2(s, np.dot(up, n))
    return wp.quat_from_axis_angle(wp.vec3(*(axis / s)), float(angle))


class Renderer:
    """Draw a World in an OpenGL window, or record it to a USD file when `path` is given."""

    def __init__(self, world, path=None, fps=60, plane_size=2.0, camera_pos=(0.0, 0.45, 1.4)):
        import warp.render

        self.world = world
        self.plane_size = plane_size
        if path:
            self.backend = warp.render.UsdRenderer(path, up_axis="Y", fps=fps)
        else:
            front = -np.asarray(camera_pos) + (0.0, 0.1, 0.0)
            self.backend = warp.render.OpenGLRenderer(
                title="DeformKit",
                up_axis="Y",
                fps=fps,
                near_plane=0.01,
                camera_pos=camera_pos,
                camera_front=tuple(front / np.linalg.norm(front)),
                draw_axis=False,
            )

    @property
    def is_running(self):
        return getattr(self.backend, "is_running", lambda: True)()

    def render(self):
        r, world = self.backend, self.world
        r.begin_frame(world.time)
        for i, p in enumerate(world.planes):
            pos = tuple(float(c) for c in np.asarray(p.normal) * p.offset)
            color, color2 = (0.78, 0.78, 0.78), (0.6, 0.6, 0.6)
            r.render_plane(
                f"plane{i}", pos, _rotation_from_up(p.normal), self.plane_size, self.plane_size, color, color2
            )
        for i, s in enumerate(world.spheres):
            r.render_sphere(f"sphere{i}", s.center, wp.quat_identity(), s.radius, color=(0.7, 0.5, 0.3))
        for i, obj in enumerate(world.objects):
            shape, x = obj.collision_shape, obj.state.x.numpy()
            if isinstance(shape, TriangleMesh):
                r.render_mesh(f"object{i}", x, shape.triangles.ravel())
            elif isinstance(shape, Points):
                r.render_points(f"object{i}", x[shape.vertex_indices], max(shape.radius, 1e-3))
        r.end_frame()

    def save(self):
        if hasattr(self.backend, "save"):
            self.backend.save()


def run(world, num_frames, steps_per_frame, viewer=None, usd=None, report=None):
    """Advance `world` for `num_frames` frames, optionally drawing or recording each one.

    viewer: open an OpenGL window; usd: path of a USD file to record into;
    report: optional callback(world, frame) for printing progress.
    """
    renderer = Renderer(world, path=usd, fps=round(1.0 / (world.dt * steps_per_frame))) if viewer or usd else None
    for frame in range(num_frames):
        for _ in range(steps_per_frame):
            world.step()
        if renderer is not None:
            if not renderer.is_running:
                break
            renderer.render()
        if report is not None:
            report(world, frame)
    if renderer is not None:
        renderer.save()
