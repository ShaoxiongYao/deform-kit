import numpy as np
import warp as wp

from deformkit.collision.contact import PenaltyContact
from deformkit.collision.detection import CollisionSystem
from deformkit.collision.geometry import Plane, Sphere
from deformkit.core.backend import get_device, to_int
from deformkit.core.state import State
from deformkit.integrators.semi_implicit import SemiImplicitEuler


@wp.kernel
def reset_forces_kernel(mass: wp.array(dtype=wp.float32), gravity: wp.vec3, force: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    force[i] = mass[i] * gravity


class World:
    """Owns all objects and obstacles, the global node arrays, and the step loop.

    Global layout: every object's nodes occupy a contiguous range
    [vertex_start, vertex_start + vertex_count) of `state.x`, `state.v`,
    `state.mass`, `force` and `object_ids`, and each object's `state` is a view
    into that range.
    """

    def __init__(
        self,
        dt=1e-3,
        gravity=(0.0, -9.81, 0.0),
        contact_model=None,
        collision_system=None,
        integrator=None,
        device=None,
    ):
        self.device = get_device(device)
        self.dt = dt
        self.gravity = wp.vec3(*gravity)
        self.contact_model = contact_model or PenaltyContact()
        self.collision_system = collision_system or CollisionSystem()
        self.integrator = integrator or SemiImplicitEuler()

        self.objects = []
        self.planes = []
        self.spheres = []
        self.time = 0.0

        self.state = None
        self.force = None
        self.object_ids = None
        self.contacts = None  # contacts found in the most recent step
        self.finalized = False

    # -- scene construction ------------------------------------------------

    def _check_not_finalized(self):
        if self.finalized:
            raise RuntimeError("Cannot modify the scene after World.finalize(); create a new World instead.")

    def add(self, obj):
        self._check_not_finalized()
        if obj.device != self.device:
            raise ValueError(f"object is on {obj.device} but world is on {self.device}")
        self.objects.append(obj)
        return obj

    def add_plane(self, normal=(0.0, 1.0, 0.0), offset=0.0):
        self._check_not_finalized()
        plane = Plane(normal, offset)
        self.planes.append(plane)
        return plane

    def add_sphere(self, center, radius):
        self._check_not_finalized()
        sphere = Sphere(tuple(center), radius)
        self.spheres.append(sphere)
        return sphere

    def finalize(self):
        """Pack all object nodes into global arrays; runs automatically on the first step().

        Afterwards each object's `state` aliases the global arrays, and the
        scene is frozen so those views can never go stale.
        """
        self._check_not_finalized()
        xs = [o.state.x.numpy() for o in self.objects]
        vs = [o.state.v.numpy() for o in self.objects]
        ms = [o.state.mass.numpy() for o in self.objects]
        cat = lambda a, shape: np.concatenate(a) if a else np.zeros(shape, np.float32)  # noqa: E731
        self.state = State.from_numpy(cat(xs, (0, 3)), cat(vs, (0, 3)), cat(ms, (0,)), device=self.device)
        self.force = wp.zeros_like(self.state.v)

        start, ids = 0, []
        for i, obj in enumerate(self.objects):
            obj.bind(self.state.view(start, obj.vertex_count), start, i)
            ids.append(np.full(obj.vertex_count, i, dtype=np.int32))
            start += obj.vertex_count
        self.object_ids = to_int(np.concatenate(ids) if ids else np.zeros(0, np.int32), self.device)

        self.collision_system.build(self)
        self.finalized = True

    @property
    def num_nodes(self):
        return 0 if self.state is None else self.state.num_nodes

    # -- simulation --------------------------------------------------------

    def step(self):
        if not self.finalized:
            self.finalize()

        # clear forces and apply gravity
        wp.launch(
            reset_forces_kernel,
            dim=self.num_nodes,
            inputs=[self.state.mass, self.gravity],
            outputs=[self.force],
            device=self.device,
        )

        for obj in self.objects:
            obj.compute_internal_forces(self.force[obj.vertex_start : obj.vertex_start + obj.vertex_count])

        self.contacts = self.collision_system.detect(self, margin=self.contact_model.margin)
        self.contact_model.apply(self, self.contacts)

        self.integrator.step(self)
        self.time += self.dt
