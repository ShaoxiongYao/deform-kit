# DeformKit

A small research codebase for GPU deformable-body simulation with simple multi-object contact, built on
[NVIDIA Warp](https://github.com/NVIDIA/warp).

It is deliberately minimal: semi-implicit Euler, penalty contact, no friction. The point is a set of clean
abstractions that FEM variants, cloth, ropes, implicit solvers and better contact models can grow into.

## Quick start

```bash
uv sync                      # core: warp-lang + numpy
uv run pytest                # 21 tests, ~5 s on a GPU
uv run examples/soft_cube_drop.py

uv sync --extra viz          # optional: OpenGL viewer (pyglet) and USD export (usd-core)
uv run examples/two_soft_objects.py --viewer
uv run examples/two_soft_objects.py --usd two_cubes.usd
```

```python
import deformkit as dk
from deformkit.utils.mesh import box_tet_mesh

world = dk.World(dt=2e-4, gravity=(0.0, -9.81, 0.0))
world.add_plane(normal=(0, 1, 0), offset=0.0)

vertices, tets = box_tet_mesh((-0.1, 0.3, -0.1), (0.1, 0.5, 0.1), resolution=5)
cube = world.add(dk.TetFEM(vertices, tets, dk.StableNeoHookean(youngs_modulus=1e5, poisson_ratio=0.4), damping=50.0))

for _ in range(5000):
    world.step()
print(cube.state.x.numpy().mean(axis=0))
```

## Design

### One step

```python
def step(self):
    reset forces to m * g                               # World
    for obj in objects: obj.compute_internal_forces(f)  # objects: internal physics only
    contacts = collision_system.detect(world, margin)   # global, geometry only
    contact_model.apply(world, contacts)                # forces (or later: corrections)
    integrator.step(world)                              # semi-implicit Euler
```

Objects never see contact, and the collision code never sees FEM or springs.

### Global node arrays

`World` owns one set of arrays for all nodes: `state.x`, `state.v`, `state.mass`, `force` and
`object_ids`. Each object occupies a contiguous range `[vertex_start, vertex_start + vertex_count)`, and
its `state` is a zero-copy **view** into that range:

```
object 0: nodes   0 ... 124     (TetFEM)
object 1: nodes 125 ... 188     (TetFEM)
object 2: nodes 189 ... 764     (ParticleSystem)
```

So objects work with local indices, while integrators and contact kernels see one flat array. A future
global solver such as implicit Euler or a sparse Newton step can assemble over the same layout.

`World.finalize()` runs on the first `step()` and freezes the scene. Adding objects or obstacles afterwards
raises `RuntimeError`, so an object's view can never point at stale storage. To change the scene, build a
new `World`.

A node with zero mass ignores forces and keeps its current velocity, so it is pinned when that velocity is
zero. There is no API yet for prescribing kinematic trajectories; that will be a separate feature rather than
an overload of `mass = 0`.

### Objects

| Object           | Internal model                                   | Collision shape             |
|------------------|--------------------------------------------------|-----------------------------|
| `ParticleSystem` | none                                             | `Points` (spheres)          |
| `SpringMesh`     | damped Hookean springs                           | `TriangleMesh` or `Points`  |
| `TetFEM`         | stable Neo-Hookean, lumped mass, strain-rate damping | boundary `TriangleMesh` |

`TetFEM` extracts its boundary triangles once at construction. The tets drive elasticity and the surface
handles collision. The material uses the stable Neo-Hookean energy of Smith, de Goes and Kim (2018),
`ψ = μ/2 (I_C − 3) + λ/2 (J − α)²` with `α = 1 + μ/λ`, without their optional `−μ/2 log(I_C + 1)` term.
This is **not** the classic `−μ log J + λ/2 (log J)²` Neo-Hookean. It has no log barrier, so it stays finite
for inverted elements and needs no SVD. λ is shifted by μ so that `StableNeoHookean(E, ν)` matches linear
elasticity at small strain.

### Collision and contact

Detection writes a fixed-capacity `Contacts` buffer on the GPU. Each record pairs one node with an "other
side" of up to three nodes plus barycentric weights, or `-1` for a static obstacle. It also stores a
normal and a signed `gap` (distance minus radii, negative when overlapping).

| Primitive          | Collides                                              |
|--------------------|-------------------------------------------------------|
| `particle_plane`   | collision vertices vs. static half-spaces             |
| `particle_sphere`  | collision vertices vs. static spheres                 |
| `vertex_triangle`  | collision vertices vs. other objects' closed meshes (Warp BVH query with inside/outside sign) |

Pair policy: each object's collision vertices are tested against every *other* object's mesh, so mesh–mesh
contact is symmetric (vertices of A against faces of B, and vertices of B against faces of A). Pairs within
the same object are skipped because self-contact is not supported.

Cost: planes and spheres are tested against every vertex, O(V × obstacles). Mesh contact goes through one
BVH per object (`wp.Mesh`, refit every step), so it costs O(V × meshes × log F) rather than brute-force
O(V × F). All candidate generation lives in `vertex_triangle.py`; contact models only see the resulting
buffer. Each vertex yields at most one contact per obstacle, so the buffer is sized exactly, and
`Contacts.count()` raises if it ever overflows.

`PenaltyContact(stiffness, damping, thickness)` turns each record into
`f = max(0, k·d − c·v_n)·n` with `d = thickness − gap`. Here `v_n < 0` means approaching, so damping adds
repulsion. While separating it only reduces the force, and the clamp keeps it from ever attracting. The
vertex gets `+f` and the triangle nodes get `−w_j·f`. Since `Σw_j = 1` and `f ∥ (p − q)`, linear and angular
momentum are both conserved.

## Extending

- **New object:** subclass `SimObject`, pass initial `x, v, mass` and a collision shape, and implement
  `compute_internal_forces(force)`. It accumulates into `force`, which is this object's slice of the
  global array.
- **New material:** write a `@wp.func` `(F, mu, lam) -> P` and a class exposing it as `stress`, like
  `StableNeoHookean`. `TetFEM` compiles one kernel per stress function.
- **New contact model:** subclass `ContactModel`, set `margin` (the detection distance), and implement
  `apply(world, contacts)`. XPBD would project positions and a barrier would use `gap` directly. Objects
  and detection stay unchanged.
- **New primitive:** write a kernel that calls `add_contact(...)`, call it from
  `CollisionSystem.detect`, and grow the buffer capacity in `build`.
- **New integrator:** any object with `step(world)`. The global arrays are all it needs.

## Choosing parameters

Everything is explicit, so the time step is limited by the stiffest, lightest thing in the scene. Rough
rules, with `m` the smallest node mass:

- contact stiffness: `dt < ~sqrt(m / k)`, where k grows when several contacts load one node
- contact damping: `c · dt / m < ~1`. Too much damping on light nodes blows up.
- FEM: `dt < ~h_min / sqrt((λ + 2μ) / ρ)`

The examples use `dt = 1e-3` for particles and `2e-4` for the ~20 cm soft cubes.

## Limitations (by design, for now)

- **No friction.** Objects slide freely. A body resting off-center on another soft body slides off because
  the support tilts, and stacks are marginal. Friction is the natural next contact feature.
- No self-contact, no particle–particle contact, no edge–edge contact, no CCD. Vertex–triangle contact
  can miss a sharp edge poking through a face between vertices.
- Vertices deeper than `CollisionSystem(max_penetration=0.1)` inside another mesh are not pushed out.
- Closed, outward-wound meshes are assumed for inside/outside classification.

## Layout

```
deformkit/
  core/          state.py  object.py  world.py  backend.py
  objects/       particle.py  spring.py  tet_fem.py
  materials/     spring.py  elasticity.py
  collision/     geometry.py  contact.py  detection.py  particle_plane.py  particle_sphere.py  vertex_triangle.py
  integrators/   semi_implicit.py
  utils/         mesh.py  render.py
examples/        particles_on_plane.py  spring_cube_drop.py  soft_cube_drop.py  two_soft_objects.py
tests/           test_springs.py  test_contact.py  test_fem.py
```
