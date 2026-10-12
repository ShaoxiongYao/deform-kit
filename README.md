# DeformKit

A small research codebase for GPU deformable-body simulation with multi-object contact, built on
[NVIDIA Warp](https://github.com/NVIDIA/warp).

It is deliberately minimal: semi-implicit Euler, penalty contact with Coulomb friction, ~1,600 lines. The
point is a set of clean abstractions that FEM variants, cloth, ropes, implicit solvers and other contact
models can grow into.

## Quick start

```bash
uv sync                      # core: warp-lang + numpy
uv run pytest                # 52 tests, ~25 s on a GPU
uv run examples/soft_stack.py

uv sync --extra viz          # optional: OpenGL viewer (pyglet) and USD export (usd-core)
uv run examples/soft_stack.py --viewer
uv run examples/soft_stack.py --usd stack.usd
```

```python
import deformkit as dk
from deformkit.utils.mesh import box_tet_mesh

world = dk.World(dt=2e-4, contact_model=dk.PenaltyContact(stiffness=2e4, damping=5.0, friction=0.5))
world.add_plane(normal=(0, 1, 0), offset=0.0)

vertices, tets = box_tet_mesh((-0.1, 0.3, -0.1), (0.1, 0.5, 0.1), resolution=5)
cube = world.add(dk.TetFEM(vertices, tets, dk.StableNeoHookean(youngs_modulus=1e5, poisson_ratio=0.4), damping=50.0))

for _ in range(5000):
    world.step()
print(cube.state.x.numpy().mean(axis=0))
```

| Example                 | Shows                                                          |
|-------------------------|----------------------------------------------------------------|
| `particles_on_plane.py` | particles vs. a sphere and a plane                             |
| `spring_cube_drop.py`   | mass-spring cube with a triangle collision surface             |
| `soft_cube_drop.py`     | Neo-Hookean FEM cube                                           |
| `two_soft_objects.py`   | frictionless cube-on-cube contact: the upper cube slides off   |
| `soft_stack.py`         | the same kind of scene with friction: an off-center tower holds |

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

### Contact records

Detection fills a GPU `Contacts` buffer. Every primitive writes the same record: up to four nodes with
signed weights, where the relative position of the two sides is `Σ wₖ xₖ`. Positive weights form side A
and negative weights side B; `-1` marks an unused slot, such as a static obstacle.

| Primitive         | Nodes            | Weights                     |
|-------------------|------------------|-----------------------------|
| vertex–obstacle   | `(p, -, -, -)`   | `(1, 0, 0, 0)`              |
| vertex–triangle   | `(p, a, b, c)`   | `(1, −w_a, −w_b, −w_c)`     |
| edge–edge         | `(a₀, a₁, b₀, b₁)` | `(1−s, s, −(1−t), −t)`    |

Each record also stores a `normal` (the direction side A is pushed) and a signed `gap` (distance minus
radii, negative when overlapping). A contact model maps a record to a force `f` and applies `wₖ f` to node
k. The weights sum to zero, so linear momentum is conserved by construction. `Contacts.numpy()` returns
the records for inspection.

### Collision detection

| Primitive          | Collides                                                                          |
|--------------------|-----------------------------------------------------------------------------------|
| `particle_plane`   | collision vertices vs. static half-spaces                                         |
| `particle_sphere`  | collision vertices vs. static spheres                                             |
| `vertex_triangle`  | collision vertices vs. other objects' closed meshes (signed: BVH query plus ray-cast inside test) |
| `edge_edge`        | surface edges vs. other objects' surface edges (sign from the adjacent faces' normals) |
| self-contact       | vertices and edges vs. their own object's surface (opt-in, unsigned)              |

`CollisionSystem(max_penetration=0.1, edge_edge=True, self_contact=False)`:

- **Pair policy.** Mesh–mesh contact is symmetric: vertices of A meet faces of B and vertices of B meet
  faces of A. Each crossing edge pair yields one record.
- **Edge–edge** covers what vertex–triangle misses: two edges crossing between vertices, e.g. two bars laid
  across each other. Closest points within 0.1% of an edge end are left to vertex–triangle contact.
- **Self-contact** skips faces and edges that share a node with the query, so it needs a margin well below
  the edge length. Its distances are unsigned, so it prevents crossings but cannot undo one.
- **Cost.** One launch per primitive covers all objects. Each (vertex or edge, mesh) pair is first culled
  by the mesh's bounding box, then queried against that mesh's BVH. With 125 small cubes this cut
  detection from 7.6 ms to 2.5 ms per step. Most of what remains is refitting the 125 BVHs, which is
  Python-side launch overhead.
- **Buffer.** When detection overflows the buffer, the buffer grows and detection reruns, so contacts are
  never dropped. The check costs one device sync per step.

### Penalty contact and friction

`PenaltyContact(stiffness, damping, thickness, friction=0.0)`:

- **Normal:** `f_n = max(0, k·d − c·v_n)` with `d = thickness − gap`. `v_n < 0` while approaching, so
  damping adds repulsion; while separating it only reduces the force, and the clamp keeps it from ever
  attracting.
- **Friction:** Coulomb, `|f_t| ≤ μ f_n`, opposing the tangential relative velocity predicted for the end of
  the step from all other forces. The force is capped at what stops that motion, so contacts stick exactly
  (a sliding particle stops at `v²/2μg` and stays) without adding a stiffness limit on `dt`. When several
  contacts share a node, its mass is split among them so their combined friction cannot overshoot.
- **Angular momentum.** Vertex–triangle normal forces lie along `p − q`, so they also conserve angular
  momentum. Edge–edge normals and friction forces act at a separation of up to `thickness`, which leaves
  a residual torque of order `thickness × f`.

## Extending

- **New object:** subclass `SimObject`, pass initial `x, v, mass` and a collision shape, and implement
  `compute_internal_forces(force)`. It accumulates into `force`, which is this object's slice of the
  global array.
- **New material:** write a `@wp.func` `(F, mu, lam) -> P` and a class exposing it as `stress`, like
  `StableNeoHookean`. `TetFEM` compiles one kernel per stress function.
- **New contact model:** subclass `ContactModel`, set `margin` (the detection distance), and implement
  `apply(world, contacts)` over the generic records. XPBD would project positions and a barrier would use
  `gap` directly. Objects and detection stay unchanged.
- **New primitive:** write a kernel that calls `add_contact(contacts, nodes, weights, normal, gap)` and
  call it from `CollisionSystem._run`. The buffer grows on its own.
- **New integrator:** any object with `step(world)`. The global arrays are all it needs.

## Choosing parameters

Everything is explicit, so the time step is limited by the stiffest, lightest thing in the scene. Rough
rules, with `m` the smallest node mass:

- contact stiffness: `dt < ~sqrt(m / k)`, where k grows when several contacts load one node
- contact damping: `c · dt / m < ~1`. Too much damping on light nodes blows up.
- FEM: `dt < ~h_min / sqrt((λ + 2μ) / ρ)`
- penalty contacts act only within `thickness` of touching (edge–edge also only within `thickness` past
  crossing), so `k` must stop the fastest impact within that distance. Two crossing bars of 1 kg with
  `k = 2e4` hold at 1 m/s but tunnel at 2 m/s.

The examples use `dt = 1e-3` for particles and `2e-4` for the ~20 cm soft cubes.

## Limitations (by design, for now)

- One friction coefficient per contact model; no per-object materials yet.
- No particle–particle contact and no CCD, so fast thin features can tunnel (see above).
- Vertices deeper than `max_penetration` inside another mesh are not pushed out. Edge–edge pairs that have
  crossed by more than the margin are left to vertex–triangle contact.
- Closed, outward-wound meshes are assumed for inside/outside classification.
- The per-step overflow check is a host sync, which a future CUDA-graph step would replace with a
  device-side flag.

## Layout

```
deformkit/
  core/          state.py  object.py  world.py  backend.py
  objects/       particle.py  spring.py  tet_fem.py
  materials/     spring.py  elasticity.py
  collision/     geometry.py  contact.py  penalty.py  detection.py  surface.py  distance.py
                 particle_plane.py  particle_sphere.py  vertex_triangle.py  edge_edge.py
  integrators/   semi_implicit.py
  utils/         mesh.py  render.py
examples/        particles_on_plane.py  spring_cube_drop.py  soft_cube_drop.py  two_soft_objects.py  soft_stack.py
tests/           test_springs.py  test_fem.py  test_contact.py  test_friction.py  test_world.py
```
