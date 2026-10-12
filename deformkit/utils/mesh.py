"""NumPy mesh utilities: box tetrahedralization, boundary extraction, edges."""

from collections import Counter
from itertools import product

import numpy as np


def grid_points(lo, hi, resolution):
    """Regular lattice of (res+1)^3 points in the box [lo, hi], ordered x-fastest."""
    res = np.broadcast_to(resolution, 3)
    axes = [np.linspace(lo[d], hi[d], res[d] + 1) for d in range(3)]
    z, y, x = np.meshgrid(axes[2], axes[1], axes[0], indexing="ij")
    return np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1).astype(np.float32)


def box_tet_mesh(lo, hi, resolution):
    """Tetrahedralize a box with 5 tets per grid cell, mirrored in alternate cells.

    The alternating split is reflection-symmetric, so the mesh has no preferred
    diagonal (a 6-tet Kuhn split shears sideways under axial load).
    Returns (vertices [N, 3], tets [M, 4]) with positively oriented tets.
    """
    res = np.broadcast_to(resolution, 3)
    vertices = grid_points(lo, hi, res)
    nx, ny = res[0] + 1, res[1] + 1
    corners = list(product((0, 1), repeat=3))

    def vid(i, j, k, c):
        return (i + c[0]) + nx * ((j + c[1]) + ny * (k + c[2]))

    def flip(c, axis):
        return tuple(1 - b if a == axis else b for a, b in enumerate(c))

    tets = []
    for k in range(res[2]):
        for j in range(res[1]):
            for i in range(res[0]):
                parity = (i + j + k) % 2
                # a central tet on four alternating corners ...
                tets.append([vid(i, j, k, c) for c in corners if (sum(c) + parity) % 2 == 0])
                # ... plus one tet cutting off each remaining corner
                for c in corners:
                    if (sum(c) + parity) % 2 == 1:
                        tets.append([vid(i, j, k, c)] + [vid(i, j, k, flip(c, a)) for a in range(3)])
    tets = orient_tets(vertices, np.array(tets, dtype=np.int32))
    return vertices, tets


def rotate(points, axis, angle, center=None):
    """Rotate points by `angle` radians about `axis` through `center` (default: centroid)."""
    points = np.asarray(points, dtype=np.float64)
    c = points.mean(axis=0) if center is None else np.asarray(center, dtype=np.float64)
    k = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
    return ((points - c) @ R.T + c).astype(np.float32)


def remove_unused_vertices(vertices, elements):
    """Drop vertices no element references; returns (vertices, reindexed elements)."""
    used, inverse = np.unique(np.asarray(elements), return_inverse=True)
    return np.asarray(vertices)[used], inverse.reshape(np.shape(elements)).astype(np.int32)


def tet_signed_volumes(vertices, tets):
    p = vertices[tets]
    return np.einsum("ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0]) / 6.0


def orient_tets(vertices, tets):
    """Swap two indices of every negatively oriented tet."""
    tets = np.array(tets, dtype=np.int32)
    neg = tet_signed_volumes(vertices, tets) < 0
    tets[neg] = tets[neg][:, [0, 1, 3, 2]]
    return tets


def boundary_triangles(tets):
    """Faces referenced by exactly one tet, wound outward (tets must be positive)."""
    # outward faces of a positively oriented tet (a, b, c, d)
    local = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]
    faces = np.concatenate([tets[:, f] for f in local])
    keys = [tuple(sorted(f)) for f in faces.tolist()]
    count = Counter(keys)
    return np.array([f for f, key in zip(faces, keys) if count[key] == 1], dtype=np.int32)


def triangle_neighbors(triangles):
    """[T, 3] index of the triangle across edge k = (t[k], t[k+1]), or -1 on an open boundary."""
    neighbors = -np.ones((len(triangles), 3), dtype=np.int32)
    open_edges = {}
    for t, tri in enumerate(np.asarray(triangles).tolist()):
        for k in range(3):
            a, b = tri[k], tri[(k + 1) % 3]
            key = (min(a, b), max(a, b))
            if key in open_edges:
                s, j = open_edges.pop(key)
                neighbors[t, k], neighbors[s, j] = s, t
            else:
                open_edges[key] = (t, k)
    return neighbors


def triangle_edges(triangles):
    """Unique edges of a triangle mesh and their adjacent faces.

    Returns (edges [E, 2], faces [E, 2]); faces[:, 1] is -1 on an open boundary.
    """
    triangles = np.asarray(triangles)
    neighbors = triangle_neighbors(triangles)
    t, k = np.nonzero((neighbors < 0) | (neighbors > np.arange(len(triangles))[:, None]))
    edges = np.stack([triangles[t, k], triangles[t, (k + 1) % 3]], axis=1)
    faces = np.stack([t, neighbors[t, k]], axis=1)
    return edges.astype(np.int32), faces.astype(np.int32)


def tet_edges(tets):
    """Unique undirected edges of a tet mesh, [E, 2]."""
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    edges = np.concatenate([tets[:, p] for p in pairs])
    return np.unique(np.sort(edges, axis=1), axis=0).astype(np.int32)
