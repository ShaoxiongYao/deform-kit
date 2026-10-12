"""Small geometric Warp functions shared by the contact primitives."""

import warp as wp


@wp.func
def triangle_normal(a: wp.vec3, b: wp.vec3, c: wp.vec3):
    return wp.normalize(wp.cross(b - a, c - a))


@wp.func
def closest_point_triangle(p: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3):
    """Barycentric weights (wa, wb, wc) of the point on triangle abc closest to p.

    Voronoi-region walk from Ericson, "Real-Time Collision Detection", 5.1.5.
    """
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = wp.dot(ab, ap)
    d2 = wp.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return wp.vec3(1.0, 0.0, 0.0)

    bp = p - b
    d3 = wp.dot(ab, bp)
    d4 = wp.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return wp.vec3(0.0, 1.0, 0.0)

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return wp.vec3(1.0 - v, v, 0.0)

    cp = p - c
    d5 = wp.dot(ab, cp)
    d6 = wp.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return wp.vec3(0.0, 0.0, 1.0)

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return wp.vec3(1.0 - w, 0.0, w)

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return wp.vec3(0.0, 1.0 - w, w)

    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    return wp.vec3(1.0 - v - w, v, w)


@wp.func
def boxes_overlap(lower_a: wp.vec3, upper_a: wp.vec3, lower_b: wp.vec3, upper_b: wp.vec3):
    return (
        lower_a[0] <= upper_b[0]
        and lower_b[0] <= upper_a[0]
        and lower_a[1] <= upper_b[1]
        and lower_b[1] <= upper_a[1]
        and lower_a[2] <= upper_b[2]
        and lower_b[2] <= upper_a[2]
    )
