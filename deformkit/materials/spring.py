import warp as wp


@wp.func
def hookean_spring_force(
    xi: wp.vec3, xj: wp.vec3, vi: wp.vec3, vj: wp.vec3, rest_length: float, stiffness: float, damping: float
):
    """Force on node i from a damped Hookean spring (i, j); node j receives the negative."""
    d = xj - xi
    length = wp.length(d)
    if length < 1.0e-12:
        return wp.vec3(0.0)
    dir = d / length
    stretch_rate = wp.dot(vj - vi, dir)
    return (stiffness * (length - rest_length) + damping * stretch_rate) * dir
