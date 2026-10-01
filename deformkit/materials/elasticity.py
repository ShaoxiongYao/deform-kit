"""Hyperelastic materials for tetrahedral FEM.

A material provides `stress`, a Warp function (F, mu, lam) -> first
Piola-Kirchhoff stress P, plus the two parameters it consumes. TetFEM compiles
one force kernel per stress function, so adding a material means adding one
`@wp.func` and one small class.
"""

import warp as wp


def lame_parameters(youngs_modulus, poisson_ratio):
    E, nu = youngs_modulus, poisson_ratio
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return mu, lam


@wp.func
def stable_neo_hookean_stress(F: wp.mat33, mu: float, lam: float):
    """P for  psi = mu/2 (tr(F^T F) - 3) - mu (J - 1) + lam/2 (J - 1)^2.

    Up to a constant this is lam/2 (J - alpha)^2 + mu/2 (I_C - 3) with
    alpha = 1 + mu/lam: the stable Neo-Hookean energy of Smith, de Goes and Kim,
    "Stable Neo-Hookean Flesh Simulation" (2018), without their optional
    -mu/2 log(I_C + 1) term. It is NOT the classic log(J) Neo-Hookean; it has no
    log barrier, so it stays finite for inverted elements (J <= 0) and needs no SVD.
    dJ/dF is the cofactor matrix, whose columns are cross products of F's columns.
    """
    Ft = wp.transpose(F)
    f0, f1, f2 = Ft[0], Ft[1], Ft[2]
    dJdF = wp.matrix_from_cols(wp.cross(f1, f2), wp.cross(f2, f0), wp.cross(f0, f1))
    J = wp.determinant(F)
    return mu * F + (lam * (J - 1.0) - mu) * dJdF


class StableNeoHookean:
    """Stable Neo-Hookean material (see `stable_neo_hookean_stress`) from E and Poisson ratio.

    The energy's small-strain limit has Lame parameters (mu, lam - mu), so the
    energy's lam is the Lame lambda plus mu, matching linear elasticity for small strains.
    """

    stress = stable_neo_hookean_stress

    def __init__(self, youngs_modulus=1e5, poisson_ratio=0.3):
        mu, lam = lame_parameters(youngs_modulus, poisson_ratio)
        self.mu = mu
        self.lam = lam + mu
