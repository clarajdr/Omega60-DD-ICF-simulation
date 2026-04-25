"""
omega60.target
==============
Step 3 – Collision of rays with the spherical ICF target.

The target is a sphere of radius *R* centred at the origin.
For each ray (origin **o**, direction **d̂**) we solve the
quadratic intersection equation:

    |o + t·d̂|² = R²
    t² + 2(o·d̂)t + (|o|² - R²) = 0

We keep the *smaller positive* root (first hit, i.e. the outer surface
of the target as seen from the incoming beam).

Hit positions are expressed both in Cartesian and in spherical
coordinates (θ ∈ [0, π], φ ∈ [0, 2π)).
"""

import numpy as np


class SphericalTarget:
    """
    Spherical ICF target centred at the origin.

    Parameters
    ----------
    radius_m : float
        Target outer radius in metres.
        Default: 450 µm (4.5 × 10⁻⁴ m), a representative OMEGA
        direct-drive target.
    """

    def __init__(self, radius_m: float = 450e-6) -> None:
        if radius_m <= 0:
            raise ValueError("Target radius must be positive.")
        self.radius_m = float(radius_m)

    # ------------------------------------------------------------------
    # Ray–sphere intersection
    # ------------------------------------------------------------------

    def intersect(self, rays: dict) -> dict:
        """
        Step 3 – Find where each ray hits the target.

        Parameters
        ----------
        rays : dict
            Output of :func:`omega60.rays.sample_rays` (keys
            ``origins``, ``directions``, ``energies``, ``beam_ids``).

        Returns
        -------
        hits : dict with keys:
            ``hit_mask``    – (N,) bool   True when the ray hits the target
            ``hit_xyz``     – (N, 3) [m]  Cartesian impact positions (NaN where no hit)
            ``hit_theta``   – (N,) [rad]  polar angle θ ∈ [0, π]   (NaN where no hit)
            ``hit_phi``     – (N,) [rad]  azimuth φ ∈ [0, 2π)      (NaN where no hit)
            ``hit_energies``– (N,) [J]    energy of hitting rays    (0 where no hit)
            ``hit_beam_ids``– (N,) int    beam id of hitting rays   (-1 where no hit)
            ``t_hit``       – (N,) [m]    path length to first hit  (NaN where no hit)
        """
        o = rays["origins"]      # (N, 3)
        d = rays["directions"]   # (N, 3) unit vectors
        e = rays["energies"]     # (N,)
        b = rays["beam_ids"]     # (N,)
        N = o.shape[0]
        R = self.radius_m

        # Quadratic coefficients  (a = 1 because d is a unit vector)
        b_coeff = 2.0 * np.einsum("ij,ij->i", o, d)   # 2 (o·d)
        c_coeff = np.einsum("ij,ij->i", o, o) - R * R  # |o|² - R²

        discriminant = b_coeff * b_coeff - 4.0 * c_coeff  # 4(… - c_coeff)
        # equivalently: discriminant = (o·d)² - (|o|² - R²)
        # We use the standard form  t² + b_coeff t + c_coeff = 0
        # discriminant_std = b_coeff² - 4*c_coeff
        discriminant_std = b_coeff ** 2 - 4.0 * c_coeff

        hit_mask = discriminant_std >= 0.0  # (N,) bool

        # Initialise output arrays with NaN / sentinel values
        t_hit = np.full(N, np.nan)
        hit_xyz = np.full((N, 3), np.nan)
        hit_theta = np.full(N, np.nan)
        hit_phi = np.full(N, np.nan)
        hit_energies = np.zeros(N)
        hit_beam_ids = np.full(N, -1, dtype=np.int32)

        if hit_mask.any():
            idx = np.where(hit_mask)[0]
            sq = np.sqrt(discriminant_std[idx])
            t1 = (-b_coeff[idx] - sq) / 2.0
            t2 = (-b_coeff[idx] + sq) / 2.0

            # Choose the smallest positive root (first hit from outside)
            t_choice = np.where(t1 > 0.0, t1, t2)
            valid_t = t_choice > 0.0

            if valid_t.any():
                vidx = idx[valid_t]
                tv = t_choice[valid_t]

                xyz = o[vidx] + tv[:, np.newaxis] * d[vidx]  # (M, 3)
                # Snap exactly to sphere radius to avoid floating-point drift
                xyz_norm = np.linalg.norm(xyz, axis=1, keepdims=True)
                xyz = xyz / xyz_norm * R

                theta = np.arccos(np.clip(xyz[:, 2] / R, -1.0, 1.0))
                phi = np.arctan2(xyz[:, 1], xyz[:, 0]) % (2.0 * np.pi)

                # Update only valid-hit slots
                hit_mask_final = np.zeros(N, dtype=bool)
                hit_mask_final[vidx] = True

                t_hit[vidx] = tv
                hit_xyz[vidx] = xyz
                hit_theta[vidx] = theta
                hit_phi[vidx] = phi
                hit_energies[vidx] = e[vidx]
                hit_beam_ids[vidx] = b[vidx]
                hit_mask = hit_mask_final

        return {
            "hit_mask": hit_mask,
            "hit_xyz": hit_xyz,
            "hit_theta": hit_theta,
            "hit_phi": hit_phi,
            "hit_energies": hit_energies,
            "hit_beam_ids": hit_beam_ids,
            "t_hit": t_hit,
        }

    def __repr__(self) -> str:
        return f"SphericalTarget(radius_m={self.radius_m:.3e})"
