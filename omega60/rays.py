"""
omega60.rays
============
Step 2 – Sample laser rays.

Each OMEGA beam is modelled as a converging cone of rays, uniformly
sampled over the circular cross-section of the focusing lens.  The
chief ray of each beam passes through the target centre; all rays in
a beam converge to the same focal point (target centre) regardless of
their position on the lens.

The output is a set of rays described by their origin and unit-direction
vectors, together with per-ray energy weights.
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .beams import Omega60Laser


def _random_disk_uniform(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sample *n* points uniformly inside a unit disk (Shirley mapping).

    Returns
    -------
    pts : ndarray of shape (n, 2)
        (u, v) coordinates with u² + v² ≤ 1.
    """
    # Concentric-disk mapping for uniform density
    r = np.sqrt(rng.uniform(0.0, 1.0, n))
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])


def _build_local_frame(axis: np.ndarray):
    """
    Build two unit vectors (u_hat, v_hat) orthogonal to *axis* using a
    numerically stable Gram–Schmidt procedure.

    Parameters
    ----------
    axis : ndarray of shape (3,)
        Unit vector.

    Returns
    -------
    u_hat, v_hat : ndarray of shape (3,)
    """
    # Choose a reference vector that is not parallel to axis
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(axis, ref)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    u_hat = np.cross(axis, ref)
    u_hat /= np.linalg.norm(u_hat)
    v_hat = np.cross(axis, u_hat)
    v_hat /= np.linalg.norm(v_hat)
    return u_hat, v_hat


def sample_rays(
    laser,
    rays_per_beam: int = 1000,
    seed: int | None = None,
) -> dict:
    """
    Step 2 – Sample rays from all 60 OMEGA beams.

    For each beam the focusing lens is modelled as a disk of radius
    ``laser.beam_radius_m`` centred at ``laser.lens_positions[i]``.
    Rays are sampled uniformly over this disk and each ray is directed
    toward the target centre (origin), making this a perfect on-axis
    focusing model.

    Parameters
    ----------
    laser : Omega60Laser
        Laser object providing geometry and energy.
    rays_per_beam : int
        Number of Monte-Carlo rays per beam.  Total rays = 60 × rays_per_beam.
    seed : int or None
        Random seed; overrides ``laser.seed`` if given.

    Returns
    -------
    rays : dict with keys:
        ``origins``    – (N, 3) float64 [m]  starting position of every ray
        ``directions`` – (N, 3) float64      unit direction vector (→ target)
        ``energies``   – (N,)   float64 [J]  energy carried by each ray
        ``beam_ids``   – (N,)   int32         beam index (0–59) for each ray
    """
    rng = np.random.default_rng(seed if seed is not None else laser.seed)

    n_total = laser.NUM_BEAMS * rays_per_beam
    origins = np.empty((n_total, 3), dtype=np.float64)
    directions = np.empty((n_total, 3), dtype=np.float64)
    energies = np.empty(n_total, dtype=np.float64)
    beam_ids = np.empty(n_total, dtype=np.int32)

    # Energy per ray (equal weight within each beam)
    energy_per_ray = laser.energy_per_beam_J / rays_per_beam

    for i in range(laser.NUM_BEAMS):
        sl = slice(i * rays_per_beam, (i + 1) * rays_per_beam)

        # Chief ray: from lens centre to target centre
        lens_centre = laser.lens_positions[i]   # (3,)
        beam_axis = laser.beam_axes[i]           # unit vec pointing away from target

        # Build orthonormal frame on the lens plane
        u_hat, v_hat = _build_local_frame(beam_axis)

        # Sample uniform positions on the lens disk
        disk_pts = _random_disk_uniform(rays_per_beam, rng)  # (n, 2)
        offsets = (
            disk_pts[:, 0:1] * laser.beam_radius_m * u_hat
            + disk_pts[:, 1:2] * laser.beam_radius_m * v_hat
        )  # (n, 3)

        # Ray origins on the lens aperture
        ray_origins = lens_centre + offsets  # (n, 3)

        # Each ray is directed from its origin toward the target centre (origin)
        to_target = -ray_origins                         # (n, 3)
        norms = np.linalg.norm(to_target, axis=1, keepdims=True)
        ray_dirs = to_target / norms                     # (n, 3) unit vectors

        origins[sl] = ray_origins
        directions[sl] = ray_dirs
        energies[sl] = energy_per_ray
        beam_ids[sl] = i

    return {
        "origins": origins,
        "directions": directions,
        "energies": energies,
        "beam_ids": beam_ids,
    }
