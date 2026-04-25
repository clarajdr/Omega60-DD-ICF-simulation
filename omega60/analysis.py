"""
omega60.analysis
================
Steps 4, 5 & 6 – Collect statistics, spherical-harmonic decomposition,
and average intensity.

Given the hit positions (θ, φ) and energies of rays that struck the
target, this module:

4. Collects statistical results: total energy on target, per-beam
   breakdown, hit-count map on a (θ, φ) grid.

5. Decomposes the surface irradiance into real spherical harmonics Yₗᵐ
   up to a chosen maximum degree L_max.

6. Computes the average intensity (W m⁻²) over the illuminated surface.

Non-uniformity metric (rms)
---------------------------
    σ_rms = √( Σ_{l=1}^{L_max} Σ_m |aₗₘ|² ) / |a₀₀|

where aₗₘ are the coefficients of the real spherical-harmonic expansion
of the irradiance map.
"""

import numpy as np
from scipy.special import sph_harm_y  # complex Yₗᵐ  (scipy ≥ 1.15)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class IrradianceAnalysis:
    """
    Analyse the irradiance distribution on the target surface.

    Parameters
    ----------
    hits : dict
        Output of :meth:`SphericalTarget.intersect`.
    laser :
        :class:`Omega60Laser` used to generate the rays (provides
        energy and geometry metadata).
    target :
        :class:`SphericalTarget` (provides target radius for normalisation).
    n_theta : int
        Number of latitude bins for the grid map.  Default: 90.
    n_phi : int
        Number of longitude bins for the grid map.  Default: 180.
    L_max : int
        Maximum spherical-harmonic degree for the decomposition.
        Default: 10.
    """

    def __init__(
        self,
        hits: dict,
        laser,
        target,
        n_theta: int = 90,
        n_phi: int = 180,
        L_max: int = 10,
    ) -> None:
        self.hits = hits
        self.laser = laser
        self.target = target
        self.n_theta = n_theta
        self.n_phi = n_phi
        self.L_max = L_max

        # ------------------------------------------------------------------
        # Step 4 – Collect statistics
        # ------------------------------------------------------------------
        mask = hits["hit_mask"]
        self.total_energy_on_target_J: float = float(
            hits["hit_energies"][mask].sum()
        )
        self.n_rays_total: int = int(mask.size)
        self.n_rays_hit: int = int(mask.sum())
        self.coupling_efficiency: float = (
            self.total_energy_on_target_J / laser.total_energy_J
        )

        # Per-beam statistics
        self.energy_per_beam: np.ndarray = np.zeros(
            laser.NUM_BEAMS, dtype=np.float64
        )
        self.hits_per_beam: np.ndarray = np.zeros(
            laser.NUM_BEAMS, dtype=np.int64
        )
        valid_beam_ids = hits["hit_beam_ids"][mask]
        valid_energies = hits["hit_energies"][mask]
        for b in range(laser.NUM_BEAMS):
            sel = valid_beam_ids == b
            self.energy_per_beam[b] = valid_energies[sel].sum()
            self.hits_per_beam[b] = int(sel.sum())

        # ------------------------------------------------------------------
        # Step 4 (continued) – Build irradiance grid map
        # ------------------------------------------------------------------
        # dΩ = sin θ dθ dφ  →  irradiance ∝ energy / (R² sin θ dθ dφ)
        theta_edges = np.linspace(0.0, np.pi, n_theta + 1)
        phi_edges = np.linspace(0.0, 2.0 * np.pi, n_phi + 1)

        theta_hit = hits["hit_theta"][mask]
        phi_hit = hits["hit_phi"][mask]
        energy_hit = hits["hit_energies"][mask]

        # Accumulate energy per cell
        grid, _, _ = np.histogram2d(
            theta_hit, phi_hit,
            bins=[theta_edges, phi_edges],
            weights=energy_hit,
        )  # (n_theta, n_phi)

        # Convert to irradiance [J/sr] (energy per steradian)
        theta_centres = 0.5 * (theta_edges[:-1] + theta_edges[1:])
        phi_centres = 0.5 * (phi_edges[:-1] + phi_edges[1:])
        dtheta = np.diff(theta_edges)
        dphi = np.diff(phi_edges)
        sin_theta = np.sin(theta_centres)[:, np.newaxis]  # (n_theta, 1)
        d_omega = sin_theta * dtheta[:, np.newaxis] * dphi[np.newaxis, :]  # sr

        # Avoid division by zero in polar caps where sin θ ≈ 0
        d_omega_safe = np.where(d_omega > 0, d_omega, np.nan)
        self.irradiance_grid: np.ndarray = grid / d_omega_safe  # J/sr  (n_theta, n_phi)
        self.theta_centres: np.ndarray = theta_centres
        self.phi_centres: np.ndarray = phi_centres

        # ------------------------------------------------------------------
        # Step 5 – Spherical-harmonic decomposition
        # ------------------------------------------------------------------
        self.sh_coefficients: dict[tuple[int, int], float] = {}
        self._compute_sh_coefficients()

        # ------------------------------------------------------------------
        # Step 6 – Average intensity
        # ------------------------------------------------------------------
        target_area = 4.0 * np.pi * target.radius_m ** 2  # m²
        self.average_intensity_W_m2: float = (
            self.total_energy_on_target_J
            / laser.pulse_duration_s
            / target_area
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_sh_coefficients(self) -> None:
        """
        Step 5 – Decompose irradiance map into real spherical harmonics.

        The real spherical harmonics Yₗᵐ are related to the complex ones
        by:
            Y_l^{m>0}  = (−1)^m √2 Re[Y_l^m_complex]
            Y_l^{  0}  = Y_l^0_complex  (real)
            Y_l^{m<0}  = (−1)^m √2 Im[Y_l^{|m|}_complex]

        Coefficients are computed by numerical integration over the
        irradiance grid with appropriate solid-angle weights.
        """
        # Replace NaN cells with zero for integration
        I = np.nan_to_num(self.irradiance_grid, nan=0.0)  # (n_theta, n_phi)

        # Solid-angle element per cell
        theta_edges = np.linspace(0.0, np.pi, self.n_theta + 1)
        phi_edges = np.linspace(0.0, 2.0 * np.pi, self.n_phi + 1)
        dtheta = np.diff(theta_edges)
        dphi = np.diff(phi_edges)
        sin_theta = np.sin(self.theta_centres)   # (n_theta,)
        d_omega = (
            sin_theta[:, np.newaxis]
            * dtheta[:, np.newaxis]
            * dphi[np.newaxis, :]
        )  # (n_theta, n_phi)

        # Meshgrid for (θ, φ)
        THETA, PHI = np.meshgrid(self.theta_centres, self.phi_centres, indexing="ij")
        # THETA, PHI each have shape (n_theta, n_phi)

        for l in range(self.L_max + 1):
            for m in range(-l, l + 1):
                m_abs = abs(m)
                # sph_harm_y(n, m, theta, phi) — scipy ≥ 1.15 convention:
                #   theta = polar angle, phi = azimuthal angle
                Y_complex = sph_harm_y(l, m_abs, THETA, PHI)  # (n_theta, n_phi) complex

                if m == 0:
                    Y_real = Y_complex.real
                elif m > 0:
                    Y_real = ((-1) ** m) * np.sqrt(2.0) * Y_complex.real
                else:  # m < 0
                    Y_real = ((-1) ** m_abs) * np.sqrt(2.0) * Y_complex.imag

                # Integration: a_lm = ∫ I(θ,φ) Y_lm(θ,φ) dΩ
                a_lm = float(np.sum(I * Y_real * d_omega))
                self.sh_coefficients[(l, m)] = a_lm

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def rms_nonuniformity(self) -> float:
        """
        RMS non-uniformity of the irradiance distribution.

            σ_rms = √( Σ_{l≥1} Σ_m a²ₗₘ ) / |a₀₀|

        A perfect uniform illumination gives σ_rms = 0.
        """
        a00 = self.sh_coefficients.get((0, 0), 0.0)
        if a00 == 0.0:
            return float("nan")
        sum_sq = sum(
            v ** 2
            for (l, m), v in self.sh_coefficients.items()
            if l >= 1
        )
        return float(np.sqrt(sum_sq)) / abs(a00)

    @property
    def mode_power(self) -> dict[int, float]:
        """
        Power per spherical-harmonic degree:

            σ_l = √( Σ_m a²ₗₘ ) / |a₀₀|
        """
        a00 = self.sh_coefficients.get((0, 0), 0.0)
        result: dict[int, float] = {}
        for l in range(1, self.L_max + 1):
            sum_sq = sum(
                self.sh_coefficients.get((l, m), 0.0) ** 2
                for m in range(-l, l + 1)
            )
            result[l] = float(np.sqrt(sum_sq)) / (abs(a00) if a00 != 0.0 else 1.0)
        return result

    # ------------------------------------------------------------------
    # Step 6 – Azimuthal average of the irradiance
    # ------------------------------------------------------------------

    def azimuthal_average(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Step 6 – Average the irradiance over the azimuthal angle φ.

        Returns
        -------
        theta_centres : ndarray of shape (n_theta,) [rad]
        I_avg : ndarray of shape (n_theta,) [J/sr]
            Mean irradiance as a function of polar angle θ.
        """
        I = np.nan_to_num(self.irradiance_grid, nan=0.0)
        # Count non-NaN cells per row for proper averaging
        nonnan_count = np.sum(~np.isnan(self.irradiance_grid), axis=1)
        nonnan_count = np.where(nonnan_count > 0, nonnan_count, 1)
        return self.theta_centres, I.sum(axis=1) / nonnan_count

    def summary(self) -> str:
        """Return a human-readable statistical summary."""
        theta_vals, I_avg = self.azimuthal_average()
        lines = [
            "=" * 55,
            "  Irradiance Analysis — Summary",
            "=" * 55,
            f"  Total rays                : {self.n_rays_total:,}",
            f"  Rays hitting target       : {self.n_rays_hit:,} ({100*self.n_rays_hit/self.n_rays_total:.1f}%)",
            f"  Energy on target          : {self.total_energy_on_target_J/1e3:.3f} kJ",
            f"  Coupling efficiency       : {100*self.coupling_efficiency:.2f} %",
            f"  Average intensity         : {self.average_intensity_W_m2:.3e} W/m²",
            f"  RMS non-uniformity (σ_rms): {100*self.rms_nonuniformity:.2f} %",
            "",
            "  Mode power σ_l (l=1..5):",
        ]
        mp = self.mode_power
        for l in range(1, min(6, self.L_max + 1)):
            lines.append(f"    l={l}: {100*mp[l]:.3f} %")
        lines.append("=" * 55)
        return "\n".join(lines)
