"""
omega60.beams
=============
Step 1 – Define the OMEGA-60 laser parameters.

The OMEGA laser at the Laboratory for Laser Energetics (LLE),
University of Rochester, is a 60-beam, UV (351 nm), direct-drive ICF
facility.  The 60 beam directions are placed at the vertices of a
*truncated icosahedron* (soccer-ball polyhedron), which gives a
nearly-uniform spherical coverage with the same symmetry group as the
real OMEGA geometry.

References
----------
T. R. Boehly et al., "Initial performance results of the OMEGA laser
system", Opt. Commun. 133 (1997) 495-506.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _truncated_icosahedron_vertices() -> np.ndarray:
    """
    Return the 60 vertices of a truncated icosahedron (normalised to the
    unit sphere).

    Coordinates (Cartesian) before normalisation::

        Type A – 12 vertices: (0, ±1, ±3φ) and even cyclic permutations
        Type B – 24 vertices: (±1, ±(2+φ), ±2φ) and even cyclic permutations
        Type C – 24 vertices: (±2, ±(1+2φ), ±φ) and even cyclic permutations

    where φ = (1 + √5) / 2 is the golden ratio.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0  # golden ratio ≈ 1.618

    verts: list[list[float]] = []

    # Type A – 12 vertices
    for s1 in (1.0, -1.0):
        for s2 in (1.0, -1.0):
            a, b, c = 0.0, s1 * 1.0, s2 * 3.0 * phi
            verts += [[a, b, c], [c, a, b], [b, c, a]]  # 3 cyclic perms

    # Type B – 24 vertices
    for s1 in (1.0, -1.0):
        for s2 in (1.0, -1.0):
            for s3 in (1.0, -1.0):
                a, b, c = s1 * 1.0, s2 * (2.0 + phi), s3 * 2.0 * phi
                verts += [[a, b, c], [c, a, b], [b, c, a]]

    # Type C – 24 vertices
    for s1 in (1.0, -1.0):
        for s2 in (1.0, -1.0):
            for s3 in (1.0, -1.0):
                a, b, c = s1 * 2.0, s2 * (1.0 + 2.0 * phi), s3 * phi
                verts += [[a, b, c], [c, a, b], [b, c, a]]

    v = np.array(verts, dtype=float)           # (60, 3)
    v /= np.linalg.norm(v, axis=1, keepdims=True)  # normalise to unit sphere
    return v


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class Omega60Laser:
    """
    Encapsulates all parameters of the OMEGA 60-beam laser system.

    Parameters
    ----------
    total_energy_J : float
        Total laser energy delivered to the target in joules.
        Default: 23 000 J (23 kJ, a representative OMEGA shot).
    pulse_duration_ns : float
        Full-width of the laser pulse in nanoseconds.  Default: 1.0 ns.
    target_distance_m : float
        Distance from each beam's focusing lens to the target centre, in
        metres.  Default: 1.8 m (f/6 geometry).
    beam_radius_m : float
        1/e² beam radius at the focusing lens in metres.  Default: 0.15 m.
    seed : int or None
        Random seed for reproducible Monte-Carlo sampling.
    """

    # Physical constants
    WAVELENGTH_M: float = 351e-9    # 3ω Nd:glass UV wavelength [m]
    NUM_BEAMS: int = 60

    def __init__(
        self,
        total_energy_J: float = 23_000.0,
        pulse_duration_ns: float = 1.0,
        target_distance_m: float = 1.8,
        beam_radius_m: float = 0.15,
        seed: int | None = None,
    ) -> None:
        self.total_energy_J = float(total_energy_J)
        self.pulse_duration_ns = float(pulse_duration_ns)
        self.target_distance_m = float(target_distance_m)
        self.beam_radius_m = float(beam_radius_m)
        self.seed = seed

        # Derived quantities
        self.energy_per_beam_J: float = self.total_energy_J / self.NUM_BEAMS
        self.pulse_duration_s: float = self.pulse_duration_ns * 1e-9
        self.peak_power_W: float = self.total_energy_J / self.pulse_duration_s
        self.f_number: float = self.target_distance_m / (2.0 * self.beam_radius_m)

        # 60 unit vectors pointing from each lens toward the target (origin).
        # The inward direction = −(outward vertex direction).
        outward = _truncated_icosahedron_vertices()        # (60, 3) unit vectors
        self._beam_axes: np.ndarray = outward              # axis pointing OUT from target
        self._beam_directions: np.ndarray = -outward       # pointing IN toward target

        # Lens positions: each beam lens sits at distance D along its outward axis
        self._lens_positions: np.ndarray = (
            outward * self.target_distance_m
        )  # (60, 3)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def beam_axes(self) -> np.ndarray:
        """Unit vectors (60, 3) pointing *away* from the target for each beam."""
        return self._beam_axes.copy()

    @property
    def beam_directions(self) -> np.ndarray:
        """Unit vectors (60, 3) pointing *toward* the target for each beam."""
        return self._beam_directions.copy()

    @property
    def lens_positions(self) -> np.ndarray:
        """Cartesian positions (60, 3) [m] of each beam's focusing lens."""
        return self._lens_positions.copy()

    @property
    def wavelength_m(self) -> float:
        """Laser wavelength in metres (351 nm UV)."""
        return self.WAVELENGTH_M

    @property
    def frequency_Hz(self) -> float:
        """Laser frequency in Hz."""
        return 3e8 / self.WAVELENGTH_M

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary of the laser parameters."""
        lines = [
            "=" * 55,
            "  OMEGA 60-Beam Laser System — Parameters",
            "=" * 55,
            f"  Number of beams        : {self.NUM_BEAMS}",
            f"  Wavelength             : {self.WAVELENGTH_M * 1e9:.1f} nm",
            f"  Total energy           : {self.total_energy_J / 1e3:.2f} kJ",
            f"  Energy per beam        : {self.energy_per_beam_J:.2f} J",
            f"  Pulse duration         : {self.pulse_duration_ns:.2f} ns",
            f"  Peak power (total)     : {self.peak_power_W / 1e12:.3f} TW",
            f"  Target distance        : {self.target_distance_m:.2f} m",
            f"  Beam radius at lens    : {self.beam_radius_m * 1e2:.1f} cm",
            f"  F-number               : f/{self.f_number:.1f}",
            "=" * 55,
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"Omega60Laser(total_energy_J={self.total_energy_J}, "
            f"pulse_duration_ns={self.pulse_duration_ns}, "
            f"num_beams={self.NUM_BEAMS})"
        )
