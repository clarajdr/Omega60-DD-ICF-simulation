"""
tests/test_simulation.py
========================
Unit tests for the omega60 simulation package.

Run with::

    python -m pytest tests/ -v
"""

import math
import numpy as np
import pytest

from omega60.beams import Omega60Laser, _truncated_icosahedron_vertices
from omega60.rays import sample_rays, _build_local_frame, _random_disk_uniform
from omega60.target import SphericalTarget
from omega60.analysis import IrradianceAnalysis
from omega60.arwen import ArwenRayWriter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def laser():
    return Omega60Laser(total_energy_J=23_000.0, pulse_duration_ns=1.0, seed=0)


@pytest.fixture(scope="module")
def target():
    return SphericalTarget(radius_m=450e-6)


@pytest.fixture(scope="module")
def rays(laser):
    return sample_rays(laser, rays_per_beam=500, seed=0)


@pytest.fixture(scope="module")
def hits(target, rays):
    return target.intersect(rays)


@pytest.fixture(scope="module")
def analysis(hits, laser, target):
    return IrradianceAnalysis(hits, laser, target, L_max=6)


# ===========================================================================
# STEP 1 – Beam geometry and parameters
# ===========================================================================

class TestBeamGeometry:
    def test_num_beams(self, laser):
        """Laser must have exactly 60 beams."""
        assert laser.NUM_BEAMS == 60

    def test_beam_directions_shape(self, laser):
        assert laser.beam_directions.shape == (60, 3)

    def test_beam_directions_unit_vectors(self, laser):
        norms = np.linalg.norm(laser.beam_directions, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_beam_axes_unit_vectors(self, laser):
        norms = np.linalg.norm(laser.beam_axes, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_beam_axes_opposite_to_directions(self, laser):
        """beam_axes should be exactly opposite to beam_directions."""
        np.testing.assert_allclose(
            laser.beam_axes + laser.beam_directions,
            np.zeros((60, 3)),
            atol=1e-12,
        )

    def test_lens_positions_at_correct_distance(self, laser):
        dists = np.linalg.norm(laser.lens_positions, axis=1)
        np.testing.assert_allclose(dists, laser.target_distance_m, atol=1e-10)

    def test_energy_per_beam(self, laser):
        assert math.isclose(
            laser.energy_per_beam_J * laser.NUM_BEAMS,
            laser.total_energy_J,
            rel_tol=1e-12,
        )

    def test_f_number(self, laser):
        expected = laser.target_distance_m / (2.0 * laser.beam_radius_m)
        assert math.isclose(laser.f_number, expected, rel_tol=1e-12)

    def test_wavelength(self, laser):
        assert laser.wavelength_m == pytest.approx(351e-9)

    def test_truncated_icosahedron_60_vertices(self):
        v = _truncated_icosahedron_vertices()
        assert v.shape == (60, 3)

    def test_truncated_icosahedron_unit_sphere(self):
        v = _truncated_icosahedron_vertices()
        norms = np.linalg.norm(v, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-12)

    def test_beams_span_sphere(self, laser):
        """Beam directions should span all octants (no hemisphere bias)."""
        d = laser.beam_directions
        assert (d[:, 0] > 0).any() and (d[:, 0] < 0).any()
        assert (d[:, 1] > 0).any() and (d[:, 1] < 0).any()
        assert (d[:, 2] > 0).any() and (d[:, 2] < 0).any()

    def test_summary_contains_60_beams(self, laser):
        s = laser.summary()
        assert "60" in s

    def test_peak_power(self, laser):
        expected = laser.total_energy_J / laser.pulse_duration_s
        assert math.isclose(laser.peak_power_W, expected, rel_tol=1e-12)


# ===========================================================================
# STEP 2 – Ray sampling
# ===========================================================================

class TestRaySampling:
    def test_ray_count(self, rays, laser):
        assert rays["origins"].shape[0] == laser.NUM_BEAMS * 500

    def test_ray_directions_unit_vectors(self, rays):
        norms = np.linalg.norm(rays["directions"], axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_beam_ids_range(self, rays, laser):
        assert rays["beam_ids"].min() == 0
        assert rays["beam_ids"].max() == laser.NUM_BEAMS - 1

    def test_beam_id_counts(self, rays, laser):
        counts = np.bincount(rays["beam_ids"], minlength=laser.NUM_BEAMS)
        assert (counts == 500).all()

    def test_total_energy_correct(self, rays, laser):
        assert math.isclose(
            rays["energies"].sum(),
            laser.total_energy_J,
            rel_tol=1e-10,
        )

    def test_rays_point_inward(self, rays):
        """
        Each ray direction should have a negative dot product with its
        origin (i.e. pointing toward the origin from outside).
        """
        dot = np.einsum("ij,ij->i", rays["origins"], rays["directions"])
        assert (dot < 0).all()

    def test_disk_sampling_inside_unit_circle(self):
        rng = np.random.default_rng(7)
        pts = _random_disk_uniform(10_000, rng)
        radii = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
        assert (radii <= 1.0 + 1e-12).all()

    def test_build_local_frame_orthogonal(self):
        for axis in [
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([1.0, 1.0, 1.0]) / np.sqrt(3),
        ]:
            u, v = _build_local_frame(axis)
            assert abs(np.dot(u, v)) < 1e-12
            assert abs(np.dot(u, axis)) < 1e-12
            assert abs(np.dot(v, axis)) < 1e-12
            assert abs(np.linalg.norm(u) - 1.0) < 1e-12
            assert abs(np.linalg.norm(v) - 1.0) < 1e-12

    def test_reproducible_with_seed(self, laser):
        r1 = sample_rays(laser, rays_per_beam=100, seed=99)
        r2 = sample_rays(laser, rays_per_beam=100, seed=99)
        np.testing.assert_array_equal(r1["origins"], r2["origins"])


# ===========================================================================
# STEP 3 – Ray–target collision
# ===========================================================================

class TestTargetCollision:
    def test_hit_mask_shape(self, hits, rays):
        assert hits["hit_mask"].shape == (rays["origins"].shape[0],)

    def test_hit_xyz_on_sphere(self, hits, target):
        mask = hits["hit_mask"]
        xyz = hits["hit_xyz"][mask]
        r = np.linalg.norm(xyz, axis=1)
        np.testing.assert_allclose(r, target.radius_m, atol=1e-12)

    def test_theta_range(self, hits):
        mask = hits["hit_mask"]
        theta = hits["hit_theta"][mask]
        assert (theta >= 0).all() and (theta <= np.pi + 1e-12).all()

    def test_phi_range(self, hits):
        mask = hits["hit_mask"]
        phi = hits["hit_phi"][mask]
        assert (phi >= 0).all() and (phi < 2.0 * np.pi + 1e-12).all()

    def test_all_rays_hit_target(self, hits):
        """
        Because every ray points directly at the target centre and the
        beams converge from outside, virtually all rays should hit.
        """
        mask = hits["hit_mask"]
        fraction = mask.sum() / mask.size
        assert fraction > 0.95, f"Only {100*fraction:.1f}% of rays hit target"

    def test_hit_energies_positive(self, hits):
        mask = hits["hit_mask"]
        assert (hits["hit_energies"][mask] > 0).all()

    def test_non_hit_energy_is_zero(self, hits):
        mask = ~hits["hit_mask"]
        if mask.any():
            assert (hits["hit_energies"][mask] == 0).all()

    def test_hit_beam_ids_valid(self, hits, laser):
        mask = hits["hit_mask"]
        b = hits["hit_beam_ids"][mask]
        assert (b >= 0).all() and (b < laser.NUM_BEAMS).all()

    def test_invalid_radius(self):
        with pytest.raises(ValueError):
            SphericalTarget(radius_m=-1.0)

    def test_chief_ray_hits_centre(self):
        """A ray along the z-axis hits the target at (0,0,R)."""
        target = SphericalTarget(radius_m=1.0)
        rays_dict = {
            "origins": np.array([[0.0, 0.0, 10.0]]),
            "directions": np.array([[0.0, 0.0, -1.0]]),
            "energies": np.array([1.0]),
            "beam_ids": np.array([0], dtype=np.int32),
        }
        result = target.intersect(rays_dict)
        assert result["hit_mask"][0]
        np.testing.assert_allclose(result["hit_xyz"][0], [0, 0, 1.0], atol=1e-12)
        assert result["hit_theta"][0] == pytest.approx(0.0, abs=1e-10)

    def test_ray_missing_target(self):
        """A ray that passes well outside the target should not hit."""
        target = SphericalTarget(radius_m=1.0)
        rays_dict = {
            "origins": np.array([[0.0, 10.0, 10.0]]),
            "directions": np.array([[0.0, 0.0, -1.0]]),
            "energies": np.array([1.0]),
            "beam_ids": np.array([0], dtype=np.int32),
        }
        result = target.intersect(rays_dict)
        assert not result["hit_mask"][0]


# ===========================================================================
# STEPS 4–6 – Statistical analysis
# ===========================================================================

class TestIrradianceAnalysis:
    def test_total_energy_on_target(self, analysis, laser):
        assert analysis.total_energy_on_target_J <= laser.total_energy_J

    def test_coupling_efficiency_range(self, analysis):
        assert 0.0 <= analysis.coupling_efficiency <= 1.0

    def test_rms_nonuniformity_finite(self, analysis):
        sigma = analysis.rms_nonuniformity
        assert np.isfinite(sigma)
        assert sigma >= 0.0

    def test_sh_coefficients_all_degrees(self, analysis):
        """All (l, m) pairs with 0 ≤ l ≤ L_max should be present."""
        for l in range(analysis.L_max + 1):
            for m in range(-l, l + 1):
                assert (l, m) in analysis.sh_coefficients

    def test_sh_l0_m0_dominant(self, analysis):
        """The l=0, m=0 coefficient should be the largest (uniform component)."""
        a00 = abs(analysis.sh_coefficients[(0, 0)])
        for (l, m), v in analysis.sh_coefficients.items():
            if l > 0:
                assert a00 >= abs(v), f"a_00={a00:.3e} < a_{l}{m}={abs(v):.3e}"

    def test_azimuthal_average_shape(self, analysis):
        theta, I_avg = analysis.azimuthal_average()
        assert theta.shape == (analysis.n_theta,)
        assert I_avg.shape == (analysis.n_theta,)

    def test_azimuthal_average_nonnegative(self, analysis):
        _, I_avg = analysis.azimuthal_average()
        assert (I_avg >= 0).all()

    def test_average_intensity_positive(self, analysis):
        assert analysis.average_intensity_W_m2 > 0.0

    def test_mode_power_keys(self, analysis):
        mp = analysis.mode_power
        assert set(mp.keys()) == set(range(1, analysis.L_max + 1))

    def test_mode_power_nonnegative(self, analysis):
        for v in analysis.mode_power.values():
            assert v >= 0.0


# ===========================================================================
# STEP 7 – ARWEN output
# ===========================================================================

class TestArwenOutput:
    @pytest.fixture(scope="class")
    def arwen(self, laser, target):
        return ArwenRayWriter(laser, target, n_rays=600, seed=0)

    def test_n_rays(self, arwen, laser):
        n = arwen.rays["origins"].shape[0]
        # Actual count may be rounded down to nearest multiple of 60
        assert n % laser.NUM_BEAMS == 0
        assert n <= 600

    def test_origins_on_launch_sphere(self, arwen):
        r = np.linalg.norm(arwen.rays["origins"], axis=1)
        np.testing.assert_allclose(r, arwen.launch_radius, atol=1e-10)

    def test_directions_unit_vectors(self, arwen):
        norms = np.linalg.norm(arwen.rays["directions"], axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_rays_point_inward(self, arwen):
        dot = np.einsum(
            "ij,ij->i",
            arwen.rays["origins"],
            arwen.rays["directions"],
        )
        assert (dot < 0).all()

    def test_total_energy(self, arwen, laser):
        e_total = arwen.rays["energies"].sum()
        assert math.isclose(e_total, laser.total_energy_J, rel_tol=1e-10)

    def test_write_and_read(self, arwen, tmp_path):
        outfile = tmp_path / "test.ray"
        arwen.write(outfile)
        lines = outfile.read_text().splitlines()
        # First two lines are comments, third is ray count
        assert lines[0].startswith("# ARWEN")
        n_rays_in_file = int(lines[2].strip())
        data_lines = [l for l in lines[3:] if l.strip()]
        assert len(data_lines) == n_rays_in_file

    def test_ray_file_columns(self, arwen, tmp_path):
        outfile = tmp_path / "cols.ray"
        arwen.write(outfile)
        lines = outfile.read_text().splitlines()
        data_line = lines[3].split()
        assert len(data_line) == 9  # x y z dx dy dz E t0 t1

    def test_launch_radius_factor(self, laser, target):
        aw = ArwenRayWriter(laser, target, n_rays=60, launch_radius_factor=2.0, seed=1)
        assert math.isclose(
            aw.launch_radius, 2.0 * target.radius_m, rel_tol=1e-12
        )
