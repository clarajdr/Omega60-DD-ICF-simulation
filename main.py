"""
main.py – OMEGA 60 DD-ICF laser simulation driver
==================================================

Runs all seven steps described in the project specification:

    1. Define the laser parameters (beams)
    2. Sample rays
    3. Collision of rays with target
    4. Collect statistic results
    5. Analysis with spherical harmonics
    6. Average the intensity
    7. Sample rays for ARWEN

Usage::

    python main.py [--rays-per-beam N] [--l-max L] [--target-radius R_um]
                   [--seed S] [--output-dir DIR] [--no-plots]

All output files are written to ``output/`` by default.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="OMEGA 60-beam DD-ICF laser simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--rays-per-beam", type=int, default=2000,
                   help="Monte-Carlo rays per beam for statistics (×60 total)")
    p.add_argument("--arwen-rays", type=int, default=60_000,
                   help="Total rays to generate for the ARWEN output file")
    p.add_argument("--l-max", type=int, default=10,
                   help="Maximum spherical-harmonic degree for decomposition")
    p.add_argument("--target-radius", type=float, default=450.0,
                   help="Target outer radius [µm]")
    p.add_argument("--total-energy", type=float, default=23_000.0,
                   help="Total laser energy on target [J]")
    p.add_argument("--pulse-duration", type=float, default=1.0,
                   help="Laser pulse duration [ns]")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for Monte-Carlo sampling")
    p.add_argument("--output-dir", type=str, default="output",
                   help="Directory for output files")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip matplotlib figures (useful in headless mode)")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_irradiance_map(analysis, outdir: Path) -> None:
    """Save a 2-D irradiance map (Mollweide projection)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={"projection": "mollweide"})

    # Convert to lon/lat for Mollweide: lon ∈ [-π, π], lat ∈ [-π/2, π/2]
    lon = analysis.phi_centres - np.pi
    lat = np.pi / 2.0 - analysis.theta_centres
    LON, LAT = np.meshgrid(lon, lat)

    I = np.nan_to_num(analysis.irradiance_grid, nan=0.0)
    im = ax.pcolormesh(LON, LAT, I, cmap="plasma", shading="auto")
    fig.colorbar(im, ax=ax, label="Irradiance [J/sr]", shrink=0.7)
    ax.set_title("OMEGA 60 – Target irradiance map")
    ax.set_xlabel("φ (azimuth)")
    ax.set_ylabel("θ (polar angle)")
    ax.grid(True, alpha=0.3)

    fname = outdir / "irradiance_map.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Saved irradiance map → {fname}")


def plot_sh_spectrum(analysis, outdir: Path) -> None:
    """Save a bar chart of mode power σ_l vs. degree l."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mp = analysis.mode_power
    ls = list(mp.keys())
    vals = [100.0 * mp[l] for l in ls]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(ls, vals, color="steelblue", edgecolor="white")
    ax.set_xlabel("Spherical-harmonic degree l")
    ax.set_ylabel("Mode power σ_l  [%]")
    ax.set_title(
        f"OMEGA 60 – SH mode spectrum   "
        f"(σ_rms = {100*analysis.rms_nonuniformity:.2f} %)"
    )
    ax.set_xticks(ls)
    ax.grid(axis="y", alpha=0.4)

    fname = outdir / "sh_spectrum.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Saved SH spectrum     → {fname}")


def plot_azimuthal_average(analysis, outdir: Path) -> None:
    """Save a plot of azimuthally averaged irradiance vs. polar angle."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    theta, I_avg = analysis.azimuthal_average()
    theta_deg = np.degrees(theta)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(theta_deg, I_avg, color="darkorange", lw=1.5)
    ax.axhline(np.nanmean(I_avg), ls="--", color="gray", label="mean")
    ax.set_xlabel("Polar angle θ [deg]")
    ax.set_ylabel("Azimuth-averaged irradiance [J/sr]")
    ax.set_title("OMEGA 60 – Azimuthally averaged irradiance (Step 6)")
    ax.legend()
    ax.grid(alpha=0.3)

    fname = outdir / "azimuthal_average.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Saved azimuthal avg   → {fname}")


def plot_beam_geometry(laser, outdir: Path) -> None:
    """Save a 3-D scatter of beam directions on the unit sphere."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    dirs = laser.beam_directions  # (60, 3) unit vectors pointing inward
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, y, z, c="royalblue", s=30, depthshade=True)
    ax.set_title("OMEGA 60 – Beam directions (unit sphere)")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

    fname = outdir / "beam_geometry.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Saved beam geometry   → {fname}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # STEP 1 – Define laser parameters
    # -----------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("  STEP 1 – Define laser parameters")
    print("=" * 55)

    from omega60 import Omega60Laser, SphericalTarget
    from omega60 import sample_rays, IrradianceAnalysis, ArwenRayWriter

    laser = Omega60Laser(
        total_energy_J=args.total_energy,
        pulse_duration_ns=args.pulse_duration,
        seed=args.seed,
    )
    print(laser.summary())

    target = SphericalTarget(radius_m=args.target_radius * 1e-6)
    print(f"\n  Target: {target}")

    if not args.no_plots:
        plot_beam_geometry(laser, outdir)

    # -----------------------------------------------------------------------
    # STEP 2 – Sample rays
    # -----------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("  STEP 2 – Sample rays")
    print("=" * 55)
    rays = sample_rays(laser, rays_per_beam=args.rays_per_beam, seed=args.seed)
    n_total = rays["origins"].shape[0]
    print(f"  Sampled {n_total:,} rays  ({args.rays_per_beam} per beam × 60 beams)")
    print(f"  Origin range x: [{rays['origins'][:,0].min():.3f}, "
          f"{rays['origins'][:,0].max():.3f}] m")

    # -----------------------------------------------------------------------
    # STEP 3 – Collision of rays with target
    # -----------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("  STEP 3 – Ray–target collision")
    print("=" * 55)
    hits = target.intersect(rays)
    n_hit = int(hits["hit_mask"].sum())
    print(f"  {n_hit:,} / {n_total:,} rays hit the target "
          f"({100*n_hit/n_total:.1f} %)")

    # -----------------------------------------------------------------------
    # STEPS 4, 5, 6 – Statistics, SH decomposition, average intensity
    # -----------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("  STEPS 4–6 – Statistics, SH analysis & average intensity")
    print("=" * 55)
    analysis = IrradianceAnalysis(
        hits, laser, target, L_max=args.l_max
    )
    print(analysis.summary())

    # Save numerical results
    np.save(outdir / "sh_coefficients.npy", analysis.sh_coefficients)
    theta, I_avg = analysis.azimuthal_average()
    np.savetxt(
        outdir / "azimuthal_average.csv",
        np.column_stack([np.degrees(theta), I_avg]),
        delimiter=",",
        header="theta_deg,irradiance_J_per_sr",
        comments="",
    )
    print(f"  SH coefficients  → {outdir / 'sh_coefficients.npy'}")
    print(f"  Azimuthal avg    → {outdir / 'azimuthal_average.csv'}")

    if not args.no_plots:
        plot_irradiance_map(analysis, outdir)
        plot_sh_spectrum(analysis, outdir)
        plot_azimuthal_average(analysis, outdir)

    # -----------------------------------------------------------------------
    # STEP 7 – Sample rays for ARWEN
    # -----------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("  STEP 7 – Sample rays for ARWEN")
    print("=" * 55)
    arwen_writer = ArwenRayWriter(
        laser, target,
        n_rays=args.arwen_rays,
        seed=args.seed,
    )
    print(arwen_writer.summary())
    arwen_path = outdir / "arwen_rays.ray"
    arwen_writer.write(arwen_path)
    print(f"  ARWEN ray file   → {arwen_path}")

    print("\n  Simulation complete.  All outputs in:", outdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
