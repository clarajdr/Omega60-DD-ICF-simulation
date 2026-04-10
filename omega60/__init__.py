"""
omega60 – OMEGA 60-beam ICF laser simulation package.

Steps implemented:
  1. Define laser parameters  (beams.py)
  2. Sample rays               (rays.py)
  3. Ray–target collision      (target.py)
  4. Collect statistics        (analysis.py)
  5. Spherical-harmonic decomposition (analysis.py)
  6. Average intensity         (analysis.py)
  7. Sample rays for ARWEN     (arwen.py)
"""

from .beams import Omega60Laser
from .rays import sample_rays
from .target import SphericalTarget
from .analysis import IrradianceAnalysis
from .arwen import ArwenRayWriter

__all__ = [
    "Omega60Laser",
    "sample_rays",
    "SphericalTarget",
    "IrradianceAnalysis",
    "ArwenRayWriter",
]
