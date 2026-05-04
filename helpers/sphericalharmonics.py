import numpy as np 
import pandas as pd 
import argparse
from scipy.special import sph_harm
from helpers import geometry
from helpers import sampler
from helpers import coord, utils
from helpers import rwhist, collision, rwbeams
import sphericalharmonics as sh
import sys


#Metadata

info = argparse.Namespace(
  name = 'omega geometry',
  desc = 'Module which generates omega60 geometry',
  author = 'Clara Jourdan',
  email = 'clara.jourdan@imt-atlantique.net',
  year = 2026,
  version = [ 1, 0, 0, ],
  copyright = 'Copyright (C) 2026 Clara Jourdan (Instituto de Fusión Nuclear Guillermo Velarde, Universidad Politécnica de Madrid)',
)

def compute_sph_coeffs_A(hist, theta_edges, phi_edges, Lmax):

    # Bin centers
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    phi_centers   = 0.5 * (phi_edges[:-1] + phi_edges[1:])

    # Meshgrid with the same order as hist: (theta, phi)
    mt, mp = np.meshgrid(theta_centers, phi_centers, indexing='ij')
    # mt = theta, mp = phi

    # Area element
    dphi = phi_edges[1:] - phi_edges[:-1]
    dtheta_cos = np.cos(theta_edges[:-1]) - np.cos(theta_edges[1:])
    S = np.outer(dtheta_cos, dphi)   # (n_theta x n_phi)

    # List of (l, m) pairs
    lm = [(l, m) for l in range(Lmax + 1) for m in range(-l, l + 1)]

    def func_coef(l, m):
        # Calculate spherical harmonics
        Ylm = sph_harm(m, l, mp, mt)
        # Compute the integral (unnormalized coefficient)
        flm = np.sum(hist * np.conjugate(Ylm) * S)
    
        return flm 

    return {(l, m): func_coef(l, m) for (l, m) in lm} 

def print_coeffs(coeffs):
    print("\n================ NORMALIZED COEFFICIENTS a_{lm} =================\n")
    print(f"{'l':>3} {'m':>4} {'Re(a_lm)':>15} {'Im(a_lm)':>15} {'|a_lm|':>15} {'|a_lm|/|a_00|':>15}")
    print("-" * 75)
    
    # Use a_00 for normalization (the DC component or average)
    a_00 = coeffs[(0,0)]
    norm = abs(a_00)

    for (l, m), val in sorted(coeffs.items()):
        if norm == 0:
            ratio = float('nan')   # Normalization undefined
        else:
            ratio = abs(val)/norm

        # Format using scientific notation for precision
        print(f"{l:3d} {m:4d} {val.real:15.6e} {val.imag:15.6e} {abs(val):15.6e} {ratio:15.6e}")

    print("\n===================================================================\n")


def analyze_histogram_A(hist, theta_edges, phi_edges, Lmax=6):

    print("\n" + "="*80)
    print("=== SPHERICAL HARMONIC ANALYSIS OF THE ACTUAL HISTOGRAM ===")
    print("="*80 + "\n")

    # Compute coefficients using Method A (Exact area integration)
    coeffs = compute_sph_coeffs_A(hist, theta_edges, phi_edges, Lmax)
    
    # Display the results in a formatted table
    print_coeffs(coeffs)

    return coeffs

def compute_sph_coeffs_B(hist, theta_edges, phi_edges, Lmax):
    # 1. Bin centers (point sampling)
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    phi_centers   = 0.5 * (phi_edges[:-1] + phi_edges[1:])
    
    # 2. Angular steps (constant)
    dtheta = theta_edges[1] - theta_edges[0]
    dphi = phi_edges[1] - phi_edges[0]
    
    # 3. Meshgrid for vectorized calculations
    mt, mp = np.meshgrid(theta_centers, phi_centers, indexing='ij')

    # 4. AREA ELEMENT METHOD B (Sine approximation)
    # Instead of cos(t1) - cos(t2), we use sin(t_center) * dt
    S = np.sin(mt) * dtheta * dphi 

    # 5. Calculation of coefficients
    lm = [(l, m) for l in range(Lmax + 1) for m in range(-l, l + 1)]
    
    def func_coef(l, m):
        # Calculate spherical harmonics
        Ylm = sph_harm(m, l, mp, mt)
        # Compute the integral (unnormalized coefficient)
        flm = np.sum(hist * np.conjugate(Ylm) * S)
        # Normalization factor (optional: integral of the function over the surface)
        wl = np.sum(hist * S) 
        return flm 

    return {(l, m): func_coef(l, m) for (l, m) in lm}

def print_coeffs(coeffs):
    print("\n================ NORMALIZED COEFFICIENTS f_{lm} =================\n")
    # Headers for the table
    print(f"{'l':>3} {'m':>4} {'Re(f_lm)':>15} {'Im(f_lm)':>15} {'|f_lm|':>15} {'|f_lm|/|f_00|':>15}")
    print("-" * 75)
    
    # Use f_00 for normalization (representing the average or DC component)
    f_00 = coeffs[(0,0)]
    norm = abs(f_00)

    # Iterate through sorted coefficients
    for (l, m), val in sorted(coeffs.items()):
        if norm == 0:
            ratio = float('nan')   # Normalization undefined
        else:
            # Calculate the ratio of the current coefficient's magnitude to f_00
            ratio = abs(val)/norm

        # Print formatted values in scientific notation
        print(f"{l:3d} {m:4d} {val.real:15.6e} {val.imag:15.6e} {abs(val):15.6e} {ratio:15.6e}")

    print("\n===================================================================\n")

def analyze_histogram_B(hist, theta_edges, phi_edges, Lmax=6):

    print("\n" + "="*80)
    print("=== SPHERICAL HARMONIC ANALYSIS OF THE ACTUAL HISTOGRAM ===")
    print("="*80 + "\n")

    # Compute coefficients using Method B (Point sampling/Sine approximation)
    coeffs = compute_sph_coeffs_B(hist, theta_edges, phi_edges, Lmax)
    
    # Display the results in a formatted table
    print_coeffs(coeffs)

    return coeffs


# TESTING


def test_sph_constant():
    print("# Running test 1: f(theta, phi) = 1")
    # Création d'une grille test 20x20
    n_theta, n_phi = 20, 20
    t_edges = np.linspace(0, np.pi, n_theta + 1)
    p_edges = np.linspace(0, 2*np.pi, n_phi + 1)
    
    # f(theta, phi) = 1 partout
    hist = np.ones((n_theta, n_phi))
    
    coeffs = sh.compute_sph_coeffs_A(hist, t_edges, p_edges, Lmax=2)
    
    # Pour une fonction constante, seul le mode (0,0) doit être non nul
    a00 = abs(coeffs[(0, 0)])
    a10 = abs(coeffs[(1, 0)])
    
    assert a00 > 0, "Error: a00 should be non-zero for a constant function"
    assert np.isclose(a10, 0, atol=1e-10), f"Error: a10 should be 0, found {a10}"
    print("Test 1 passed: Constant function only has a (0,0) component.")

def test_sph_parity_zonal():
    print("# Running test 2: f(theta, phi) = cos(theta)^2")
    n_t, n_p = 30, 30
    t_edges = np.linspace(0, np.pi, n_t + 1)
    p_edges = np.linspace(0, 2*np.pi, n_p + 1)
    t_centers = 0.5 * (t_edges[:-1] + t_edges[1:])
    p_centers = 0.5 * (p_edges[:-1] + p_edges[1:])
    mt, mp = np.meshgrid(t_centers, p_centers, indexing='ij')
    
    # f(theta, phi) = cos(theta)^2 (indépendant de phi, donc zonal)
    hist = np.cos(mt)**2
    
    coeffs = sh.compute_sph_coeffs_A(hist, t_edges, p_edges, Lmax=3)
    
    # 1. Indépendant de phi => m=0 uniquement
    for (l, m), val in coeffs.items():
        if m != 0:
            assert np.isclose(abs(val), 0, atol=1e-10), f"Error: m={m} should be 0 for zonal function"
            
    # 2. Fonction paire (cos^2) => l doit être pair (l=0, 2)
    a10 = abs(coeffs[(1, 0)])
    a30 = abs(coeffs[(3, 0)])
    assert np.isclose(a10, 0, atol=1e-10), "Error: l=1 should be 0 for even function"
    assert np.isclose(a30, 0, atol=1e-10), "Error: l=3 should be 0 for even function"
    
    print("Test 2 passed: cos(theta)^2 is correctly identified as zonal and even.")

def test_sph_linear_combination():
    print("# Running test 3: Linear combination a00*Y00 + a10*Y10")
    n_t, n_p = 40, 40
    t_edges = np.linspace(0, np.pi, n_t + 1)
    p_edges = np.linspace(0, 2*np.pi, n_p + 1)
    t_centers = 0.5 * (t_edges[:-1] + t_edges[1:])
    p_centers = 0.5 * (p_edges[:-1] + p_edges[1:])
    mt, mp = np.meshgrid(t_centers, p_centers, indexing='ij')
    
    # On construit f = 1.0*Y00 + 0.5*Y10
    y00 = sph_harm(0, 0, mp, mt).real
    y10 = sph_harm(0, 1, mp, mt).real
    hist = 1.0 * y00 + 0.5 * y10
    
    coeffs = sh.compute_sph_coeffs_A(hist, t_edges, p_edges, Lmax=1)
    
    # On vérifie que l'on retrouve les poids (normalisés par l'élément de surface)
    # Note: flm dans le code est l'intégrale brute sum(hist * conj(Ylm) * S)
    assert abs(coeffs[(0, 0)]) > abs(coeffs[(1, 1)]), "Error: (0,0) should dominate (1,1)"
    print("Test 3 passed: Linear combination coefficients identified.")

def test_sph_real_property():
    print("# Running test 4: Real function property a_{l,m} = (-1)^m * conj(a_{l,-m})")
    n_t, n_p = 20, 20
    t_edges = np.linspace(0, np.pi, n_t + 1)
    p_edges = np.linspace(0, 2*np.pi, n_p + 1)
    
    # Génération d'un signal réel aléatoire
    hist = np.random.rand(n_t, n_p) 
    
    coeffs = sh.compute_sph_coeffs_A(hist, t_edges, p_edges, Lmax=2)
    
    # Vérification de la propriété pour l=2, m=1
    l, m = 2, 1
    alm = coeffs[(l, m)]
    al_minus_m = coeffs[(l, -m)]
    
    # Propriété : a_{l,m} = (-1)^m * conj(a_{l,-m})
    assert np.isclose(alm, ((-1)**m) * np.conj(al_minus_m)), "Error: Real function symmetry not respected"
    print("Test 4 passed: Real function symmetry property verified.")

if __name__ == '__main__':
  utils.show_message()
  sys.exit( utils.run_test( info, globals(), ), )
else:
  print( utils.write_disclaimer(info), )