import numpy as np 
import pandas as pd 
import argparse
from scipy.special import sph_harm
from helpers import geometry
from helpers import sampler
from helpers import coord, utils
from helpers import rwhist, collision, rwbeams, sphericalharmonics
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

# Reconstruction error
def reconstruct_histogram(coeffs, theta_edges, phi_edges):
    """
    Étape B : Reconstruit l'histogramme f* à partir des coefficients flm.
    f* = sum(flm * Ylm)
    """
    # 1. Préparation de la grille identique à l'originale
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    phi_centers   = 0.5 * (phi_edges[:-1] + phi_edges[1:])
    mt, mp = np.meshgrid(theta_centers, phi_centers, indexing='ij')

    # 2. Somme pondérée des harmoniques
    hist_star = np.zeros_like(mt, dtype=complex)
    for (l, m), flm in coeffs.items():
        Ylm = sph_harm(m, l, mp, mt)
        hist_star += flm * Ylm
    
    # On retourne la partie réelle car l'intensité physique est réelle
    return hist_star.real

def compute_reconstruction_error(hist_orig, hist_reconstructed):
    """
    Étape C : Calcule la carte d'erreur relative point par point.
    e_ij = sqrt(|1 - f*/f|^2)
    """
    # On évite la division par zéro si une case de l'histogramme est vide
    safe_hist = np.where(hist_orig == 0, np.nan, hist_orig)
    
    # Application de la formule du schéma
    error_map = np.sqrt(np.abs(1 - hist_reconstructed / safe_hist)**2)
    
    return error_map

def analyze_model_fidelity(hist, theta_edges, phi_edges, Lmax=6):
    """Calcule tout le modèle et renvoie les objets sans afficher de texte."""
    # A. Décomposition
    coeffs = compute_sph_coeffs_A(hist, theta_edges, phi_edges, Lmax)
    
    # B. Reconstruction
    hist_star = reconstruct_histogram(coeffs, theta_edges, phi_edges)
    
    # C. Calcul de l'erreur point par point
    e_ij = compute_reconstruction_error(hist, hist_star)
    
    # D. Calcul des indicateurs demandés par le schéma[cite: 3]
    metrics = {
        'mean_error': np.nanmean(e_ij) * 100,
        'max_error': np.nanmax(e_ij) * 100,
        'hist_star': hist_star,
        'error_map': e_ij
    }
    return metrics


# TESTING


def test_sph_constant():
    print("# Running test 1: f(theta, phi) = 1")
    # Création d'une grille test 20x20
    n_theta, n_phi = 20, 20
    t_edges = np.linspace(0, np.pi, n_theta + 1)
    p_edges = np.linspace(0, 2*np.pi, n_phi + 1)
    
    # f(theta, phi) = 1 partout
    hist = np.ones((n_theta, n_phi))
    
    coeffs = compute_sph_coeffs_A(hist, t_edges, p_edges, Lmax=2)
    
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
    
    # f(theta, phi) = cos(theta)^2
    hist = np.cos(mt)**2
    
    # On monte jusqu'à Lmax=3 pour bien voir la parité
    coeffs = compute_sph_coeffs_A(hist, t_edges, p_edges, Lmax=3)
    
    for (l, m), val in coeffs.items():
        magnitude = abs(val)
        
        # 1. Test de zonalité : si m != 0, le coeff doit être nul
        if m != 0:
            assert np.isclose(magnitude, 0, atol=1e-10), \
                f"Error: m={m} (l={l}) should be 0 for zonal function"
            
        # 2. Test de parité : si l est impair (1, 3), le coeff doit être nul
        if l % 2 != 0:
            assert np.isclose(magnitude, 0, atol=1e-10), \
                f"Error: degree l={l} is odd, all its m-components should be 0"
    
    # Vérification spécifique des composantes m pour l=1 et l=3 (incluant m négatifs)
    # C'est ce que tu voulais vérifier avec l=-1 ou l=-3 (en parlant de m)
    for m_val in [-1, 1]:
        assert np.isclose(abs(coeffs[(1, m_val)]), 0, atol=1e-10), f"Error: (l=1, m={m_val}) should be 0"
    
    for m_val in [-3, -2, -1, 1, 2, 3]:
        assert np.isclose(abs(coeffs[(3, m_val)]), 0, atol=1e-10), f"Error: (l=3, m={m_val}) should be 0"

    print("Test 2 passed: cos(theta)^2 is perfectly zonal (m=0) and even (l=0, 2).")

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
    
    coeffs = compute_sph_coeffs_A(hist, t_edges, p_edges, Lmax=1)
    
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
    
    coeffs = compute_sph_coeffs_A(hist, t_edges, p_edges, Lmax=2)
    
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

def test_sph_delta_single_mode():
    """
    Test 5 : Delta Test (Single Mode)
    On définit f(theta, phi) = Y_{l0, m0}(theta, phi).
    L'algorithme doit extraire c_{l0, m0} = 1.0 et 0.0 ailleurs.
    """
    print("# Running test 5: Delta Test (Single Mode Y_{2,1})")
    
    # Résolution de la grille
    n_theta, n_phi = 60, 60
    t_edges = np.linspace(0, np.pi, n_theta + 1)
    p_edges = np.linspace(0, 2*np.pi, n_phi + 1)
    
    t_centers = 0.5 * (t_edges[:-1] + t_edges[1:])
    p_centers = 0.5 * (p_edges[:-1] + p_edges[1:])
    mt, mp = np.meshgrid(t_centers, p_centers, indexing='ij')

    # 1. Définition du mode cible (l=2, m=1)
    l0, m0 = 2, 1
    
    # 2. Génération du signal f = Y_{2,1}
    # Note: sph_harm utilise l'ordre (m, l, phi, theta)
    hist = sph_harm(m0, l0, mp, mt)

    # 3. Calcul des coefficients via la Méthode A
    Lmax = 3
    coeffs = compute_sph_coeffs_A(hist, t_edges, p_edges, Lmax)

    # 4. Validation
    target_key = (l0, m0)
    
    for (l, m), val in coeffs.items():
        magnitude = abs(val)
        if (l, m) == target_key:
            # Le mode cible doit valoir 1.0
            assert np.isclose(magnitude, 1.0, atol=1e-3), \
                f"Error: Mode ({l},{m}) should be 1.0, found {magnitude:.5f}"
        else:
            # Tous les autres modes doivent être nuls
            assert np.isclose(magnitude, 0.0, atol=1e-3), \
                f"Error: Mode ({l},{m}) should be 0.0, found {magnitude:.5e}"

    print(f"Test 5 passed: Only mode ({l0},{m0}) is 1.0, others are ~0.")

if __name__ == '__main__':
    utils.show_message()
    sys.exit(utils.run_test(info, globals()))
else:
    print(utils.write_disclaimer(info))