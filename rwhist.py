# rwhist.py
import numpy as np
import pandas as pd

# ON NE MET QUE DES FONCTIONS (RECETTES)
def compute_omega_coeffs(hist_2d, theta_edges, phi_edges, Lmax):
    # (Le code de calcul ici...)
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    phi_centers   = 0.5 * (phi_edges[:-1] + phi_edges[1:])
    mt, mp = np.meshgrid(theta_centers, phi_centers, indexing='ij')
    
    dphi = phi_edges[1:] - phi_edges[:-1]
    dtheta_cos = np.cos(theta_edges[:-1]) - np.cos(theta_edges[1:])
    S = np.outer(dtheta_cos, dphi)
    
    total_energy = np.sum(hist_2d * S)
    coeffs = {}
    for l in range(Lmax + 1):
        for m in range(-l, l + 1):
            Ylm = sph_harm(m, l, mp, mt)
            alm = np.sum(hist_2d * np.conjugate(Ylm) * S)
            coeffs[(l, m)] = alm / total_energy
    return coeffs

def print_omega_results(coeffs):
    # (Le code d'affichage ici...)
    a_00 = abs(coeffs[(0,0)])
    for (l, m), val in sorted(coeffs.items()):
        ratio = abs(val) / a_00
        if l > 0 and ratio > 0.0001:
            print(f"{l:3d} {m:4d} {ratio:35.2%}")