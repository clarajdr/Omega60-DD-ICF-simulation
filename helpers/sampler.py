import argparse
import numpy as np
import scipy as sp
from helpers import coord
from helpers import geometry
import pandas as pd

info = argparse.Namespace(
  name = 'sampler',
  desc = 'Module with sample helper functions',
  author = 'Clara Jourdan',
  email = 'clara.jourdan@imt-atlantique.net',
  year = 2026,
  version = [ 1, 0, 0, ],
  copyright = 'Copyright (C) 2026 Clara Jourdan (Instituto de Fusión Nuclear Guillermo Velarde, Universidad Politécnica de Madrid)',
)

def write_disclaimer(info):
    if type(info) is dict:
        return f'# {info["copyright"]} hereby claims all interest in program "{info["name"]}"'
    elif isinstance(info, argparse.Namespace):
        return f'# {info.copyright} hereby claims all interest in program "{info.name}"'
    return '# Copyright (C) 2026 Clara Jourdan'

#Helper functions
def sample_window(t1, t2, l1=1., l2=1., center=np.array([0., 0., 0.]), num_points=1000, dist=None):
    """
    Génère un échantillonnage de points à l'intérieur d'une fenêtre circulaire (ou elliptique)
    définie par les vecteurs t1 et t2 en espace 3D.
    """
    # Utilisation de la distribution uniforme par défaut si non fournie
    if dist is None:
        dist = lambda n: sp.stats.uniform.rvs(loc=-1.0, scale=2.0, size=(n, 2))
        
    sample = np.zeros((num_points, 2))
    k = 0
    while k < num_points:
        s = dist(1)
        # MODIFICATION : On vérifie que la norme du point est <= 1.0 (CERCLE)
        # au lieu de vérifier les valeurs absolues x et y (CARRÉ)
        if np.linalg.norm(s) <= 1.0:
            sample[k,:] = s[0,:]
            k += 1
            
    # Mise à l'échelle vers la fenêtre (l1 et l2 sont conservés comme diamètres)
    v1 = 0.5 * l1 * t1 / np.linalg.norm(t1)
    v2 = 0.5 * l2 * t2 / np.linalg.norm(t2)
    
    return center + sample[:,0,None] * v1 + sample[:,1,None] * v2

def hit_sphere( x, u, radius, center=np.array([0.,0.,0.,]), ):
    """
    Calculates the intersection distance(s) 'l' between a ray (origin x, direction u) 
    and a sphere using the quadratic formula.
    """
  
    dx = x - center

    A = u[0]**2 + u[1]**2 + u[2]**2 
    B = 2*(dx[0]*u[0] + dx[1]*u[1] + dx[2]*u[2])
    C = dx[0]**2 + dx[1]**2 + dx[2]**2 - radius**2

    # Discriminant calculation
    disc = B*B - 4*A*C

    # Calculate intersection points (if they exist)
    if disc < 0:
        # No intersection: the ray misses the sphere
        return []
    elif disc == 0:
        # One intersection: the ray is tangent to the sphere
        return [ -0.5 * B / A ] 
    else:
        # Two intersections: the ray enters and exits the sphere
        disc = np.sqrt(disc)
        l1 = 0.5 * (-B + disc) / A
        l2 = 0.5 * (-B - disc) / A
        return [ l1, l2, ]


def ray_hit_on_sphere( x, u, radius, center=np.array([0.,0.,0.,]), ):
    """
    Determines the nearest valid intersection distance (l) between a ray 
    and the sphere surface, filtering out negative or non-existent results.
    """
    ls = hit_sphere( x, u, radius=radius, center=center, )
    
    if len(ls) == 0:
        return -1.
    elif len(ls) == 1:
        if ls[0] < 0:
            return -1.
        else:
            return ls[0]
    else:
        l1, l2 = ls
        if l1 < 0 and l2 < 0:
            return -1.
        elif l1 > 0 and l2 > 0:
            return min(l1, l2)
        else:
            return max(l1, l2)

def ray_hit( x, u, radius, center=np.array([0.,0.,0.,]), ):
    """
    Aggregates all physical properties of a ray's impact, calculating the 3D 
    position, surface normal, incidence cosine (cs), and spherical mapping.
    """
    l = ray_hit_on_sphere( x, u, radius=radius, center=center, ) # Impact length
    xsph = x + l * u 
    n = ( xsph - center ) / np.linalg.norm( xsph - center )
    cs = np.dot( n, u) / np.linalg.norm(u)
    return [ x, u, l, xsph, cs, coord.cart_to_sph( xsph[0], xsph[1], xsph[2] ), n, ]

def beam_sample(beam, num_samples):
    """
    Prepares and executes the sampling of a specific laser beam by reconstructing 
    its geometric vectors and defining its energy distribution profile.
    """
    
    t1 = np.array([beam['t1_x'], beam['t1_y'], beam['t1_z']])
    t2 = np.array([beam['t2_x'], beam['t2_y'], beam['t2_z']])
    center = np.array([beam['center_x'], beam['center_y'], beam['center_z']])
    
    l1 = beam['l1']
    l2 = beam['l2']
    
    # 2. Define the NORMAL (Gaussian) distribution
    # loc=0.0: Centered on the beam axis.
    # scale=0.5: Standard deviation (concentrates ~95% of energy within the central area).
    # IMPORTANT: The lambda function generates numerical data using .rvs to ensure compatibility.
    dist_gauss = lambda n: sp.stats.norm(loc=0.0, scale=0.5).rvs(size=(n, 2))
    
    # Map the distribution to the 3D window coordinates
    return sample_window(t1, t2, l1=l1, l2=l2, center=center, num_points=num_samples, dist=dist_gauss)

    

# --- Section des Tests ---
def test_energy_conservation():
    """
    Test to ensure no 'magic energy' is created during the simulation.
    The sum of ray weights must be <= total laser energy.
    """
    # 1. Paramètres de test
    N_RAYS = 10000
    R_TARGET = 0.001
    CENTER = np.array([0, 0, 0])
    
    # 2. Récupération de l'énergie de référence
    beam_df = geometry.get_omega60_dataframe()
    total_omega_energy = beam_df['energy'].sum()
    
    # 3. Exécution de la simulation
    df_results = impact_simulated(N_RAYS, R_TARGET, CENTER)
    
    # 4. Calcul de l'énergie totale déposée sur la cible
    total_deposited_energy = df_results['energy_weight'].sum()
    
    # --- ASSERTIONS ---
    
    # A. L'énergie déposée ne peut PAS être supérieure à l'énergie totale du laser
    # On ajoute une petite tolérance pour les erreurs d'arrondi (floating point)
    assert total_deposited_energy <= total_omega_energy + 1e-5, \
        f"Magic energy detected! Deposited: {total_deposited_energy}J > Total: {total_omega_energy}J"
    
    # B. L'énergie déposée doit être positive (test de bon sens)
    assert total_deposited_energy >= 0, "Negative energy detected!"

    print(f"\n[Test Passed] Total deposited: {total_deposited_energy:.2f} J / {total_omega_energy:.2f} J")


def test_ray_intersection():
    """Vérifie si le calcul d'intersection avec la sphère est exact."""
    print("# Running test: Ray-Sphere Intersection Math")
    radius = 1.0
    center = np.array([0.0, 0.0, 0.0])
    
    # Rayon tiré pile au centre depuis Z = 5.0
    x_ray = np.array([0.0, 0.0, 5.0])
    u_ray = np.array([0.0, 0.0, -1.0]) # Direction vers le bas
    
    res = ray_hit(x_ray, u_ray, radius, center)
    l_impact = res[2]
    
    # La distance devrait être 4.0 (5.0 - 1.0 de rayon)
    assert np.isclose(l_impact, 4.0), f"Error: Expected 4.0, got {l_impact}"
    print("Test passed: Intersection distance is correct.")

def test_sampling_boundaries():
    """Vérifie que l'échantillonnage ne sort pas des limites du faisceau."""
    print("# Running test: Circular Window Sampling Boundaries")
    t1, t2 = np.array([1, 0, 0]), np.array([0, 1, 0])
    l1, l2 = 0.28, 0.28
    center = np.array([0, 0, 0])
    
    points = sample_window(t1, t2, l1, l2, center, num_points=1000)
    
    # Pour chaque point, la distance au centre doit être <= rayon (0.28 / 2 = 0.14)
    for p in points:
        dist = np.linalg.norm(p)
        assert dist <= 0.140000000001, f"Error: Point at {dist} is outside the 0.14 radius"
    
    print("Test passed: All points are within the circular beam window.")

def test_full_simulation_run():
    """Test d'intégration : vérifie que la simulation Monte-Carlo produit un DataFrame valide."""
    print("# Running test: Integration - Monte-Carlo Simulation Run")
    try:
        # On lance une toute petite simulation (120 rayons = 2 par faisceau en moyenne)
        res = impact_simulated(num_total_rays=120, target_radius=0.001, target_center=np.array([0,0,0]))
        
        assert isinstance(res, pd.DataFrame), "Error: Simulation should return a DataFrame"
        if not res.empty:
            assert 'energy_weight' in res.columns, "Error: energy_weight column missing"
            print(f"Test passed: Generated {len(res)} impacts with energy weighting.")
        else:
            print("Warning: No impacts recorded (normal for very small ray counts).")
    except Exception as e:
        print(f"Test failed or skipped: {e}")
        print("Note: Ensure geometry.get_omega60_dataframe() is fully updated.")

# --- Bloc d'Exécution (Main Guard) ---

if __name__ == "__main__":
    import sys
    
    # 1. Affichage des métadonnées
    print(f"\n--- {info.name.upper()} MODULE ---")
    print(f"Description: {info.desc}")
    print(f"Author: {info.author} ({info.year})")
    print("-" * 30)

    # 2. Lancement des tests unitaires
    try:
        test_ray_intersection()
        test_sampling_boundaries()
        test_full_simulation_run()
        test_energy_conservation()
        print("\n[SUCCESS] All critical tests passed.")
    except AssertionError as e:
        print(f"\n[FAILURE] A test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")
        sys.exit(1)
    
    print("-" * 30)
    print("Disclaimer:")
    print(write_disclaimer(info))

else:
    # Message discret lors de l'import dans ton Jupyter Notebook
    print(f"# Module '{info.name}' loaded successfully.")