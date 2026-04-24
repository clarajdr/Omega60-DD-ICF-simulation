import argparse
import numpy as np
import math
import pandas as pd
from utils import coord

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

def write_disclaimer(info):
    if type(info) is dict:
        return f'# {info["copyright"]} hereby claims all interest in program "{info["name"]}"'
    elif isinstance(info, argparse.Namespace):
        return f'# {info.copyright} hereby claims all interest in program "{info.name}"'
    return '# Copyright (C) 2026 Clara Jourdan' 

#Parameters
BEAM_RADIUS = 0.14 #meters
TARGET_RADIUS = 0.001 #meters 
CHAMBER_RADIUS = 1.65 #meters 
WINDOW_SIZE = 0.3 #meters 0.3 initially

#Truncated icosahedron parameter
a = (1. + np.sqrt(5.)) / 2

#Helper functions to define omega60 geometry

def generare_Truncated_icosahedron():
    perm = np.array([
      [ 1., 1., 1., ],
             [ 1., 1., -1., ],
            [ 1., -1., 1., ],
            [ 1., -1., -1., ],
             [ -1., 1., 1., ],
            [ -1., 1., -1., ],
             [ -1., -1., 1., ],
             [ -1., -1., -1., ], ] )
    
    ss = np.array([
      [      0,    1, 3.*a, ],
      [     1., 3.*a,   0., ],
      [   3.*a,   0.,   1., ],
      [     1., 2. + a, 2. * a, ],
      [ 2. + a, 2. * a, 1., ],
      [ 2. * a, 1., 2. + a, ],
      [      a, 2., 2.*a + 1., ],
      [     2., 2.*a + 1., a, ],
      [ 2.*a + 1., a, 2., ]])
    
    nodes = []
    nodes_sph=[]
    
    for si in ss: 
        for p in perm : 
            nodes.append(si*p)
    
    nodes_cart = np.unique(np.round(nodes , 10), axis=0)

    # convert the cartesian coordonates of the points in the sphere into spherical coordonates
    for x, y, z in nodes_cart:
        r, theta, phi = coord.cart_to_sph(x, y, z)
    # Arrondir à 10 décimales pour éviter les erreurs de flottants
        nodes_sph.append([CHAMBER_RADIUS,theta,phi])

    return np.unique(nodes_sph, axis=0)



def generate_omega60_geometry():
    omega_beams= generare_Truncated_icosahedron()
    all_beams=[]
    beam_id=1

    for r, theta, phi in omega_beams:

        x, y, z= coord.sph_to_cart(CHAMBER_RADIUS, theta, phi) #laser ports are on the chamber
        theta= math.degrees(theta)
        phi=math.degrees(phi)


        all_beams.append({'id': f'Beam_{beam_id:02d}',
                'theta_deg':theta,
                'phi_deg': phi % 360, # Wrap angle to [0, 360)
                'X': x,
                'Y': y,
                'Z': z,
                'color': 'blue' if theta < 90 else 'red'}) # Blue = Northern Hemisphere, Red = Southern)
        beam_id +=1
    return pd.DataFrame(all_beams)

df_omega= generate_omega60_geometry()
print(df_omega.head())
print(len(df_omega))

def compute_focal_distance(defocus_mm=0.0):
    """
    Calculates the focal distance required to achieve a specific beam spot size.
    A positive defocus (in mm) shifts the focal point beyond the target center, 
    widening the irradiation area to improve illumination uniformity.
    """
    # Convert defocus from mm to meters and calculate the effective target radius
    # This accounts for the beam divergence/convergence at the target surface.
    effective_target_radius = TARGET_RADIUS + (defocus_mm / 1000.0) 
    
    # Using Thales' theorem (intercept theorem) to determine the focal length (f_window)
    # relative to the chamber geometry and beam aperture.
    f_window = (WINDOW_SIZE * CHAMBER_RADIUS) / (WINDOW_SIZE - effective_target_radius)
    return f_window

def compute_focus_position(port_center, f_window):
    """
    Computes the 3D coordinates of the focal point along the beam's optical axis.
    """
    # Normal unit vector pointing from the port center toward the chamber origin (0,0,0)
    propagation_vector = port_center / np.linalg.norm(port_center)
    
    # Position the focal point along the optical axis at distance f_window
    focal_point_pos = port_center - (propagation_vector * f_window)
    return focal_point_pos

def assign_beam_focus(beam_row):
    """
    Helper function to apply a standardized defocus to a specific beam port.
    A 2.0mm defocus is typically used to over-fill the target and reduce intensity peaks.
    """
    f_dist = compute_focal_distance(defocus_mm=2) 
    return compute_focus_position(beam_row["center"], f_dist)

def get_omega60_dataframe():
    DEFAULT_ENERGY = 500.0     # Default beam energy [Joules]
    WAVELENGTH = 351e-9  

    omega_config = generare_Truncated_icosahedron()
    all_data = []
    beam_count = 1

    for r, theta, phi in omega_config:

        x, y, z = coord.sph_to_cart(CHAMBER_RADIUS, theta, phi) # laser ports are on the chamber
        theta_rad = theta
        phi_rad = phi % 60 # Note: Gardé tel quel selon ton code

        nx = np.sin(theta_rad) * np.cos(phi_rad)
        ny = np.sin(theta_rad) * np.sin(phi_rad)
        nz = np.cos(theta_rad)
        normal_u = np.array([nx, ny, nz])

        center = normal_u * CHAMBER_RADIUS

        # --- 3. Local Tangent Basis Calculation (t1, t2) ---
        t1 = np.array([-np.sin(phi_rad), np.cos(phi_rad), 0.0]) 
            
        # Singularity handling at the poles
        if np.linalg.norm(t1) < 1e-6: 
            t1 = np.array([1, 0, 0])
        t1 /= np.linalg.norm(t1)
            
        t2 = np.cross(normal_u, t1)
        t2 /= np.linalg.norm(t2)
        
        # --- CALCUL DU FOCUS INTÉGRÉ ---
        # On utilise tes fonctions pour un defocus standard de 2.0mm
        f_dist = compute_focal_distance(defocus_mm=2.0) 
        focus = compute_focus_position(center, f_dist) 
        
        all_data.append({
                'port': f'OMEGA_J{beam_count:02d}',
                'subwindow': 0,
                'LAT_rad': theta_rad,
                'LONG_rad': phi_rad,
                'color': 'blue' if theta < math.pi/2 else 'red',
                'energy': DEFAULT_ENERGY,
                'wavelength': WAVELENGTH,
                'center': center,
                'normal': normal_u,
                't1': t1,
                't2': t2,
                'beam_radius': BEAM_RADIUS,
                'focus': focus  # La variable n'est plus None
            })
        beam_count += 1

    return pd.DataFrame(all_data)

#TESTING
def test_omega_properties():
    print("# Running test: Omega60 basic properties")
    df = generate_omega60_geometry()
    
    # Test 1 : Nombre de faisceaux
    assert len(df) == 60, f"Error: Should be 60 beams, found {len(df)}"
    
    # Test 2 : Rayon de la chambre
    for _, row in df.iterrows():
        dist = np.linalg.norm(row['center'])
        assert np.isclose(dist, CHAMBER_RADIUS), "Error: Beam port not on chamber wall"
    
    print("Test passed: 60 beams correctly placed on the sphere.")


def test_focal_logic():
    print("# Running test: Focal distance calculation")
    f_dist = compute_focal_distance(defocus_mm=2)
    # Si defocus=0, le foyer doit être proche du centre (calcul géométrique)
    assert f_dist > CHAMBER_RADIUS, "Error: Focal point must be beyond the port"
    print("Test passed: Focal logic is consistent.")


# Runing the code 
# --- Main Guard ---
if __name__ == "__main__":
    print(f"--- Running tests for module: {info.name} ---")
    test_omega_properties()
    test_focal_logic()
    
    # Exemple d'utilisation (uniquement si on lance le script en direct)
    df_sample = generate_omega60_geometry()
    print("\nSample Geometry Data:")
    print(df_sample.head())
    print("\nAll tests successful.")
else:
    # Ce disclaimer s'affiche lors de l'import
    import sys
    print(f"# Module {info.name} loaded (Author: {info.author})", file=sys.stderr)


