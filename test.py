import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


#Parameters
BEAM_RADIUS = 0.14 #meters
TARGET_RADIUS = 0.001 #meters 
CHAMBER_RADIUS = 1.65 #meters 
WINDOW_SIZE = 0.3 #meters 0.3 initially

#Truncated icosahedron parameter
a = (1. + np.sqrt(5.)) / 2


def sph_to_cart(r, theta, phi):
    
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])

def cart_to_sph(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.pi + np.arctan2(y, x)
    return np.array([r, theta, phi])


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
        r, theta, phi = cart_to_sph(x, y, z)
    # Arrondir à 10 décimales pour éviter les erreurs de flottants
        nodes_sph.append([CHAMBER_RADIUS,theta,phi])

# np.unique trouvera alors exactement 60 points
    return np.unique(nodes_sph, axis=0)



def generate_omega60_geometry():
    omega_beams= generare_Truncated_icosahedron()
    all_beams=[]
    beam_id=1

    for r, theta, phi in omega_beams:

        x, y, z= sph_to_cart(CHAMBER_RADIUS, theta, phi) #laser ports are on the chamber
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
    f_dist = compute_focal_distance(defocus_mm=2.0) 
    return compute_focus_position(beam_row["center"], f_dist)


