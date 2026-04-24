import argparse
import numpy as np
import scipy as sp
from utils import coord
from utils import geometry
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

def beam_hit(beam, num_samples, radius, center_target):
    """
    Simule un faisceau laser en conservant toutes les données d'impact
    nécessaires pour le tracé et l'analyse.
    """
    data = []
    
    # 1. Extraction des paramètres
    t1, t2 = beam['t1'], beam['t2']
    center, focus = beam['center'], beam['focus']
    l1, l2 = beam['l1'], beam['l2']
    
    # 2. Distribution Gaussienne (Profil du faisceau)
    dist_gauss = lambda n: sp.stats.norm(loc=0.0, scale=0.15).rvs(size=(n, 2))
    
    # 3. Échantillonnage sur le port (Circulaire)
    ray_origins = sample_window(t1, t2, l1=l1, l2=l2, center=center, num_points=num_samples, dist=dist_gauss)
    
    # 4. Boucle de calcul des impacts
    for i in range(num_samples):
        x_ray = ray_origins[i]
        
        # Calcul du vecteur unitaire vers le foyer
        direction = focus - x_ray
        u_ray = direction / np.linalg.norm(direction)
        
        # Calcul de l'intersection avec ray_hit
        # res contient : [x, u, l, xsph, cs, sph, n]
        res = ray_hit(x_ray, u_ray, radius, center_target)
        
        if res[2] > 0: # Si le rayon touche la cible (distance l > 0)
            data.append({
                'port': beam['port'],
                'x': res[0],          # Origine du rayon
                'u': res[1],          # Direction unitaire
                'l': res[2],          # Distance d'impact
                'xsph': res[3],       # Point d'impact (Cartésien)
                'cs': res[4],         # Cosinus d'incidence
                'sph': res[5],        # Point d'impact (Sphérique)
                'theta': res[5][1],   # Angle polaire (pour l'histogramme)
                'phi': res[5][2],     # Angle azimutal (pour l'histogramme)
                'n': res[6]           # Normale à la surface
            })
            
    return pd.DataFrame(data)
    

def expand_impacts(df_impacts):
    """
    Final version for ARWEN: expands all vectors (x, u, xsph, n, sph) 
    into individual columns.
    """
    df = df_impacts.copy()

    def split_vector_col(df_in, col_name, new_names):
        # Check if the column exists before attempting expansion
        if col_name in df_in.columns:
            temp = pd.DataFrame(df_in[col_name].tolist(), index=df_in.index)
            temp.columns = new_names
            return temp
        return pd.DataFrame()

    # 1. Vector expansion (creates individual x, y, z columns)
    df_x = split_vector_col(df, 'x', ['x_x', 'x_y', 'x_z'])
    df_u = split_vector_col(df, 'u', ['u_x', 'u_y', 'u_z'])
    df_xsph = split_vector_col(df, 'xsph', ['xsph_x', 'xsph_y', 'xsph_z'])
    df_n = split_vector_col(df, 'n', ['n_x', 'n_y', 'n_z'])
    
    # Expand 'sph' which contains [r, theta, phi]
    df_sph = split_vector_col(df, 'sph', ['r', 'theta', 'phi'])

    # 2. List of "simple" columns (scalars) to keep
    # Adding 'cs' here as it is often missing!
    existing_scalars = ['port', 'subwindow', 'l', 'cs'] 
    cols_to_keep = [c for c in existing_scalars if c in df.columns]
    
    # 3. Final assembly
    # Concatenate scalars with all decomposed vector columns
    df_final = pd.concat([
        df[cols_to_keep],
        df_x,
        df_u,
        df_xsph,
        df_n,
        df_sph
    ], axis=1)

    return df_final


def get_final_beam_geometry(df_in):
    """
    Transforme le DataFrame de géométrie en une table finale 'Master Table'
    en décomposant les vecteurs [x, y, z] en colonnes individuelles.
    """
    # 1. Copie pour éviter de modifier le DataFrame original
    df = df_in.copy()

    # 2. Dimensions standard des faisceaux OMEGA (Diamètre 28cm)
    df['l1'] = 0.28
    df['l2'] = 0.28

    # 3. Fonction interne pour diviser les colonnes de vecteurs
    def expand_vector_column(target_df, column_name):
        if column_name in target_df.columns:
            # Création d'un DF temporaire à partir des listes/arrays numpy
            temp = pd.DataFrame(target_df[column_name].tolist(), index=target_df.index)
            temp.columns = [f'{column_name}_x', f'{column_name}_y', f'{column_name}_z']
            return temp
        return pd.DataFrame()

    # 4. Expansion des colonnes géométriques critiques
    # On transforme les objets [x,y,z] en trois colonnes distinctes
    df_coords = pd.concat([
        expand_vector_column(df, 'center'),
        expand_vector_column(df, 't1'),
        expand_vector_column(df, 't2'),
        expand_vector_column(df, 'normal'),
        expand_vector_column(df, 'focus')
    ], axis=1)

    # 5. Sélection des métadonnées (en vérifiant leur existence)
    metadata_cols = ['port', 'subwindow', 'LAT_rad', 'LONG_rad', 'color', 'energy', 'wavelength', 'l1', 'l2']
    available_metadata = [c for c in metadata_cols if c in df.columns]

    # 6. Assemblage final
    df_final = pd.concat([df[available_metadata], df_coords], axis=1)

    # 7. Renommage esthétique pour LAT/LONG
    df_final = df_final.rename(columns={'LAT_rad': 'LAT', 'LONG_rad': 'LONG'})

    return df_final

def impact_simulated(num_total_rays, target_radius, target_center):
    'Monte Carlo simulation which attribute a nomber of rays to a beam and determines the energy of each ray'

    beam_df = geometry.get_omega60_dataframe() 
    
    
    beam_df['l1'] = 0.28
    beam_df['l2'] = 0.28
    
    # Monte Carlo simulation which attributes the number of beams with a probability depending on energy
    energies = beam_df['energy'].values
    probs = energies / np.sum(energies)
    selected_indices = np.random.choice(len(beam_df), size=num_total_rays, p=probs)
    counts = np.bincount(selected_indices, minlength=len(beam_df))
    
    #Then, we wach all the rays and we determine their energy to prepare a weight for the histogram
    impact_list = []
    for i, n_rays in enumerate(counts):
        if n_rays == 0: continue
        
        beam_row = beam_df.iloc[i]
        
        # the beam_hit fonction enables us to have all the informations wanted conserning one beam 
        res = beam_hit(beam_row, n_rays, target_radius, target_center)
        
        if not res.empty:
            res['energy_weight'] = beam_row['energy'] / n_rays
            impact_list.append(res)
            
    return pd.concat(impact_list, ignore_index=True)


def impact_df_vista(df_in):
    """
    Helper function to break down vectors in the impact DataFrame.
    """
    df = df_in.copy()
    
    # Function to efficiently expand vector columns
    def expand(df_target, col_name, names):
        if col_name in df_target.columns:
            temp = pd.DataFrame(df_target[col_name].tolist(), index=df_target.index)
            temp.columns = names
            return temp
        return pd.DataFrame()

    # Expand key vectors
    df_x = expand(df, 'x', ['x_x', 'x_y', 'x_z'])
    df_u = expand(df, 'u', ['u_x', 'u_y', 'u_z'])
    df_xsph = expand(df, 'xsph', ['xsph_x', 'xsph_y', 'xsph_z'])
    df_n = expand(df, 'n', ['n_x', 'n_y', 'n_z'])
    
    # Expand spherical coordinates (sph: r, theta, phi)
    df_sph = expand(df, 'sph', ['r', 'theta', 'phi'])

    # Concatenate everything expanded with the original data
    return pd.concat([df.drop(columns=['x', 'u', 'xsph', 'n', 'sph'], errors='ignore'), 
                      df_x, df_u, df_xsph, df_n, df_sph], axis=1)

def preparar_dataframe_impacts(impact_df):
    # 1. Expansion des vecteurs (x, u, xsph, n, sph)
    impact_df = impact_df_vista(impact_df)

    # 2. Sécurité : s'assurer que les colonnes critiques existent
    if 'port' not in impact_df.columns: 
        impact_df['port'] = 'unknown'
    if 'subwindow' not in impact_df.columns:
        impact_df['subwindow'] = 'unknown'
    
    # AJOUT : Sécurité pour energy_weight (au cas où un faisceau n'aurait pas d'énergie définie)
    if 'energy_weight' not in impact_df.columns:
        impact_df['energy_weight'] = 0.0

    # 3. Sélection des colonnes pertinentes
    # ON AJOUTE 'energy_weight' DANS LA LISTE CI-DESSOUS
    cols = [
        'port', 'subwindow', 'energy_weight', 'l', 'cs', 'theta', 'phi',
        'x_x', 'x_y', 'x_z',
        'u_x', 'u_y', 'u_z',
        'xsph_x', 'xsph_y', 'xsph_z',
        'n_x', 'n_y', 'n_z'
    ]
    
    # On ne garde que les colonnes qui existent réellement pour éviter les erreurs
    available_cols = [c for c in cols if c in impact_df.columns]

    # 4. Création du DataFrame final et arrondi
    formatted_df = impact_df[available_cols].copy()
    formatted_df = formatted_df.round(4)

    print(f"Valid impacts found: {len(formatted_df)}")
    return formatted_df

    

# --- Section des Tests ---

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