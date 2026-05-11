from helpers import sampler
import argparse
import numpy as np
import scipy as sp
from helpers import coord
from helpers import geometry
from helpers import sampler
import matplotlib.pyplot as plt
import pandas as pd

info = {
  'name': 'collision', 
  'desc': 'Module to read and write laser beam geometry parameters',
  'author': 'Clara Jourdan',
  'email': 'clara.jourdan@imt-atlantique.net',
  'year': 2026,
  'version': [ 1, 0, 1, ],
  'copyright': 'Copyright (C) 2026 Clara Jourdan (Instituto de Fusión Nuclear Guillermo Velarde, Universidad Politécnica de Madrid)',
}

def beam_hit(beam_row, num_samples, radius, center_target):
    """
    Simulation de tir laser.
    Retourne un DataFrame avec les colonnes exactes demandées par l'utilisateur.
    """
    impacts = []
    
    # 1. Reconstruction des vecteurs de base (Lecture depuis la Master Table)
    t1 = np.array([beam_row['t1x'], beam_row['t1y'], beam_row['t1z']])
    t2 = np.array([beam_row['t2x'], beam_row['t2y'], beam_row['t2z']])
    center = np.array([beam_row['xc'], beam_row['yc'], beam_row['zc']])
    focus = np.array([beam_row['fx'], beam_row['fy'], beam_row['fz']])
    
    # 2. Échantillonnage sur le port laser (Profil Gaussien)
    dist_gauss = lambda n: sp.stats.norm(loc=0.0, scale=0.5).rvs(size=(n, 2))
    ray_origins = sampler.sample_window(t1, t2, l1=beam_row['l1'], l2=beam_row['l2'], 
                                center=center, num_points=num_samples, dist=dist_gauss)
    
    # 3. Boucle de tir et remplissage du tableau
    for i in range(num_samples):
        x_ray = ray_origins[i]
        direction = focus - x_ray
        u_ray = direction / np.linalg.norm(direction)
        
        # Ray-Tracing : res = [x, u, l, xsph, cs, [r, theta, phi], n]
        res = sampler.ray_hit(x_ray, u_ray, radius, center_target)
        
        if res[2] > 0:  # Si impact sur la cible
            impacts.append({
                'id': beam_row['id'],           # Identifiant du faisceau
                'cs': res[4],
                'l': res[2],                    # Distance d'impact
                # Origine du rayon (xx, xy, xz)
                'xx': res[0][0], 'xy': res[0][1], 'xz': res[0][2],
                # Direction unitaire (ux, uy, uz)
                'ux': res[1][0], 'uy': res[1][1], 'uz': res[1][2],
                # Point d'impact sur cible (xsphx, xsphy, xsphz)
                'xsphx': res[3][0], 'xsphy': res[3][1], 'xsphz': res[3][2],
                # Normale à la surface (nx, ny, nz)
                'nx': res[6][0], 'ny': res[6][1], 'nz': res[6][2],
                # Coordonnées sphériques d'impact
                'r': res[5][0],
                'long_rad': res[5][2],          # Azimut (phi)
                'lat_rad': res[5][1],           # Inclinaison (theta)
                'R': beam_row['R']              # Rayon du faisceau original
            })
            
    return pd.DataFrame(impacts)

def plot_impact_footprint(impact_df, sphere_radius):
    """
    Plots the 3D footprint of a single beam's rays on the target sphere.
    """
    # 1. Create 3D figure and axis
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 2. Draw the SOLID colored sphere (target)
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x_sphere = sphere_radius * np.outer(np.cos(u), np.sin(v))
    y_sphere = sphere_radius * np.outer(np.sin(u), np.sin(v))
    z_sphere = sphere_radius * np.outer(np.ones_like(u), np.cos(v))
    
    # Surface plot with transparency to see impacts on both sides
    ax.plot_surface(x_sphere, y_sphere, z_sphere, 
                    color='lightblue', 
                    alpha=0.3,       # Transparency
                    linewidth=0, 
                    antialiased=True)

    # 3. Plot the impact points from the expanded DataFrame
    # Using the columns we just created: xsph_x, xsph_y, xsph_z
    xs = impact_df['xsphx']
    ys = impact_df['xsphy']
    zs = impact_df['xsphz']
    
    # Scatter plot of the 10,000 rays (or as many as hit the target)
    ax.scatter(xs, ys, zs, color='red', s=2, alpha=0.6, label='Ray Impacts')

    # 4. Aesthetic configuration
    beam_id = impact_df['id'].iloc[0] if not impact_df.empty else "N/A"
    ax.set_title(f"Target Impact Footprint - Beam Port: {beam_id}")
    
    # Set axis limits to fit the sphere
    limit = sphere_radius * 1.1
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    
    # Set 1:1:1 aspect ratio so it looks like a sphere, not an egg
    ax.set_box_aspect([1, 1, 1])
    
    plt.legend()
    plt.show()

def impact_simulated(num_total_rays, target_radius, target_center, beam_df):
    """
    SIMULATION UNIQUE : Répartit les rayons, simule les impacts et 
    renvoie un tableau final déjà formaté et arrondi.
    """
    # 1. Calcul des probabilités par énergie
    energies = beam_df['energy'].values
    probs = energies / np.sum(energies)
    
    # 2. Répartition des rayons (Monte Carlo)
    selected_indices = np.random.choice(len(beam_df), size=num_total_rays, p=probs)
    counts = np.bincount(selected_indices, minlength=len(beam_df))
    
    impact_list = [] 
    
    # 3. Boucle sur les faisceaux
    for i, n_rays in enumerate(counts):
        if n_rays == 0: continue
        
        beam_row = beam_df.iloc[i]
        
        # On appelle beam_hit qui nous rend DÉJÀ les colonnes (xx, xy, xsphx, etc.)
        res = beam_hit(beam_row, n_rays, target_radius, target_center)
        
        if not res.empty:
            # Ajout du poids énergétique
            res['energy_weight'] = (beam_row['energy'] / n_rays).round(6)
            impact_list.append(res)
    
    # 4. Fusion et Nettoyage final
    if not impact_list:
        return pd.DataFrame()
        
    df_final = pd.concat(impact_list, ignore_index=True)
    
    # On arrondit tout le tableau à 4 décimales pour la lisibilité
    df_final = df_final.round(4)
    
    print(f"Simulation terminée : {len(df_final)} impacts valides enregistrés.")
    return df_final


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

if __name__ == '__main__':
    import sys
    from helpers import utils  # Assure-toi que utils est accessible
    
    # Affiche le message de bienvenue du module
    utils.show_message()
    
    # Lance tous les tests commençant par "test_" trouvés dans les globals()
    sys.exit(utils.run_test(info, globals()))