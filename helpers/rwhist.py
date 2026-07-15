# Copyright (C) 2026 Manuel Cotelo Ferreiro (Instituto de Fusión Nuclear Guillermo Velarde, Universidad Politécnica de Madrid)

import sys
from turtle import lt

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

try:
  from . import utilsclara
except ImportError:
  import helpers.utilsclara as utilsclara

info = {
  'name': 'hist',
  'desc': 'Module to compute histograms of impact distributions',
  'author': 'Manuel Cotelo Ferreiro',
  'email': 'manuel.cotelo@upm.es',
  'year': 2026,
  'version': [ 1, 0, 0, ],
  'copyright': 'Copyright (C) 2026 Manuel Cotelo Ferreiro (Instituto de Fusión Nuclear Guillermo Velarde, Universidad Politécnica de Madrid)',
}

class SphereSectors():

  @staticmethod
  def spherical_sector_area( theta1, theta2, phi1, phi2, radius=1., ): #sij
      """
      Calculates the surface area of a spherical sector (a cell on the grid)
      defined by theta and phi boundaries.
      """
      # Calculate the area of the spherical zone (ring) between theta1 and theta2
      area_anillo = 2 * np.pi * radius**2 * np.abs(np.cos(theta1) - np.cos(theta2))
      
      # Calculate the longitudinal span
      delta_phi = np.abs(phi2 - phi1)
      
      # The sector area is the fraction of the ring corresponding to delta_phi
      area_sector = (delta_phi / (2 * np.pi)) * area_anillo
      
      return area_sector
  
  @staticmethod
  def factory( num_theta, num_phi, ):
      return SphereSectors( np.linspace(0, np.pi, num_theta + 1), np.linspace(0, 2 * np.pi, num_phi + 1) )
  
  def __init__(self, theta_edges, phi_edges):
      
    # validation
    assert len(theta_edges) >= 2, "There must be at least 2 theta edges"
    assert len(phi_edges) >= 2, "There must be at least 2 phi edges"
    assert np.all(theta_edges >= 0) and np.all(theta_edges <= np.pi), "Theta edges must be in [0, pi]"
    assert np.all(phi_edges >= 0) and np.all(phi_edges <= 2 * np.pi), "Phi edges must be in [0, 2*pi]"
    assert np.all(np.diff(theta_edges) > 0), "Theta edges must be strictly increasing"
    assert np.all(np.diff(phi_edges) > 0), "Phi edges must be strictly increasing"

    # Create the theta-phi grid edges
    self.theta_edges = theta_edges  # includes poles
    self.phi_edges = phi_edges  # includes closure

    # Coordinates of the spherical sector centers
    self.theta = 0.5 * (self.theta_edges[:-1] + self.theta_edges[1:])
    self.phi = 0.5 * (self.phi_edges[:-1] + self.phi_edges[1:])

    self.mtheta, self.mphi = np.meshgrid(self.theta, self.phi, indexing='ij')       

    # compute areas of the sectors
    self.areas = self.spherical_sector_area( 
        self.theta_edges[:-1][:,None], 
        self.theta_edges[1:][:,None], 
        self.phi_edges[:-1][None,:], 
        self.phi_edges[1:][None,:], 
        radius=1. 
    )

    # Verification: the sum of all sector areas must equal the total sphere area (4*pi*r^2)
    assert np.isclose( np.sum( self.areas ), 4.*np.pi, atol=1e-4 ), "The calculated total area does not match the sphere's surface area."

    return
  
  def plot_mesh( self, radius=1., kwargs={} ):
    
    # Create 3D figure
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize=(8, 8)) 

    # Draw horizontal lines (theta)
    for theta in self.theta:
      x = radius * np.sin(theta) * np.cos(self.phi_edges)
      y = radius * np.sin(theta) * np.sin(self.phi_edges)
      z = radius * np.cos(theta)
      ax.plot(x, y, z, color='blue', linewidth=0.8, **kwargs)

    # Draw vertical lines (phi)
    for phi in self.phi:
      x = radius * np.sin(self.theta_edges) * np.cos(phi)
      y = radius * np.sin(self.theta_edges) * np.sin(phi)
      z = radius * np.cos(self.theta_edges)
      ax.plot(x, y, z, color='red', linewidth=0.8, **kwargs)

    # Aesthetics
    ax.set_box_aspect([1, 1, 1])
    ax.set_title("Sphere division into θ/φ sectors")
    ax.set_axis_off()

    return fig, ax
  
  def plot_centers( self, radius=1., kwargs={} ):
    
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize=(8, 8))

    x = radius * np.sin(self.mtheta) * np.cos(self.mphi)
    y = radius * np.sin(self.mtheta) * np.sin(self.mphi)
    z = radius * np.cos(self.mtheta)

    ax.scatter(x, y, z, **kwargs)

    return fig, ax
  
def compute_histogram(hits, sectors):
    """
    Calcule l'histogramme de densité d'énergie déposée sur la sphère.
    Utilise l'objet SphereSectors pour le partitionnement et le calcul des aires.
    """
    # 1. Préparation des données physiques
    # On multiplie l'énergie par le cosinus d'incidence (Loi de Lambert)
    theta = hits['lat_rad'].values
    phi = hits['long_rad'].values
    weights = (hits['energy_weight'] ).values

    # 2. Histogramme 2D (Calcul brut par secteur)
    # On utilise les 'edges' (bords) définis dans ton objet sectors
    hs_raw, _, _ = np.histogram2d(
        theta, 
        phi, 
        bins=[sectors.theta_edges, sectors.phi_edges], 
        weights=weights
    )

    # 3. Calcul de la densité (Energie par unité de surface)
    # ATTENTION : on utilise sectors.areas qui est le tableau 2D
    # calculé automatiquement par le module rwhist
    hs_density = hs_raw / sectors.areas

    # 4. Normalisation (0 à 1) pour le tracé de la carte de chaleur
    h_min, h_max = np.min(hs_density), np.max(hs_density)
    hs_norm = (hs_density - h_min) / (h_max - h_min) if h_max > h_min else hs_density

    # 5. Création du DataFrame de résultat (Aplatissement pour ARWEN/Plotly)
    return pd.DataFrame({
        'theta': sectors.mtheta.ravel(),
        'phi': sectors.mphi.ravel(),
        'area': sectors.areas.ravel(),
        'energy_raw': hs_raw.ravel(),
        'intensity': hs_density.ravel(),
        'intensity_norm': hs_norm.ravel()
    })

def write_histogram( theta_edges, phi_edges, hs, filename ):

  data = []
  for i in range(len(theta_edges)-1):
    for j in range(len(phi_edges)-1):
      data.append({
        'i': i,
        'j': j,
        'theta_lo': theta_edges[i],
        'phi_lo': phi_edges[j],
        'theta_hi': theta_edges[i+1],
        'phi_hi': phi_edges[j+1],
        'hs': hs[i,j],
      })
  
  df = pd.DataFrame(data)
  df.to_csv(filename, index=False) 
    
  return

def read_histogram( filename ):

    df = pd.read_csv(filename)

    theta_edges = np.unique( np.concatenate( ( df['theta_lo'].values, df['theta_hi'].values ) ) )
    phi_edges = np.unique( np.concatenate( ( df['phi_lo'].values, df['phi_hi'].values ) ) )
    
    hs = df['hs'].values.reshape( ( len(theta_edges)-1, len(phi_edges)-1 ) )

    return theta_edges, phi_edges, hs

#
# testing
#

def test_SphereSectors():

    num_theta = 10
    num_phi = 20

    sectors = SphereSectors.factory(num_theta, num_phi)

    assert len(sectors.theta_edges) == num_theta + 1
    assert len(sectors.phi_edges) == num_phi + 1
    assert sectors.mtheta.shape == (num_theta, num_phi)
    assert sectors.mphi.shape == (num_theta, num_phi)
    assert sectors.areas.shape == (num_theta, num_phi)

    return

def test_SphereSectors_plot():

    num_theta = 10
    num_phi = 20

    sectors = SphereSectors.factory(num_theta, num_phi)

    fig, ax = sectors.plot_mesh()
    plt.show()

    return

def test_SphereSectors_plot_centers():

    num_theta = 10
    num_phi = 20

    sectors = SphereSectors.factory(num_theta, num_phi)

    fig, ax = sectors.plot_centers()
    plt.show()

    return

def test_histogram_io():

    num_theta = 10
    num_phi = 20

    sectors = SphereSectors.factory(num_theta, num_phi)

    hs = np.random.rand(num_theta, num_phi)

    filename = 'test_histogram.csv'
    write_histogram(sectors.theta_edges, sectors.phi_edges, hs, filename)

    theta_edges, phi_edges, hs_read = read_histogram(filename)

    assert np.allclose(sectors.theta_edges, theta_edges)
    assert np.allclose(sectors.phi_edges, phi_edges)
    assert np.allclose(hs, hs_read)

    return

# run this script
if __name__ == '__main__':
  utilsclara.show_message()
  sys.exit( utilsclara.run_test( info, globals(), ), )
else:
  print( utilsclara.write_disclaimer(info), )
  