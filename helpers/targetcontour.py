import numpy as np
import pandas as pd

def generate_sphere(radius, N):
    """Generates the core (A half-disk in positive R for R-Z geometry)"""
    # Create a half-circle from -90 to +90 degrees (South to North pole)
    theta = np.linspace(-np.pi/2, np.pi/2, N)
    R = radius * np.cos(theta)
    Z = radius * np.sin(theta)
    
    # Close the shape straight down along the Z axis (R=0)
    R = np.append(R, 0.0)
    Z = np.append(Z, -radius)
    return pd.DataFrame({'R': R, 'Z': Z})

def generate_shell(r_out, r_in, N):
    """Generates a hollow shell contour (A C-shape in positive R)"""
    # Outer arc going up (from South to North pole)
    theta_out = np.linspace(-np.pi/2, np.pi/2, N)
    R_out = r_out * np.cos(theta_out)
    Z_out = r_out * np.sin(theta_out)
    
    # Inner arc going down (from North to South pole)
    theta_in = np.linspace(np.pi/2, -np.pi/2, N)
    R_in = r_in * np.cos(theta_in)
    Z_in = r_in * np.sin(theta_in)
    
    # Connect the outer and inner arcs to form a closed shell
    R_shell = np.concatenate([R_out, R_in])
    Z_shell = np.concatenate([Z_out, Z_in])
    return pd.DataFrame({'R': R_shell, 'Z': Z_shell})
