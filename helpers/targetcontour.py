import numpy as np
import pandas as pd


def generate_sphere(radius, N):
    """Generates the core (A half-disk in positive R for R-Z geometry)"""
    # Create a half-circle from -90 to +90 degrees (South to North pole)
    theta = np.linspace( 0.0, 2.*np.pi, N)
    R = radius * np.cos(theta)
    Z = radius * np.sin(theta)
    
    # Close the shape straight down along the Z axis (R=0)
    R = np.append(R, R[0])
    Z = np.append(Z, Z[0])

    df = pd.DataFrame({'R': R, 'Z': Z})

    df['R'] = - df['R']  # Flip R to be positive
    return df

def generate_shell(r_out, r_in, N):
    
    """Generates a hollow shell contour (A C-shape in positive R)"""
    
    theta = np.linspace( 0.0, 2.*np.pi, N)

    # Outer arc going up (from South to North pole)
    R_out = r_out * np.cos(theta)
    Z_out = r_out * np.sin(theta)
    
    # Inner arc going down (from North to South pole)
    R_in = r_in * np.cos(theta)
    Z_in = r_in * np.sin(theta)
    
    # Connect the outer and inner arcs to form a closed shell
    R_shell = np.concatenate([ R_out, R_in[::-1], [R_out[0]], ])
    Z_shell = np.concatenate([ Z_out, Z_in[::-1], [Z_out[0]], ])
    
    df = pd.DataFrame({'R': R_shell, 'Z': Z_shell})
    df['R'] = - df['R']  # Flip R to be positive
    return df
