import numpy as np
import pandas as pd

def generate_sphere(radius,N):
    'generates a sphere by points'
    theta = np.linspace(0, 2*np.pi,N)
    R=radius*np.cos(theta)
    Z=radius*np.sin(theta)

    return pd.DataFrame({'R': R, 'Z': Z})

