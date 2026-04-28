import argparse
import numpy as np

# Metadata
info = argparse.Namespace(
    name = 'coord',
    desc = 'Module with coord helper functions',
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

def sph_to_cart(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])

def cart_to_sph(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    # On utilise le modulo 2*pi pour rester entre 0 et 2*pi proprement
    phi = np.arctan2(y, x) % (2 * np.pi)
    return np.array([r, theta, phi])

# --- Testing ---
def test_random():
    print("# Running test: Random Spherical <-> Cartesian consistency")
    num_samples = 10000
    
    # 1. Génération de points aléatoires dans un cube [-1, 1]
    sample = 2 * np.random.random_sample(size=(num_samples, 3)) - 1
    
    # 2. Normalisation pour les placer sur la sphère unité (r=1)
    # On s'assure que r=1 pour simplifier la vérification
    sample = sample / np.linalg.norm(sample, axis=1)[:, np.newaxis]
    
    x_in, y_in, z_in = sample[:, 0], sample[:, 1], sample[:, 2]

    # 3. Test : Cartésien -> Sphérique
    # Ta fonction renvoie [r, theta, phi]
    r, theta, phi = cart_to_sph(x_in, y_in, z_in)
    
    # 4. Test : Sphérique -> Cartésien
    # Ta fonction renvoie [x, y, z], on transpose (.T) pour retrouver la forme (N, 3)
    reconstructed_cart = sph_to_cart(r, theta, phi).T

    # 5. Comparaison
    np.testing.assert_allclose(reconstructed_cart, sample, atol=1e-7)
    
    print(f"Random test passed with {num_samples} samples!")

# Pour l'exécuter si tu es dans le module coord.py :
if __name__ == "__main__":
    test_random()

def test_conversion():
    print("# Running test: Spherical <-> Cartesian consistency")
    
    r_in, t_in, p_in = 1.0, np.pi/2, 0.0
    
    # Sph -> Cart
    cart = sph_to_cart(r_in, t_in, p_in)
    expected_cart = np.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(cart, expected_cart, atol=1e-7)
    
    # Cart -> Sph
    sph = cart_to_sph(*cart)
    expected_sph = np.array([r_in, t_in, p_in])
    np.testing.assert_allclose(sph, expected_sph, atol=1e-7)
    
    print("Test passed !")

# --- Main Guard ---
if __name__ == "__main__":
    # Correction : info.name au lieu de info['name'] + alignement corrigé
    print(f"--- Running tests for module: {info.name} ---")
    test_conversion()
    test_random()
    print("All tests successful.")
else:
    print(write_disclaimer(info))

