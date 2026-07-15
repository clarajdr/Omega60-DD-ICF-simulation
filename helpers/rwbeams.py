import numpy as np
import pandas as pd
import sys
from helpers import utilsclara

info = {
  'name': 'beams', # Corrigé de 'hist' à 'beams'
  'desc': 'Module to read and write laser beam geometry parameters',
  'author': 'Clara Jourdan',
  'email': 'clara.jourdan@imt-atlantique.net',
  'year': 2026,
  'version': [ 1, 0, 1, ],
  'copyright': 'Copyright (C) 2026 Clara Jourdan (Instituto de Fusión Nuclear Guillermo Velarde, Universidad Politécnica de Madrid)',
}

# List of the required colums for the simulation
required_columns = [ 
    'id', 'LAT_rad', 'LONG_rad', 'energy', 'wavelength', 
    'l1', 'l2',                       # Tailles de la fenêtre (AJOUTÉ)
    'xc', 'yc', 'zc',                 # Centre
    'nx', 'ny', 'nz',                 # Normale
    't1x', 't1y', 't1z', 't2x', 't2y', 't2z', # Tangentes
    'R',                              # Rayon
    'fx', 'fy', 'fz'                  # Focus
]

def validate_beams(df):
    """Vérifie si le DataFrame contient toutes les colonnes requises."""
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f'Missing required column: {col}')
    return True

def write_beams(df, file_name):
    """Valide et écrit les données dans un fichier Excel."""
    validate_beams(df)
    # index=False évite d'ajouter une colonne d'index inutile dans Excel
    df.to_excel(file_name, index=False)
    print(f"# Beams successfully written to {file_name}")
    return

def read_beams(file_name):
    """Lit le fichier Excel et valide sa structure."""
    df = pd.read_excel(file_name)
    validate_beams(df)
    return df

def params_testing():
    """
    Génère un faisceau de test avec TOUTES les colonnes requises.
    Utilise l1=0.28 et l2=0.28 (standard OMEGA).
    """
    data = {
        'id': ['TestBeam'], 
        'LAT_rad': [0.0], 
        'LONG_rad': [0.0],
        'energy': [100.0], 
        'wavelength': [0.351e-6],
        'l1': [0.28], 'l2': [0.28],             # Valeurs de test
        
        'xc': [1.65], 'yc': [0.0], 'zc': [0.0], # Centre
        'nx': [-1.0], 'ny': [0.0], 'nz': [0.0], # Normale
        
        't1x': [0.0], 't1y': [1.0], 't1z': [0.0],
        't2x': [0.0], 't2y': [0.0], 't2z': [-1.0],
        
        'R': [0.14],                            # Rayon (R)
        'fx': [0.0], 'fy': [0.0], 'fz': [0.0]   # Focus
    }
    
    df = pd.DataFrame(data)
    validate_beams(df)
    return df

def cosine_test():
    """
    Génère un faisceau spécifique pour le test de la loi de Lambert.
    Les rayons seront envoyés parallèlement le long de l'axe X.
    """
    data = {
        'id': ['LambertTestBeam'], 
        'LAT_rad': [0.0], 
        'LONG_rad': [0.0],
        'energy': [1.0],          # On met l'énergie à 1 pour faciliter le test
        'wavelength': [0.351e-6], # Longueur d'onde standard OMEGA
        
        # Fenêtre de 3mm pour bien couvrir la bille de 2mm (0.001 de rayon)
        'l1': [0.003], 'l2': [0.003], 
        
        # Positionnement sur l'axe X, à 5cm du centre
        'xc': [0.05], 'yc': [0.0], 'zc': [0.0], 
        
        # La normale pointe vers l'origine (direction -X)
        'nx': [-1.0], 'ny': [0.0], 'nz': [0.0], 
        
        # Tangentes définissant le plan YZ de la fenêtre
        't1x': [0.0], 't1y': [1.0], 't1z': [0.0],
        't2x': [0.0], 't2y': [0.0], 't2z': [1.0],
        
        'R': [0.0015], # Rayon de la fenêtre (l1/2)
        
        # Le focus est mis à l'origine, mais on l'ignorera pour forcer le parallélisme
        'fx': [0.0], 'fy': [0.0], 'fz': [0.0] 
    }
    
    df = pd.DataFrame(data)
    
    # On utilise ta fonction de validation pour être sûr que tout y est[cite: 2]
    validate_beams(df) 
    
    return df

# --- Testing ---

def test_beam_io():
    """Test d'intégrité pour l'écriture et la lecture Excel."""
    df = params_testing()
    file_name = 'testing_beams.xlsx'
    
    write_beams(df, file_name)
    df2 = read_beams(file_name)
    
    # Vérification que les colonnes l1 et l2 sont bien présentes et correctes
    assert 'l1' in df2.columns and 'l2' in df2.columns
    assert df2['l1'].iloc[0] == 0.28
    
    print("# Test passed: l1 and l2 are correctly handled in the Excel file.")
    return

if __name__ == '__main__':
    utilsclara.show_message()
    sys.exit(utilsclara.run_test(info, globals()))
else:
    print(utilsclara.write_disclaimer(info))


