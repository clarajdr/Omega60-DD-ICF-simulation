# Copyright (C) 2026 Clara Jourdan & Manuel Cotelo (IFN-GV)
# Simplified utility module for beam and histogram management

import os
import sys
import argparse
import datetime
import pickle

# --- 1. MÉTADONNÉES DU MODULE ---
info = argparse.Namespace(
    name = 'utils',
    desc = 'Simplified utility functions for IFN-GV project',
    author = 'Clara Jourdan',
    year = 2026,
    version = [1, 0, 0],
    copyright = 'Copyright (C) 2026 Instituto de Fusión Nuclear Guillermo Velarde',
)

# --- 2. FONCTIONS D'AFFICHAGE ET CRÉDITS ---
# Ces fonctions servent à afficher proprement qui a fait le code et quand.

def write_disclaimer(info_obj):
    """Affiche la mention de copyright légale."""
    name = info_obj.name if hasattr(info_obj, 'name') else info_obj.get('name', 'unknown')
    copy = info_obj.copyright if hasattr(info_obj, 'copyright') else info_obj.get('copyright', 'IFN-GV')
    return f'# {copy} hereby claims all interest in program "{name}"'

def write_info(info_obj):
    """Affiche les informations du module (Auteur, Version, etc.)."""
    if isinstance(info_obj, dict):
        items = info_obj.items()
    else:
        items = vars(info_obj).items()
    return '\n# info:\n' + '\n'.join([f'# {k:>18s} = {v}' for k, v in items])

def show_message():
    """Message d'avertissement quand on lance utils.py seul."""
    print('\n# warning :: this file is a helper module, not the main program.', file=sys.stderr)

# --- 3. SAUVEGARDE DE DONNÉES (PICKLE) ---
# Le format 'pickle' est beaucoup plus rapide que Excel pour les gros objets Python.

def apply_pickle(file_name, data):
    """Sauvegarde n'importe quel objet Python (DataFrame, Matrice, Classe) en binaire."""
    with open(file_name, 'wb') as fd:
        pickle.dump(data, fd, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"# Data pickled to {file_name}")

def apply_unpickle(file_name):
    """Recharge un objet sauvegardé avec apply_pickle."""
    with open(file_name, 'rb') as fd:
        return pickle.load(fd)

# --- 4. MOTEUR DE TESTS AUTOMATIQUES ---
# C'est ce qui fait tourner tes fonctions test_beam_io() tout seul.

def run_test(info_obj, global_vars):
    """Détecte et lance toutes les fonctions commençant par 'test_'."""
    # 1. Afficher qui lance le test
    print(write_info(info_obj))
    
    # 2. Lister les tests trouvés
    tests = {n: f for n, f in global_vars.items() if n.startswith('test_') and callable(f)}
    print(f"\n# Available tests: {', '.join(tests.keys())}")

    # 3. Exécuter chaque test
    for name, func in tests.items():
        print(f'\n# --- Running test: "{name}" ---')
        try:
            func()
            print(f"# Result: {name} PASSED")
        except Exception as e:
            print(f"# Result: {name} FAILED\n# Error: {e}")
    
    print("\n# All tests completed.\n")
    return 0

def nmsanitizer(name):
    """Nettoie une chaîne de caractères (espaces et minuscules)."""
    return str(name).strip().lower()

# --- EXÉCUTION ---
if __name__ == '__main__':
    show_message()
    # On lance les tests internes de utils.py (s'il y en avait)
    sys.exit(run_test(info, globals()))