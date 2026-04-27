# Copyright (C) 2026 Clara Jourdan (Instituto de Fusión Nuclear Guillermo Velarde, Universidad Politécnica de Madrid)

import argparse
import pandas as pd

info = argparse.Namespace(
  name = 'utils',
  desc = 'Module with utility functions',
  author = 'Manuel Cotelo Ferreiro',
  email = 'manuel.cotelo@upm.es',
  year = 2024,
  version = [ 1, 0, 4, ],
  copyright = 'Copyright (C) 2021 Manuel Cotelo Ferreiro (Instituto de Fusión Nuclear Guillermo Velarde, Universidad Politécnica de Madrid)',
)


def write(df, filename):
    """
    Guarda un DataFrame en formato HDF5 con una clave estándar.
    """
    # 'key' est comme le nom de la feuille dans un fichier Excel
    # 'mode="w"' permet d'écraser le fichier s'il existe déjà
    df.to_hdf(filename, key="impactos", mode="w")