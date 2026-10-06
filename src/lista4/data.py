"""Carga de datos.

Todos los cuadernos leen los CSV a través de :func:`cargar_datos`, de modo que
la ruta a ``data/`` se define en un único lugar.
"""

from pathlib import Path

import pandas as pd

# Raíz del repositorio: lista_4/src/lista4/data.py -> lista_4/
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def cargar_datos(nombre: str) -> pd.DataFrame:
    """Lee un CSV de la carpeta ``data/``.

    Parameters
    ----------
    nombre : str
        Nombre del archivo, con o sin la extensión ``.csv``
        (por ejemplo ``"experimento_1"`` o ``"clientes.csv"``).

    Returns
    -------
    pandas.DataFrame
        Los datos tal como vienen en el archivo.
    """
    if not nombre.endswith(".csv"):
        nombre = f"{nombre}.csv"
    ruta = DATA_DIR / nombre
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró {ruta}. Verifique que los CSV estén en la carpeta data/."
        )
    return pd.read_csv(ruta)
