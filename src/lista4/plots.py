"""Gráficas reutilizables."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def grafica_dispersion_con_recta(
    df: pd.DataFrame, x: str, resumen: dict, objetivo: str = "y", ax=None
):
    """Dispersión de ``objetivo`` contra ``x`` con la recta ajustada.

    ``resumen`` es el diccionario que devuelve ``resumen_regresion`` para un
    modelo de **una sola variable** (``x``).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    b0 = resumen["intercepto"]
    b1 = resumen["coeficientes"][x]
    xs = np.linspace(df[x].min(), df[x].max(), 200)

    ax.scatter(df[x], df[objetivo], s=14, alpha=0.5, label="Datos")
    ax.plot(xs, b0 + b1 * xs, color="crimson", linewidth=2,
            label=f"Recta: {b0:.3f} + {b1:.3f}·{x}")
    ax.set_xlabel(x)
    ax.set_ylabel(objetivo)
    ax.set_title(f"{objetivo} contra {x}  (riesgo empírico = {resumen['riesgo']:.4f})")
    ax.legend()
    ax.grid(alpha=0.3)
    return ax
