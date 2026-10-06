"""Ajuste de modelos (regresión lineal y regresión logística).

Cada función recibe el ``DataFrame`` y los nombres de las columnas, así que el
mismo código sirve para todos los archivos y todos los ejercicios.
"""

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from .metrics import (
    errores_clasificacion,
    riesgo_empirico_cuadratico,
    riesgo_empirico_logistico,
)


# --------------------------------------------------------------------------- #
# Regresión lineal
# --------------------------------------------------------------------------- #
def ajustar_regresion_lineal(
    df: pd.DataFrame, predictores: Sequence[str], objetivo: str = "y"
) -> LinearRegression:
    """Ajusta una regresión lineal con intercepto por mínimos cuadrados."""
    modelo = LinearRegression(fit_intercept=True)
    modelo.fit(df[list(predictores)].to_numpy(), df[objetivo].to_numpy())
    return modelo


def resumen_regresion(
    df: pd.DataFrame, predictores: Sequence[str], objetivo: str = "y"
) -> dict:
    """Ajusta el modelo y devuelve intercepto, coeficientes, riesgo y predicciones.

    Returns
    -------
    dict con las llaves ``modelo``, ``predictores``, ``intercepto``,
    ``coeficientes`` (dict nombre -> valor), ``riesgo`` (riesgo empírico
    cuadrático) y ``y_pred``.
    """
    predictores = list(predictores)
    modelo = ajustar_regresion_lineal(df, predictores, objetivo)
    y_pred = modelo.predict(df[predictores].to_numpy())
    return {
        "modelo": modelo,
        "predictores": predictores,
        "intercepto": float(modelo.intercept_),
        "coeficientes": dict(zip(predictores, map(float, modelo.coef_))),
        "riesgo": riesgo_empirico_cuadratico(df[objetivo], y_pred),
        "y_pred": y_pred,
    }


# --------------------------------------------------------------------------- #
# Regresión logística
# --------------------------------------------------------------------------- #
def ajustar_regresion_logistica(
    df: pd.DataFrame,
    predictores: Sequence[str],
    objetivo: str = "abandona_30d",
    max_iter: int = 10_000,
) -> LogisticRegression:
    """Regresión logística prácticamente sin regularización.

    Se usa ``solver='lbfgs'`` y ``C=1e6`` (penalización despreciable), como pide
    el enunciado, con ``max_iter`` grande para asegurar convergencia.
    """
    modelo = LogisticRegression(solver="lbfgs", C=1e6, max_iter=max_iter)
    modelo.fit(df[list(predictores)].to_numpy(), df[objetivo].to_numpy())
    return modelo


def resumen_logistica(
    df: pd.DataFrame,
    predictores: Sequence[str],
    objetivo: str = "abandona_30d",
    umbral: float = 0.5,
    max_iter: int = 10_000,
) -> dict:
    """Ajusta el modelo logístico y calcula riesgo y errores de clasificación."""
    predictores = list(predictores)
    modelo = ajustar_regresion_logistica(df, predictores, objetivo, max_iter)
    p_hat = modelo.predict_proba(df[predictores].to_numpy())[:, 1]
    n_err, prop_err = errores_clasificacion(df[objetivo], p_hat, umbral)
    return {
        "modelo": modelo,
        "predictores": predictores,
        "riesgo": riesgo_empirico_logistico(df[objetivo], p_hat),
        "errores": n_err,
        "prop_errores": prop_err,
        "p_hat": p_hat,
    }


# --------------------------------------------------------------------------- #
# Tablas resumen
# --------------------------------------------------------------------------- #
def tabla_resumen_regresion(
    df: pd.DataFrame, especificaciones: dict, objetivo: str = "y"
) -> pd.DataFrame:
    """Tabla con una fila por modelo: intercepto, coeficientes y riesgo empírico.

    Parameters
    ----------
    especificaciones : dict
        ``{"nombre del modelo": [lista de predictores]}``. Las columnas de
        coeficientes quedan en ``NaN`` para los modelos que no usan esa variable.
    """
    filas = {}
    for nombre, predictores in especificaciones.items():
        r = resumen_regresion(df, predictores, objetivo)
        fila = {"Intercepto": r["intercepto"]}
        fila.update({f"coef_{k}": v for k, v in r["coeficientes"].items()})
        fila["Riesgo empírico"] = r["riesgo"]
        filas[nombre] = fila
    tabla = pd.DataFrame.from_dict(filas, orient="index")
    # El riesgo empírico siempre va al final
    cols = [c for c in tabla.columns if c != "Riesgo empírico"] + ["Riesgo empírico"]
    return tabla[cols]
