"""Métricas y riesgos empíricos usados en la lista."""

import numpy as np
import pandas as pd


def riesgo_empirico_cuadratico(y, y_pred) -> float:
    r"""Riesgo empírico con pérdida cuadrática.

    .. math:: \hat R_S(h) = \frac{1}{n}\sum_{i=1}^n (y_i - h(x_i))^2
    """
    y = np.asarray(y, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y - y_pred) ** 2))


def riesgo_empirico_logistico(y, p_hat, eps: float = 1e-15) -> float:
    r"""Riesgo empírico logístico (entropía cruzada promedio).

    .. math:: \hat R_S(h) = -\frac1n \sum_{i=1}^n
              \left[y_i\log\hat p_i + (1-y_i)\log(1-\hat p_i)\right]

    Se recortan las probabilidades a ``[eps, 1-eps]`` solo para evitar
    ``log(0)`` por redondeo numérico.
    """
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p_hat, dtype=float), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def errores_clasificacion(y, p_hat, umbral: float = 0.5) -> tuple[int, float]:
    """Número y proporción de errores al clasificar con ``p_hat >= umbral``."""
    y = np.asarray(y)
    clase = (np.asarray(p_hat) >= umbral).astype(int)
    n_err = int(np.sum(clase != y))
    return n_err, n_err / len(y)


def correlacion(df: pd.DataFrame, a: str, b: str) -> float:
    """Correlación de Pearson entre dos columnas de ``df``."""
    return float(np.corrcoef(df[a], df[b])[0, 1])
