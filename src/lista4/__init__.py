"""Paquete local de la Cuarta lista de ejercicios de Introducción al Machine Learning.

Contiene el código reutilizable (carga de datos, ajuste de modelos y cálculo de
riesgos) que importan los cuadernos de la carpeta ``notebooks/``.
"""

from .data import cargar_datos, DATA_DIR
from .metrics import (
    riesgo_empirico_cuadratico,
    riesgo_empirico_logistico,
    errores_clasificacion,
    correlacion,
)
from .models import (
    ajustar_regresion_lineal,
    resumen_regresion,
    ajustar_regresion_logistica,
    resumen_logistica,
    tabla_resumen_regresion,
)
from .plots import grafica_dispersion_con_recta

__all__ = [
    "cargar_datos",
    "DATA_DIR",
    "riesgo_empirico_cuadratico",
    "riesgo_empirico_logistico",
    "errores_clasificacion",
    "correlacion",
    "ajustar_regresion_lineal",
    "resumen_regresion",
    "ajustar_regresion_logistica",
    "resumen_logistica",
    "tabla_resumen_regresion",
    "grafica_dispersion_con_recta",
]
