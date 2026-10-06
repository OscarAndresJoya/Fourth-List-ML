# Cuarta lista de ejercicios — Introducción al Machine Learning

Universidad El Bosque · Programa de Matemáticas y Estadística · Semestre 2026-2

## Estructura del repositorio

```
lista_4/
|-- README.md
|-- requirements.txt
|-- pyproject.toml
|-- data/                    # CSV de la lista (experimentos 1-3 y clientes)
|-- src/lista4/              # código reutilizable (paquete local)
|   |-- data.py              # carga de datos: cargar_datos(nombre)
|   |-- models.py            # regresión lineal / logística y tablas resumen
|   |-- metrics.py           # riesgos empíricos, errores y correlación
|   `-- plots.py             # gráfica de dispersión con recta ajustada
`-- notebooks/               # un cuaderno ejecutado por ejercicio
    |-- ejercicio_1.ipynb
    |-- ejercicio_2.ipynb
    |-- ejercicio_3.ipynb
    `-- ejercicio_4.ipynb
```

Los cuadernos solo contienen la ejecución de cada ejercicio, sus tablas, gráficas y
respuestas; toda la lógica repetida (carga, ajuste, riesgos) está en `src/lista4/`.

