"""Explicabilidad simple: importancia de variables segun la primera capa."""

import os
import numpy as np
import matplotlib.pyplot as plt

FEATURES = ["studytime", "failures", "absences", "G1", "G2"]
# Se guarda en evaluation/ para que el informe Quarto la encuentre ahi
OUTPUT_PATH = os.path.join("evaluation", "importancia_variables.png")


def calcular_importancia(modelo):
    """Suma los pesos absolutos de la primera capa por cada variable de entrada."""
    pesos_primera_capa = modelo.layers[0].get_weights()[0]  # (n_features, n_neuronas)
    importancia = np.sum(np.abs(pesos_primera_capa), axis=1)
    return importancia


def graficar_importancia(modelo, ruta_salida=OUTPUT_PATH):
    """Genera y guarda un grafico de barras con la importancia de cada variable."""
    importancia = calcular_importancia(modelo)

    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)

    fig, eje = plt.subplots(figsize=(8, 5))
    eje.bar(FEATURES, importancia, color="steelblue")
    eje.set_title("Importancia de variables (pesos primera capa)")
    eje.set_ylabel("Importancia (suma de pesos absolutos)")
    fig.tight_layout()
    fig.savefig(ruta_salida)
    plt.close(fig)

    return importancia
