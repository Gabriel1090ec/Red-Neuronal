"""Entrenamiento del modelo y guardado de resultados."""

import os
import sys

# Permite ejecutar este archivo directamente (python model/train.py)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
from pipeline.preprocess import preparar_datos
from model.build_model import construir_modelo

DATA_PATH = os.path.join("data", "student-mat.csv")
MODEL_PATH = os.path.join("model", "model.h5")
PLOT_PATH = os.path.join("evaluation", "training_metrics.png")
EPOCHS = 50


def graficar_historial(historial, ruta_salida):
    """Genera graficas de accuracy y loss (train vs validation)."""
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)

    fig, ejes = plt.subplots(1, 2, figsize=(12, 5))

    ejes[0].plot(historial.history["accuracy"], label="train")
    ejes[0].plot(historial.history["val_accuracy"], label="validation")
    ejes[0].set_title("Accuracy")
    ejes[0].set_xlabel("Epoca")
    ejes[0].legend()

    ejes[1].plot(historial.history["loss"], label="train")
    ejes[1].plot(historial.history["val_loss"], label="validation")
    ejes[1].set_title("Loss")
    ejes[1].set_xlabel("Epoca")
    ejes[1].legend()

    fig.tight_layout()
    fig.savefig(ruta_salida)
    plt.close(fig)


def entrenar_modelo(ruta_csv=DATA_PATH):
    """Entrena el modelo completo y guarda modelo, scaler y graficas."""
    X_train, X_test, y_train, y_test, scaler = preparar_datos(ruta_csv)

    modelo = construir_modelo()
    historial = modelo.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        verbose=1,
    )

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    modelo.save(MODEL_PATH)

    graficar_historial(historial, PLOT_PATH)

    return modelo, X_test, y_test, scaler


if __name__ == "__main__":
    entrenar_modelo()
