"""Orquestador principal: carga el modelo ya entrenado y muestra resultados.

El entrenamiento se hace aparte (Colab) con model/train.py, no aqui.
"""

import os
import sys

DATA_PATH = os.path.join("data", "student-mat.csv")
MODEL_PATH = os.path.join("model", "model.h5")
SCALER_PATH = os.path.join("model", "scaler.joblib")
PLOT_PATH = os.path.join("evaluation", "training_metrics.png")


def verificar_archivos():
    """Revisa que existan los archivos necesarios antes de arrancar."""
    requeridos = {
        "Dataset": DATA_PATH,
        "Modelo entrenado": MODEL_PATH,
        "Scaler": SCALER_PATH,
    }
    faltantes = [f"  - {nombre}: {ruta}" for nombre, ruta in requeridos.items() if not os.path.exists(ruta)]

    if faltantes:
        print("No se puede continuar, faltan estos archivos:")
        print("\n".join(faltantes))
        print("\nSi falta el modelo o el scaler: entrenalos en Colab con")
        print("model/train.py y copia model/model.h5 y model/scaler.joblib aqui.")
        sys.exit(1)


def mostrar_imagen(ruta, titulo):
    """Muestra una imagen guardada en disco, si existe."""
    if not os.path.exists(ruta):
        return
    import matplotlib.pyplot as plt
    imagen = plt.imread(ruta)
    plt.figure()
    plt.imshow(imagen)
    plt.axis("off")
    plt.title(titulo)
    plt.show()


def main():
    """Carga el modelo entrenado y muestra metricas e importancia de variables."""
    verificar_archivos()

    # Import de tensorflow aislado para dar un mensaje claro si falta
    try:
        from tensorflow.keras.models import load_model
    except ModuleNotFoundError:
        print("Falta el paquete 'tensorflow' en este Python.")
        print("Instalalo con: python -m pip install tensorflow")
        sys.exit(1)

    from pipeline.preprocess import preparar_datos
    from evaluation.metrics import evaluar_modelo
    from explainability.explain import graficar_importancia, OUTPUT_PATH as EXPLAIN_PATH

    print("Cargando modelo existente (no se re-entrena)...")
    modelo = load_model(MODEL_PATH)
    _, X_test, _, y_test, _ = preparar_datos(DATA_PATH)

    mostrar_imagen(PLOT_PATH, "Metricas de entrenamiento")

    print("\nEvaluacion del modelo:")
    evaluar_modelo(modelo, X_test, y_test)

    print("\nImportancia de variables:")
    graficar_importancia(modelo)
    mostrar_imagen(EXPLAIN_PATH, "Importancia de variables")


if __name__ == "__main__":
    main()
