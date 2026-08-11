"""Evaluacion del modelo: matriz de confusion y metricas."""

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


def evaluar_modelo(modelo, X_test, y_test):
    """Calcula matriz de confusion, accuracy, precision, recall y F1."""
    probs = modelo.predict(X_test, verbose=0)
    y_pred = (probs > 0.5).astype(int).ravel()

    matriz = confusion_matrix(y_test, y_pred)
    metricas = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }

    print("Matriz de confusion:")
    print(matriz)
    print("\nMetricas:")
    for nombre, valor in metricas.items():
        print(f"  {nombre}: {valor:.4f}")
    print("\nReporte de clasificacion:")
    print(classification_report(y_test, y_pred, target_names=["estable", "riesgo"]))

    return matriz, metricas
