"""Definicion de la arquitectura de la red neuronal."""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout

N_FEATURES = 5


def construir_modelo():
    """Crea y compila la red neuronal para clasificacion binaria."""
    modelo = Sequential([
        Input(shape=(N_FEATURES,)),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])

    modelo.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return modelo
