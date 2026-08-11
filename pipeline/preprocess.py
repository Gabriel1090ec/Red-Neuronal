"""Carga y preparacion de datos del dataset Student Performance."""

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Columnas de entrada del modelo
FEATURES = ["studytime", "failures", "absences", "G1", "G2"]
SCALER_PATH = os.path.join("model", "scaler.joblib")


def cargar_datos(ruta_csv):
    """Carga el CSV del dataset (separador ';')."""
    return pd.read_csv(ruta_csv, sep=";")


def crear_columna_riesgo(df):
    """Crea la etiqueta binaria: 1 = riesgo (G3 < 10), 0 = estable."""
    df = df.copy()
    df["riesgo"] = (df["G3"] < 10).astype(int)
    return df


def dividir_datos(df):
    """Separa en train/test (80/20) de forma estratificada."""
    X = df[FEATURES]
    y = df["riesgo"]
    return train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )


def escalar_datos(X_train, X_test, guardar_scaler=True):
    """Ajusta el StandardScaler solo con train y transforma ambos sets."""
    scaler = StandardScaler()
    X_train_esc = scaler.fit_transform(X_train)
    X_test_esc = scaler.transform(X_test)

    if guardar_scaler:
        os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
        joblib.dump(scaler, SCALER_PATH)

    return X_train_esc, X_test_esc, scaler


def preparar_datos(ruta_csv):
    """Pipeline completo: carga, etiqueta, split y escalado."""
    df = cargar_datos(ruta_csv)
    df = crear_columna_riesgo(df)
    X_train, X_test, y_train, y_test = dividir_datos(df)
    X_train_esc, X_test_esc, scaler = escalar_datos(X_train, X_test)
    return X_train_esc, X_test_esc, y_train, y_test, scaler
