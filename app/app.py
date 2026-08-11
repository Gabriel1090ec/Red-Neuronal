"""Aplicacion Streamlit para predecir riesgo academico (SIPRE)."""

import os
import numpy as np
import joblib
import streamlit as st
from tensorflow.keras.models import load_model

MODEL_PATH = os.path.join("model", "model.h5")
SCALER_PATH = os.path.join("model", "scaler.joblib")

st.set_page_config(
    page_title="SIPRE - Riesgo Academico",
    page_icon="🎓",
    layout="centered",
)

# Estilos personalizados: tema azul institucional
st.markdown(
    """
    <style>
    .stApp {
        background-color: #F4F6F8;
        color: #1A2B3C;
    }
    /* Etiquetas de los sliders bien visibles (respaldo del tema en config.toml) */
    .stApp [data-testid="stWidgetLabel"] p {
        color: #1A2B3C;
        font-weight: 600;
    }
    .encabezado {
        background-color: #0B3D66;
        padding: 1.5rem 2rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
    }
    .encabezado h1 {
        color: #FFFFFF;
        margin: 0;
        font-size: 2rem;
    }
    .encabezado p {
        color: #D6E4F0;
        margin: 0.3rem 0 0 0;
        font-size: 0.95rem;
    }
    .seccion {
        color: #0B3D66;
        font-weight: 700;
        margin-top: 1rem;
        border-bottom: 2px solid #2E5C8A;
        padding-bottom: 0.3rem;
    }
    div.stButton > button {
        background-color: #0B3D66;
        color: #FFFFFF;
        font-weight: 700;
        padding: 0.6rem 1.5rem;
        border-radius: 6px;
        border: none;
        width: 100%;
    }
    div.stButton > button:hover {
        background-color: #2E5C8A;
        color: #FFFFFF;
    }
    .caja-resultado {
        padding: 1.2rem 1.5rem;
        border-radius: 8px;
        margin-top: 1rem;
        text-align: center;
    }
    .caja-riesgo {
        background-color: #B02A2A;
    }
    .caja-estable {
        background-color: #1E7145;
    }
    .caja-resultado h2 {
        color: #FFFFFF;
        margin: 0 0 0.5rem 0;
        font-size: 1.6rem;
    }
    .caja-resultado p {
        color: #FFFFFF;
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
    }
    .etiqueta-progreso {
        color: #0B3D66;
        font-weight: 700;
        font-size: 1.1rem;
        margin: 1rem 0 0.3rem 0;
    }
    .stProgress > div > div > div {
        background-color: #0B3D66;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Encabezado del sistema
st.markdown(
    """
    <div class="encabezado">
        <h1>SIPRE</h1>
        <p>Sistema de Prediccion de Riesgo Estudiantil: estima si un
        estudiante esta en riesgo academico a partir de sus datos.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def cargar_recursos():
    """Carga el modelo y el scaler una sola vez."""
    modelo = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return modelo, scaler


st.markdown('<p class="seccion">Datos academicos del estudiante</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    studytime = st.slider("Tiempo de estudio semanal", 1, 4, 2)
    failures = st.slider("Materias reprobadas", 0, 3, 0)
    absences = st.slider("Ausencias", 0, 93, 6)
with col2:
    g1 = st.slider("Nota primer periodo (G1)", 0, 20, 10)
    g2 = st.slider("Nota segundo periodo (G2)", 0, 20, 10)

st.write("")
predecir = st.button("Predecir")

if predecir:
    modelo, scaler = cargar_recursos()

    # Mismas 5 variables, mismo orden y misma normalizacion que en el entrenamiento
    entrada = np.array([[studytime, failures, absences, g1, g2]])
    entrada_esc = scaler.transform(entrada)

    # Se acota a [0, 1] por seguridad antes de pasarlo a st.progress
    prob = float(np.clip(modelo.predict(entrada_esc, verbose=0)[0][0], 0.0, 1.0))
    riesgo = prob > 0.5

    if riesgo:
        st.markdown(
            f"""
            <div class="caja-resultado caja-riesgo">
                <h2>⚠️ RIESGO DETECTADO</h2>
                <p>{prob:.1%}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="caja-resultado caja-estable">
                <h2>✅ ESTUDIANTE ESTABLE</h2>
                <p>{prob:.1%}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Etiqueta y barra juntas, mismo valor de probabilidad
    st.markdown(
        f'<p class="etiqueta-progreso">Probabilidad de riesgo: {prob:.1%}</p>',
        unsafe_allow_html=True,
    )
    st.progress(int(round(prob * 100)))
