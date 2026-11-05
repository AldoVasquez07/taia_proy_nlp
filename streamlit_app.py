"""Interfaz web (Streamlit) del clasificador de reseñas.

Uso (desde la raíz del proyecto, donde está esta carpeta `resenas/`):
    streamlit run streamlit_app.py
Requiere haber entrenado antes el modelo:  python -m resenas.train
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st
from fastapi import HTTPException

from resenas.api import predecir

METRICAS = Path(__file__).resolve().parent / "models" / "metricas.json"
ICONOS = {"negativo": "😠", "neutro": "😐", "positivo": "😀"}
EJEMPLOS = [
    "El producto llegó roto y no funciona",
    "Excelente calidad, llegó rápido y muy bien empacado",
    "Está bien, cumple pero nada especial",
    "No lo recomiendo, la batería no dura nada",
]

st.set_page_config(page_title="Clasificador de reseñas", page_icon="🛒")
st.title("🛒 Clasificador de reseñas en español")
st.caption("TF-IDF + Regresión Logística · entrenado con reseñas de Mercado Libre (MeLiSA)")

tab_pred, tab_met = st.tabs(["Predecir", "Métricas del modelo"])

# ---------------------------------------------------------------- Predecir
with tab_pred:
    st.write("Prueba con un ejemplo o escribe tu propia reseña:")
    cols = st.columns(2)
    for i, ej in enumerate(EJEMPLOS):
        cols[i % 2].button(ej, key=f"ej{i}", use_container_width=True,
                           on_click=lambda t=ej: st.session_state.update(texto=t))

    texto = st.text_area("Reseña", key="texto", height=130,
                         placeholder="Ej: El vendedor fue amable pero el producto vino sin accesorios")

    if st.button("Clasificar", type="primary"):
        if len(texto.strip()) < 3:
            st.warning("Escribe al menos unas palabras.")
        else:
            try:
                res = predecir(texto)
            except HTTPException as e:
                st.error(f"{e.detail}")
            else:
                etiqueta = res["etiqueta"]
                confianza = res["probabilidades"][etiqueta]
                c1, c2 = st.columns(2)
                c1.metric("Sentimiento", f"{ICONOS.get(etiqueta, '')} {etiqueta}")
                c2.metric("Confianza", f"{confianza:.0%}")

                st.subheader("Probabilidades")
                st.bar_chart(pd.Series(res["probabilidades"], name="probabilidad"))

                st.subheader("Términos que más pesaron")
                if res["terminos_influyentes"]:
                    df_t = pd.DataFrame(res["terminos_influyentes"])
                    st.dataframe(df_t.rename(columns={"termino": "término"}),
                                 hide_index=True, use_container_width=True)
                else:
                    st.info("Ningún término del texto tuvo peso a favor de la clase predicha.")

                with st.expander("Texto después del preprocesamiento"):
                    st.code(res["texto_preprocesado"] or "(vacío)")

# ---------------------------------------------------------------- Métricas
with tab_met:
    if not METRICAS.exists():
        st.info("Aún no hay métricas. Entrena el modelo con `python -m resenas.train`.")
    else:
        datos = json.loads(METRICAS.read_text(encoding="utf-8"))
        rep, clases = datos["reporte"], datos["clases"]

        c1, c2 = st.columns(2)
        c1.metric("Accuracy", f"{rep['accuracy']:.3f}")
        c2.metric("F1 macro", f"{rep['macro avg']['f1-score']:.3f}")

        st.subheader("Por clase")
        filas = {c: {k: round(rep[c][k], 3) for k in ("precision", "recall", "f1-score")}
                 for c in clases}
        st.dataframe(pd.DataFrame(filas).T, use_container_width=True)

        st.subheader("Matriz de confusión")
        st.caption("Filas = clase real · columnas = clase predicha")
        st.dataframe(pd.DataFrame(datos["matriz_confusion"],
                                  index=[f"real: {c}" for c in clases],
                                  columns=[f"pred: {c}" for c in clases]),
                     use_container_width=True)