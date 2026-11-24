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

from resenas.api import MAX_LOTE, predecir, predecir_lote

MODELS = Path(__file__).resolve().parent / "models"
METRICAS = MODELS / "metricas.json"
COMPARACION = MODELS / "comparacion.json"
ERRORES = MODELS / "errores.csv"
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

tab_pred, tab_lote, tab_met, tab_cmp = st.tabs(
    ["Predecir", "Lote (CSV)", "Métricas del modelo", "Comparación y errores"])

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

# ---------------------------------------------------------------- Lote (CSV)
with tab_lote:
    st.write("Sube un CSV con reseñas para clasificarlas todas y ver cómo se distribuyen.")
    archivo = st.file_uploader("Archivo CSV", type="csv")
    if archivo is not None:
        try:
            df_in = pd.read_csv(archivo)
        except Exception as e:  # CSV mal formado / codificación inválida
            st.error(f"No se pudo leer el CSV: {e}")
            df_in = None

        if df_in is not None:
            if df_in.empty:
                st.warning("El archivo no tiene filas.")
            else:
                defecto = next((i for i, c in enumerate(df_in.columns)
                                if str(c).lower() in {"texto", "reseña", "resena", "review", "comentario"}), 0)
                col = st.selectbox("Columna con el texto de la reseña", df_in.columns, index=defecto)
                limite = st.number_input("Máximo de filas a clasificar", 1, MAX_LOTE,
                                         min(200, MAX_LOTE))
                if st.button("Clasificar lote", type="primary"):
                    textos = df_in[col].dropna().astype(str).str.strip()
                    textos = textos[textos.str.len() >= 3].head(int(limite))
                    if textos.empty:
                        st.warning("No hay textos válidos (mínimo 3 caracteres) en esa columna.")
                    else:
                        try:
                            with st.spinner(f"Clasificando {len(textos)} reseñas..."):
                                res = predecir_lote(textos.tolist())
                        except HTTPException as e:
                            st.error(f"{e.detail}")
                        else:
                            out = pd.DataFrame({"texto": textos.values,
                                                "etiqueta": [r["etiqueta"] for r in res],
                                                "confianza": [r["confianza"] for r in res]})
                            st.session_state["lote"] = out

    if "lote" in st.session_state:
        out = st.session_state["lote"]
        st.subheader(f"Resultados ({len(out)} reseñas)")
        conteo = out["etiqueta"].value_counts().reindex(["negativo", "neutro", "positivo"],
                                                        fill_value=0)
        st.bar_chart(conteo)
        st.caption(f"Confianza media: {out['confianza'].mean():.0%} · "
                   f"reseñas con confianza < 50 %: {(out['confianza'] < 0.5).sum()}")
        st.dataframe(out, hide_index=True, use_container_width=True)
        st.download_button("Descargar resultados (CSV)", out.to_csv(index=False).encode("utf-8"),
                           file_name="resultados.csv", mime="text/csv")

# ---------------------------------------------------------------- Métricas
with tab_met:
    if not METRICAS.exists():
        st.info("Aún no hay métricas. Entrena el modelo con `python -m resenas.train`.")
    else:
        datos = json.loads(METRICAS.read_text(encoding="utf-8"))
        rep, clases = datos["reporte"], datos["clases"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{rep['accuracy']:.3f}")
        c2.metric("F1 macro", f"{rep['macro avg']['f1-score']:.3f}")
        if "kappa" in datos:
            c3.metric("Kappa de Cohen", f"{datos['kappa']:.3f}")

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

# ---------------------------------------------------- Comparación y errores
with tab_cmp:
    if not COMPARACION.exists():
        st.info("Aún no hay comparación. Ejecuta `python -m resenas.evaluar`.")
    else:
        cmp = json.loads(COMPARACION.read_text(encoding="utf-8"))

        st.subheader("Comparación de configuraciones")
        st.caption(f"Mismo conjunto de prueba para todas ({cmp['n_test']:,} reseñas). "
                   "F1 macro pondera igual a las tres clases.")
        tabla = pd.DataFrame(cmp["comparacion"]).set_index("configuracion")
        tabla = tabla[["accuracy", "f1_macro", "mcc", "segundos"]].round(3)
        st.dataframe(tabla.sort_values("f1_macro", ascending=False),
                     use_container_width=True)
        st.bar_chart(tabla["f1_macro"])

        if cmp.get("validacion_cruzada"):
            cv = cmp["validacion_cruzada"]
            st.success(f"Validación cruzada {cv['k']}-fold (config. final): "
                       f"accuracy {cv['accuracy_media']:.3f} ± {cv['accuracy_std']:.3f} · "
                       f"F1 macro {cv['f1_macro_media']:.3f} ± {cv['f1_macro_std']:.3f}")

        st.subheader("¿Dónde se equivoca?")
        err = cmp["errores"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Errores", f"{err['n_errores']:,}")
        c2.metric("Graves (pos ↔ neg)", f"{err['n_errores_graves']:,}")
        c3.metric("Errores que involucran «neutro»",
                  f"{err['pct_errores_que_involucran_neutro']:.0%}")
        acc_est = pd.Series({f"{e}★": v["accuracy"] for e, v in err["por_estrellas"].items()},
                            name="accuracy")
        st.caption("Accuracy según las estrellas reales de la reseña")
        st.bar_chart(acc_est)

        if ERRORES.exists():
            st.subheader("Fallos más «seguros» del modelo")
            st.caption("Errores con mayor confianza: suelen ser etiquetas ruidosas, ironía o mezcla de opiniones.")
            st.dataframe(pd.read_csv(ERRORES)[["texto", "estrellas", "clase", "prediccion", "confianza"]],
                         hide_index=True, use_container_width=True)
