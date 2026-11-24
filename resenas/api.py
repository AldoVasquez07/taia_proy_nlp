import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from resenas.preprocess import normalizar, normalizar_lote

RAIZ = Path(__file__).resolve().parent.parent
MODELO = RAIZ / "models" / "modelo.joblib"
METRICAS = RAIZ / "models" / "metricas.json"
MAX_LOTE = 1000

app = FastAPI(title="Clasificador de reseñas en español",
              description="TF-IDF + Regresión Logística sobre reseñas de Mercado Libre (MeLiSA).")


class Entrada(BaseModel):
    texto: str = Field(..., min_length=3, examples=["El producto llegó roto y no funciona"])


class EntradaLote(BaseModel):
    textos: list[str] = Field(..., min_length=1, max_length=MAX_LOTE)


@lru_cache(maxsize=1)
def _cargar():
    if not MODELO.exists():
        raise HTTPException(503, "Modelo no entrenado. Ejecuta: python -m resenas.train")
    return joblib.load(MODELO)


def _terminos(vec, clf, fila, idx: int, top_k: int) -> list[dict]:
    """Términos que más empujaron hacia la clase predicha: peso TF-IDF * coeficiente."""
    contrib = fila.toarray().ravel() * clf.coef_[idx]
    nombres = vec.get_feature_names_out()
    top = [i for i in np.argsort(contrib)[::-1][:top_k] if contrib[i] > 0]
    return [{"termino": nombres[i], "peso": round(float(contrib[i]), 4)} for i in top]


def predecir(texto: str, top_k: int = 5) -> dict:
    art = _cargar()
    vec, clf = art["vectorizador"], art["modelo"]
    limpio = normalizar(texto)
    X = vec.transform([limpio])
    probas = clf.predict_proba(X)[0]
    idx = int(np.argmax(probas))
    return {
        "etiqueta": str(clf.classes_[idx]),
        "probabilidades": {str(c): round(float(p), 4) for c, p in zip(clf.classes_, probas)},
        "terminos_influyentes": _terminos(vec, clf, X, idx, top_k),
        "texto_preprocesado": limpio,
    }


def predecir_lote(textos: list[str]) -> list[dict]:
    """Versión vectorizada (una sola llamada al modelo) para muchos textos.
    Devuelve etiqueta, confianza y probabilidades; sin términos influyentes (más liviano)."""
    art = _cargar()
    vec, clf = art["vectorizador"], art["modelo"]
    probas = clf.predict_proba(vec.transform(normalizar_lote(textos)))
    salida = []
    for fila in probas:
        idx = int(np.argmax(fila))
        salida.append({
            "etiqueta": str(clf.classes_[idx]),
            "confianza": round(float(fila[idx]), 4),
            "probabilidades": {str(c): round(float(p), 4) for c, p in zip(clf.classes_, fila)},
        })
    return salida


@app.get("/salud")
def salud():
    return {"estado": "ok", "modelo_entrenado": MODELO.exists()}


@app.post("/predecir")
def endpoint_predecir(entrada: Entrada):
    return predecir(entrada.texto)


@app.post("/predecir_lote")
def endpoint_predecir_lote(entrada: EntradaLote):
    textos = [t.strip() for t in entrada.textos]
    if any(len(t) < 3 for t in textos):
        raise HTTPException(422, "Todos los textos deben tener al menos 3 caracteres.")
    return {"n": len(textos), "resultados": predecir_lote(textos)}


@app.get("/metricas")
def endpoint_metricas():
    if not METRICAS.exists():
        raise HTTPException(404, "Aún no hay métricas. Ejecuta: python -m resenas.train")
    return json.loads(METRICAS.read_text(encoding="utf-8"))
