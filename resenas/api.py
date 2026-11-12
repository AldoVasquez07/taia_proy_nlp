from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from resenas.preprocess import preprocesar

MODELO = Path(__file__).resolve().parent.parent / "models" / "modelo.joblib"

app = FastAPI(title="Clasificador de reseñas en español",
              description="TF-IDF + Regresión Logística sobre reseñas de Mercado Libre (MeLiSA).")


class Entrada(BaseModel):
    texto: str = Field(..., min_length=3, examples=["El producto llegó roto y no funciona"])


@lru_cache(maxsize=1)
def _cargar():
    if not MODELO.exists():
        raise HTTPException(503, "Modelo no entrenado. Ejecuta: python -m resenas.train")
    return joblib.load(MODELO)


def predecir(texto: str, top_k: int = 5) -> dict:
    art = _cargar()
    vec, clf = art["vectorizador"], art["modelo"]
    limpio = " ".join(preprocesar(texto))
    X = vec.transform([limpio])
    probas = clf.predict_proba(X)[0]
    idx = int(np.argmax(probas))

    # Términos que más empujaron hacia la clase predicha: peso TF-IDF * coeficiente
    contrib = X.toarray().ravel() * clf.coef_[idx]
    nombres = vec.get_feature_names_out()
    top = [i for i in np.argsort(contrib)[::-1][:top_k] if contrib[i] > 0]
    return {
        "etiqueta": str(clf.classes_[idx]),
        "probabilidades": {str(c): round(float(p), 4) for c, p in zip(clf.classes_, probas)},
        "terminos_influyentes": [{"termino": nombres[i], "peso": round(float(contrib[i]), 4)}
                                 for i in top],
        "texto_preprocesado": limpio,
    }


@app.get("/salud")
def salud():
    return {"estado": "ok", "modelo_entrenado": MODELO.exists()}


@app.post("/predecir")
def endpoint_predecir(entrada: Entrada):
    return predecir(entrada.texto)
