"""Baseline con Transformer: comparación OFFLINE (no forma parte de la API ni del dashboard).

Compara el modelo final (TF-IDF + regresión logística) con un Transformer preentrenado para
sentimiento en español, sobre la MISMA submuestra del conjunto de prueba.

OJO con lo que mide: el Transformer NO se reentrena con MeLiSA. Fue ajustado con otros datos
(tweets), así que esto mide cuánto se transfiere a reseñas sin entrenar. No es un fine-tuning.

Uso (requiere `pip install -r requirements-baseline.txt` y haber corrido `resenas.train`):
    python -m resenas.baseline_transformer --n 50      # prueba rápida: verifica que todo corre
    python -m resenas.baseline_transformer --n 2000    # corrida real
    python -m resenas.baseline_transformer --modelo finiteautomata/beto-sentiment-analysis
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from resenas.preprocess import normalizar_lote
from resenas.train import CLASES, DATOS, MODELO, SEMILLA, calcular_metricas, cargar_datos, dividir

RAIZ = Path(__file__).resolve().parent.parent
SALIDA = RAIZ / "models" / "baseline_transformer.json"
MODELO_HF = "pysentimiento/robertuito-sentiment-analysis"

# Etiquetas que usan los modelos de sentimiento en español de pysentimiento / finiteautomata
_ETIQUETAS = {"NEG": "negativo", "NEU": "neutro", "POS": "positivo",
              "NEGATIVE": "negativo", "NEUTRAL": "neutro", "POSITIVE": "positivo"}


def mapear_etiqueta(etiqueta: str) -> str:
    """'POS' -> 'positivo'. Falla con un mensaje claro si el modelo usa otras etiquetas."""
    try:
        return _ETIQUETAS[str(etiqueta).strip().upper()]
    except KeyError:
        raise ValueError(f"Etiqueta desconocida del modelo: {etiqueta!r}. "
                         f"Agrégala a _ETIQUETAS en baseline_transformer.py.") from None


def submuestra_prueba(df, idx_test, n: int, semilla: int = SEMILLA) -> np.ndarray:
    """n reseñas del conjunto de prueba, estratificadas por clase y reproducibles."""
    idx_test = np.asarray(idx_test)
    if n >= len(idx_test):
        return np.sort(idx_test)
    sub, _ = train_test_split(idx_test, train_size=n, random_state=semilla,
                              stratify=df.loc[idx_test, "clase"])
    return np.sort(sub)


def crear_predictor_clasico(ruta_modelo: Path = MODELO):
    """Predictor del modelo final (el mismo pipeline que usa la API)."""
    art = joblib.load(ruta_modelo)

    def predecir(textos: list[str]) -> list[str]:
        return art["modelo"].predict(art["vectorizador"].transform(normalizar_lote(textos))).tolist()
    return predecir


def crear_predictor_transformer(modelo_id: str = MODELO_HF, batch_size: int = 16,
                                max_length: int = 128, dispositivo: str | None = None):
    """Predictor con un Transformer de Hugging Face. Los imports pesados son perezosos para que
    el resto del módulo (y sus pruebas) funcione sin torch/transformers instalados."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dispositivo = dispositivo or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizador = AutoTokenizer.from_pretrained(modelo_id)
    modelo = AutoModelForSequenceClassification.from_pretrained(modelo_id).to(dispositivo).eval()
    etiquetas = [mapear_etiqueta(modelo.config.id2label[i]) for i in range(modelo.config.num_labels)]

    def predecir(textos: list[str]) -> list[str]:
        salida: list[str] = []
        with torch.inference_mode():
            for i in range(0, len(textos), batch_size):
                enc = tokenizador(textos[i:i + batch_size], truncation=True, max_length=max_length,
                                  padding=True, return_tensors="pt").to(dispositivo)
                salida.extend(etiquetas[j] for j in modelo(**enc).logits.argmax(dim=-1).tolist())
        return salida

    predecir.info = {"dispositivo": dispositivo,
                     "parametros": sum(p.numel() for p in modelo.parameters())}
    return predecir


def _medir(predictor, textos: list[str]) -> tuple[list[str], float]:
    """(predicciones, milisegundos por reseña). Es rendimiento por lotes, no latencia de una
    sola petición."""
    t0 = time.perf_counter()
    pred = predictor(textos)
    return pred, (time.perf_counter() - t0) * 1000 / len(textos)


def bootstrap_diferencia_f1(y, pred_a, pred_b, n_boot: int = 1000, semilla: int = SEMILLA) -> dict:
    """Diferencia de F1 macro (B - A) con IC 95 % por bootstrap pareado (mismas reseñas)."""
    y, a, b = np.asarray(y), np.asarray(pred_a), np.asarray(pred_b)

    def f1(real, pred):
        return f1_score(real, pred, labels=CLASES, average="macro", zero_division=0)

    rng = np.random.default_rng(semilla)
    difs = np.empty(n_boot)
    for k in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        difs[k] = f1(y[i], b[i]) - f1(y[i], a[i])
    bajo, alto = np.percentile(difs, [2.5, 97.5])
    return {"diferencia_f1_macro": float(f1(y, b) - f1(y, a)),
            "ic95": [float(bajo), float(alto)],
            "significativa": bool(bajo > 0 or alto < 0)}


def ejecutar(n: int = 2000, modelo_id: str = MODELO_HF, batch_size: int = 16,
             ruta_datos: Path = DATOS, ruta_modelo: Path = MODELO, salida: Path = SALIDA,
             predictor_transformer=None, n_boot: int = 1000) -> dict:
    df = cargar_datos(ruta_datos)
    _, idx_test = dividir(df)                       # la MISMA partición que train.py
    idx = submuestra_prueba(df, idx_test, n)
    textos = df.loc[idx, "texto"].astype(str).tolist()
    y = df.loc[idx, "clase"].tolist()

    pred_c, ms_c = _medir(crear_predictor_clasico(ruta_modelo), textos)
    if predictor_transformer is None:
        print(f"Cargando {modelo_id} ...")
        predictor_transformer = crear_predictor_transformer(modelo_id, batch_size)
    print(f"Clasificando {len(textos)} reseñas con el Transformer...")
    pred_t, ms_t = _medir(predictor_transformer, textos)

    resultado = {
        "modelo_transformer": modelo_id, "n": len(idx), "semilla": SEMILLA,
        "clasico": {**calcular_metricas(y, pred_c), "ms_por_resena": ms_c},
        "transformer": {**calcular_metricas(y, pred_t), "ms_por_resena": ms_t,
                        **getattr(predictor_transformer, "info", {})},
        "comparacion": bootstrap_diferencia_f1(y, pred_c, pred_t, n_boot=n_boot),
        "advertencias": [],
    }

    # Control: en la submuestra el modelo clásico debe rendir parecido a su métrica guardada
    ruta_metricas = ruta_modelo.parent / "metricas.json"
    if ruta_metricas.exists():
        guardada = json.loads(ruta_metricas.read_text(encoding="utf-8"))["reporte"]["accuracy"]
        if abs(accuracy_score(y, pred_c) - guardada) > 0.05:
            resultado["advertencias"].append(
                "La exactitud del modelo clásico en la submuestra difiere >5 puntos de la guardada "
                "en metricas.json: ¿el modelo se entrenó con otros datos o partición?")

    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    _imprimir(resultado)
    return resultado


def _imprimir(r: dict) -> None:
    print(f"\nComparación sobre {r['n']} reseñas de prueba (misma muestra para ambos)\n")
    print(f"{'':<22}{'Exactitud':>10}{'F1 macro':>10}{'F1 neutro':>11}{'ms/reseña':>11}")
    for nombre, clave in (("TF-IDF + LogReg", "clasico"), ("Transformer", "transformer")):
        m = r[clave]["reporte"]
        print(f"{nombre:<22}{m['accuracy']:>10.3f}{m['macro avg']['f1-score']:>10.3f}"
              f"{m['neutro']['f1-score']:>11.3f}{r[clave]['ms_por_resena']:>11.1f}")
    c = r["comparacion"]
    veredicto = "SIGNIFICATIVA" if c["significativa"] else "no concluyente"
    print(f"\nF1 macro (Transformer - clásico): {c['diferencia_f1_macro']:+.3f}  "
          f"IC95% [{c['ic95'][0]:+.3f}, {c['ic95'][1]:+.3f}]  -> {veredicto}")
    for a in r["advertencias"]:
        print("ADVERTENCIA:", a)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=2000, help="reseñas de la submuestra de prueba")
    ap.add_argument("--modelo", default=MODELO_HF, help="id del modelo en Hugging Face")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    ejecutar(n=args.n, modelo_id=args.modelo, batch_size=args.batch)