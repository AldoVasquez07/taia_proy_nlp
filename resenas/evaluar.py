"""Evaluación completa del proyecto (Evidencia 4).

Hace tres cosas sobre la MISMA partición 80/20 que usa `train.py`:
  1. Ablación: compara preprocesamiento (lemas spaCy vs. texto normalizado vs. marcado de
     negaciones) × representación (BoW vs. TF-IDF, 1 vs. 1-2 gramas) × clasificador.
  2. Validación cruzada estratificada (K-Fold) de la configuración final.
  3. Análisis de errores: errores por número de estrellas y los fallos más "seguros".

Salidas (en `models/`): `comparacion.json` y `errores.csv`.

Uso:  python -m resenas.evaluar [--cv 5]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from resenas.preprocess import limpiar, normalizar_lote, preprocesar_lote
from resenas.train import (CLASES, DATOS, RAIZ, SEMILLA, cargar_datos, crear_clasificador,
                           crear_vectorizador, dividir)

SALIDA_JSON = RAIZ / "models" / "comparacion.json"
SALIDA_ERRORES = RAIZ / "models" / "errores.csv"

def _tfidf(ng):
    return lambda: crear_vectorizador(ng)


def _bow():
    return CountVectorizer(min_df=2, ngram_range=(1, 2), max_features=200_000)


_LR = crear_clasificador
FINAL = "Normalizado + negaciones + TF-IDF(1-2) + LogReg  [FINAL]"

# Nombre -> (variante de texto, fábrica del vectorizador, fábrica del clasificador).
# Orden: de la línea base de la Evidencia 3 hacia el modelo final, y luego otros clasificadores.
CONFIGURACIONES = {
    "Lemas sin stopwords + TF-IDF(1-2) + LogReg  [base Evid. 3]": (
        "lemas", _tfidf((1, 2)), _LR),
    "Lemas con stopwords + TF-IDF(1-2) + LogReg": ("lemas_con_stop", _tfidf((1, 2)), _LR),
    "Normalizado + BoW(1-2) + LogReg": ("normalizado", _bow, _LR),
    "Normalizado + TF-IDF(1) + LogReg": ("normalizado", _tfidf((1, 1)), _LR),
    "Normalizado + TF-IDF(1-2) + LogReg": ("normalizado", _tfidf((1, 2)), _LR),
    "Normalizado + negaciones + TF-IDF(1) + LogReg": (
        "negaciones", _tfidf((1, 1)), _LR),
    FINAL: ("negaciones", _tfidf((1, 2)), _LR),
    "Final + emojis como tokens": ("neg_emojis", _tfidf((1, 2)), _LR),
    "Final + alargamientos colapsados": ("neg_alarg", _tfidf((1, 2)), _LR),
    "Final + sin tildes": ("neg_tildes", _tfidf((1, 2)), _LR),
    "Final + emojis + alargamientos + sin tildes": ("neg_todo", _tfidf((1, 2)), _LR),
    "Normalizado + negaciones + TF-IDF(1-2) + ComplementNB": (
        "negaciones", _tfidf((1, 2)), lambda: ComplementNB()),
    "Normalizado + negaciones + TF-IDF(1-2) + LinearSVC": (
        "negaciones", _tfidf((1, 2)), lambda: LinearSVC(class_weight="balanced")),
}


def preparar_textos(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Las variantes de texto que se comparan (se calculan una sola vez).
      normalizado    limpieza + minúsculas
      negaciones     normalizado + marcado de ámbito de negación  (= modelo final)
      lemas          spaCy: lemas, sin stopwords (salvo negaciones)  (= base de la Evid. 3)
      lemas_con_stop spaCy: lemas, conservando stopwords
      neg_*          modelo final + emojis / alargamientos / sin tildes / las tres"""
    textos = df["texto"].tolist()
    print("Preprocesando con spaCy (una sola vez, tarda unos minutos)...")
    serie = lambda xs: pd.Series(xs, index=df.index)
    return {
        "normalizado": serie([limpiar(t).lower() for t in textos]),
        "negaciones": serie(normalizar_lote(textos)),
        "neg_emojis": serie(normalizar_lote(textos, emojis=True)),
        "neg_alarg": serie(normalizar_lote(textos, alargamientos=True)),
        "neg_tildes": serie(normalizar_lote(textos, tildes=True)),
        "neg_todo": serie(normalizar_lote(textos, emojis=True, alargamientos=True,
                                          tildes=True)),
        "lemas": serie(preprocesar_lote(textos)),
        "lemas_con_stop": serie(preprocesar_lote(textos, quitar_stop=False)),
    }


def metricas_resumen(y_real, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_real, y_pred),
        "f1_macro": f1_score(y_real, y_pred, average="macro", labels=CLASES, zero_division=0),
        "f1_por_clase": dict(zip(CLASES, f1_score(y_real, y_pred, average=None,
                                                   labels=CLASES, zero_division=0))),
        "mcc": matthews_corrcoef(y_real, y_pred),
    }


def comparar(df: pd.DataFrame, textos: dict[str, pd.Series], i_tr, i_te) -> list[dict]:
    filas = []
    for nombre, (variante, f_vec, f_clf) in CONFIGURACIONES.items():
        t0 = time.time()
        X = textos[variante]
        vec, clf = f_vec(), f_clf()
        clf.fit(vec.fit_transform(X[i_tr]), df["clase"][i_tr])
        pred = clf.predict(vec.transform(X[i_te]))
        fila = {"configuracion": nombre, **metricas_resumen(df["clase"][i_te], pred),
                "segundos": round(time.time() - t0, 1)}
        filas.append(fila)
        print(f"  {nombre:<58} acc={fila['accuracy']:.3f}  F1m={fila['f1_macro']:.3f}")
    return filas


def validacion_cruzada(df: pd.DataFrame, textos: dict[str, pd.Series], k: int) -> dict:
    """K-Fold estratificado de la configuración final (el vectorizador se reajusta por fold)."""
    variante, f_vec, f_clf = CONFIGURACIONES[FINAL]
    pipe = make_pipeline(f_vec(), f_clf())
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEMILLA)
    res = cross_validate(pipe, textos[variante], df["clase"], cv=cv,
                         scoring=["accuracy", "f1_macro"], n_jobs=1)
    return {"k": k,
            "accuracy_media": float(np.mean(res["test_accuracy"])),
            "accuracy_std": float(np.std(res["test_accuracy"])),
            "f1_macro_media": float(np.mean(res["test_f1_macro"])),
            "f1_macro_std": float(np.std(res["test_f1_macro"]))}


def analisis_errores(df: pd.DataFrame, textos: dict[str, pd.Series], i_tr, i_te,
                     top: int = 30) -> tuple[dict, pd.DataFrame]:
    """Reentrena la config final y estudia dónde y cómo se equivoca."""
    variante, f_vec, f_clf = CONFIGURACIONES[FINAL]
    vec, clf = f_vec(), f_clf()
    clf.fit(vec.fit_transform(textos[variante][i_tr]), df["clase"][i_tr])
    X = vec.transform(textos[variante][i_te])
    probas = clf.predict_proba(X)
    pred = clf.classes_[probas.argmax(axis=1)]

    test = df.loc[i_te, ["texto", "estrellas", "clase"]].copy()
    test["prediccion"] = pred
    test["confianza"] = probas.max(axis=1)
    test["acierto"] = test["clase"] == test["prediccion"]

    por_estrellas = {int(e): {"n": int(len(g)), "accuracy": float(g["acierto"].mean())}
                     for e, g in test.groupby("estrellas")}

    # "Error grave" = positivo<->negativo; el resto son errores adyacentes (con el neutro)
    errores = test[~test["acierto"]]
    graves = errores[(errores["clase"] == "positivo") & (errores["prediccion"] == "negativo")
                     | (errores["clase"] == "negativo") & (errores["prediccion"] == "positivo")]
    resumen = {
        "por_estrellas": por_estrellas,
        "n_errores": int(len(errores)),
        "n_errores_graves": int(len(graves)),
        # Sin errores no hay media que calcular (evita NaN, que rompería el JSON)
        "pct_errores_que_involucran_neutro": float(
            ((errores["clase"] == "neutro") | (errores["prediccion"] == "neutro")).mean())
        if len(errores) else 0.0,
    }
    peores = errores.sort_values("confianza", ascending=False).head(top)
    return resumen, peores


def main(k_cv: int, ruta_datos: Path = DATOS) -> dict:
    df = cargar_datos(ruta_datos)
    i_tr, i_te = dividir(df)
    textos = preparar_textos(df)

    print("\n== 1. Comparación de configuraciones (partición hold-out 80/20) ==")
    filas = comparar(df, textos, i_tr, i_te)

    cv = None
    if k_cv > 1:
        print(f"\n== 2. Validación cruzada ({k_cv}-fold) de la configuración final ==")
        cv = validacion_cruzada(df, textos, k_cv)
        print(f"  accuracy={cv['accuracy_media']:.3f}±{cv['accuracy_std']:.3f}  "
              f"F1 macro={cv['f1_macro_media']:.3f}±{cv['f1_macro_std']:.3f}")

    print("\n== 3. Análisis de errores ==")
    errores, peores = analisis_errores(df, textos, i_tr, i_te)
    print(f"  errores={errores['n_errores']}  graves(pos<->neg)={errores['n_errores_graves']}  "
          f"involucran 'neutro'={errores['pct_errores_que_involucran_neutro']:.0%}")

    salida = {"comparacion": filas, "validacion_cruzada": cv, "errores": errores,
              "n_total": int(len(df)), "n_test": int(len(i_te))}
    SALIDA_JSON.parent.mkdir(parents=True, exist_ok=True)
    SALIDA_JSON.write_text(json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8")
    peores.to_csv(SALIDA_ERRORES, index=False, encoding="utf-8")
    print(f"\nGuardado: {SALIDA_JSON.name}, {SALIDA_ERRORES.name} (en models/)")
    return salida


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cv", type=int, default=5, help="folds de validación cruzada (0 = omitir)")
    main(p.parse_args().cv)