import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from resenas.preprocess import preprocesar_lote

RAIZ = Path(__file__).resolve().parent.parent
DATOS = RAIZ / "data" / "raw" / "reviews.csv"
MODELO = RAIZ / "models" / "modelo.joblib"
METRICAS = RAIZ / "models" / "metricas.json"
CLASES = ["negativo", "neutro", "positivo"]


def estrellas_a_clase(e: int) -> str:
    e = int(e)
    return "negativo" if e <= 2 else "neutro" if e == 3 else "positivo"


def entrenar(ruta_datos: Path = DATOS, ruta_modelo: Path = MODELO) -> dict:
    df = pd.read_csv(ruta_datos).dropna()
    df["clase"] = df["estrellas"].map(estrellas_a_clase)
    print("Distribución de clases:\n", df["clase"].value_counts())

    print("Preprocesando (spaCy puede tardar unos minutos)...")
    X = preprocesar_lote(df["texto"].tolist())
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, df["clase"], test_size=0.2, random_state=42, stratify=df["clase"])

    vec = TfidfVectorizer(min_df=2, ngram_range=(1, 2), max_features=100_000)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(vec.fit_transform(X_tr), y_tr)

    pred = clf.predict(vec.transform(X_te))
    reporte = classification_report(y_te, pred, output_dict=True, zero_division=0)
    matriz = confusion_matrix(y_te, pred, labels=CLASES).tolist()
    print(classification_report(y_te, pred, zero_division=0))

    ruta_modelo.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizador": vec, "modelo": clf}, ruta_modelo)
    resultado = {"reporte": reporte, "matriz_confusion": matriz, "clases": CLASES}
    METRICAS.write_text(json.dumps(resultado, indent=2, ensure_ascii=False))
    print(f"Modelo guardado en {ruta_modelo}")
    return resultado


if __name__ == "__main__":
    entrenar()
