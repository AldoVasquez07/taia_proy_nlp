import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, cohen_kappa_score,
                             confusion_matrix, matthews_corrcoef)
from sklearn.model_selection import train_test_split

from resenas.preprocess import normalizar_lote

RAIZ = Path(__file__).resolve().parent.parent
DATOS = RAIZ / "data" / "raw" / "reviews.csv"
MODELO = RAIZ / "models" / "modelo.joblib"
CLASES = ["negativo", "neutro", "positivo"]
SEMILLA = 42


def crear_vectorizador(ngram_range=(1, 2)) -> TfidfVectorizer:
    """TF-IDF final. `sublinear_tf` (1 + log tf) evita que una palabra repetida domine."""
    return TfidfVectorizer(min_df=2, ngram_range=ngram_range, max_features=200_000,
                           sublinear_tf=True)


def crear_clasificador() -> LogisticRegression:
    return LogisticRegression(max_iter=2000, class_weight="balanced")


def estrellas_a_clase(e: int) -> str:
    e = int(e)
    return "negativo" if e <= 2 else "neutro" if e == 3 else "positivo"


def cargar_datos(ruta_datos: Path = DATOS) -> pd.DataFrame:
    """Lee el CSV (columnas `texto`, `estrellas`), quita nulos y agrega la columna `clase`."""
    df = pd.read_csv(ruta_datos).dropna(subset=["texto", "estrellas"]).reset_index(drop=True)
    df["clase"] = df["estrellas"].map(estrellas_a_clase)
    return df


def dividir(df: pd.DataFrame, test_size: float = 0.2):
    """Partición estratificada y reproducible. Devuelve (índices_train, índices_test)."""
    return train_test_split(df.index.to_numpy(), test_size=test_size,
                            random_state=SEMILLA, stratify=df["clase"])


def calcular_metricas(y_real, y_pred) -> dict:
    """Métricas de clasificación usadas en la app, la evaluación y el informe."""
    reporte = classification_report(y_real, y_pred, labels=CLASES, output_dict=True,
                                    zero_division=0)
    return {
        "reporte": reporte,
        "matriz_confusion": confusion_matrix(y_real, y_pred, labels=CLASES).tolist(),
        "clases": CLASES,
        "kappa": cohen_kappa_score(y_real, y_pred),
        "mcc": matthews_corrcoef(y_real, y_pred),
    }


def entrenar(ruta_datos: Path = DATOS, ruta_modelo: Path = MODELO,
             ruta_metricas: Path | None = None) -> dict:
    # Por defecto, las métricas se guardan junto al modelo
    ruta_metricas = ruta_metricas or ruta_modelo.parent / "metricas.json"

    df = cargar_datos(ruta_datos)
    print("Distribución de clases:\n", df["clase"].value_counts())

    print("Normalizando textos...")
    X = pd.Series(normalizar_lote(df["texto"].tolist()), index=df.index)
    i_tr, i_te = dividir(df)

    vec, clf = crear_vectorizador(), crear_clasificador()
    clf.fit(vec.fit_transform(X[i_tr]), df["clase"][i_tr])

    pred = clf.predict(vec.transform(X[i_te]))
    resultado = calcular_metricas(df["clase"][i_te], pred)
    resultado["n_train"], resultado["n_test"] = len(i_tr), len(i_te)
    print(classification_report(df["clase"][i_te], pred, zero_division=0))

    ruta_modelo.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizador": vec, "modelo": clf}, ruta_modelo)
    ruta_metricas.write_text(json.dumps(resultado, indent=2, ensure_ascii=False),
                             encoding="utf-8")
    print(f"Modelo guardado en {ruta_modelo}")
    return resultado


if __name__ == "__main__":
    entrenar()
