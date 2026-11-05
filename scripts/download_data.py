"""Descarga el dataset MeLiSA (reseñas de Mercado Libre en español) y lo guarda
como data/raw/reviews.csv con las columnas: texto, estrellas.

Uso:
    python scripts/download_data.py
    python scripts/download_data.py --max-filas 60000

Licencia: MeLiSA se distribuye bajo el acuerdo de Amazon
(https://docs.opendata.aws/amazon-reviews-ml/license.txt). Úsalo solo con fines
académicos y cita la fuente en el informe.
"""
import argparse
from pathlib import Path

import pandas as pd

DATASET = "lpsc-fiuba/melisa"
SALIDA = Path(__file__).resolve().parent.parent / "data" / "raw" / "reviews.csv"

# Nombres de columna posibles según la versión del dataset
COLS_TEXTO = ["review_body", "review_content", "text", "review"]
COLS_ESTRELLAS = ["review_rate", "star_rating", "rating", "stars", "label"]


def primera_columna(df: pd.DataFrame, candidatas: list[str]) -> str:
    for c in candidatas:
        if c in df.columns:
            return c
    raise KeyError(f"No encontré ninguna de {candidatas}. Columnas: {list(df.columns)}")


def main(max_filas: int) -> None:
    from datasets import load_dataset  # import tardío: solo se necesita aquí

    print(f"Descargando {DATASET} (español)...")
    ds = load_dataset(DATASET, "es")
    split = "train" if "train" in ds else list(ds.keys())[0]
    df = ds[split].to_pandas()
    print("Columnas disponibles:", list(df.columns))

    c_texto = primera_columna(df, COLS_TEXTO)
    c_est = primera_columna(df, COLS_ESTRELLAS)
    df = df[[c_texto, c_est]].rename(columns={c_texto: "texto", c_est: "estrellas"})
    df = df.dropna().drop_duplicates(subset="texto")

    if len(df) > max_filas:
        df = df.sample(max_filas, random_state=42)

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SALIDA, index=False)
    print(f"Guardado: {SALIDA} ({len(df)} filas)")
    print(df["estrellas"].value_counts().sort_index())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--max-filas", type=int, default=60000)
    main(p.parse_args().max_filas)
