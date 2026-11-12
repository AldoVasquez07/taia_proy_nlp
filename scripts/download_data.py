import argparse
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

REPO = "lpsc-fiuba/melisa"
REVISION = "701a2f09c91fcbf1d7da0c485c6482e920dd9aa8"  # commit con los datos completos
ARCHIVO = "es/train.csv"
SALIDA = Path(__file__).resolve().parent.parent / "data" / "raw" / "reviews.csv"

COL_TEXTO = "review_content"
COL_ESTRELLAS = "review_rate"


def main(max_filas: int) -> None:
    print(f"Descargando {ARCHIVO} de {REPO} (revisión {REVISION[:7]})...")
    ruta = hf_hub_download(repo_id=REPO, filename=ARCHIVO, repo_type="dataset",
                           revision=REVISION)
    df = pd.read_csv(ruta)
    print("Columnas:", list(df.columns), "| filas:", len(df))

    faltan = {COL_TEXTO, COL_ESTRELLAS} - set(df.columns)
    if faltan:
        raise KeyError(f"Faltan columnas {faltan}. Columnas reales: {list(df.columns)}")

    df = df[[COL_TEXTO, COL_ESTRELLAS]].rename(
        columns={COL_TEXTO: "texto", COL_ESTRELLAS: "estrellas"})
    df = df.dropna().drop_duplicates(subset="texto")

    if len(df) < 1000:
        print("AVISO: muy pocas filas; probablemente es la versión demo. "
              "Revisa REVISION en este script.")

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