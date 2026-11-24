import json

import joblib
import numpy as np
import pytest

from resenas import train
from resenas.train import CLASES, calcular_metricas, cargar_datos, dividir, estrellas_a_clase


@pytest.mark.parametrize("estrellas, clase", [
    (1, "negativo"), (2, "negativo"), (3, "neutro"), (4, "positivo"), (5, "positivo"),
    ("5", "positivo"), (2.0, "negativo"),
])
def test_estrellas_a_clase(estrellas, clase):
    assert estrellas_a_clase(estrellas) == clase


def test_cargar_datos_agrega_clase_y_quita_nulos(tmp_path, df_sintetico):
    sucio = df_sintetico.copy()
    sucio.loc[0, "texto"] = None
    sucio.loc[1, "estrellas"] = None
    ruta = tmp_path / "sucio.csv"
    sucio.to_csv(ruta, index=False)

    df = cargar_datos(ruta)
    assert len(df) == len(df_sintetico) - 2
    assert set(df["clase"]) == set(CLASES)
    assert list(df.index) == list(range(len(df)))  # índice limpio: lo asumen train/evaluar


def test_dividir_es_estratificada_y_reproducible(csv_sintetico):
    df = cargar_datos(csv_sintetico)
    tr1, te1 = dividir(df)
    tr2, te2 = dividir(df)
    assert np.array_equal(te1, te2)                     # misma semilla -> misma partición
    assert not set(tr1) & set(te1)                      # sin fuga entre train y test
    assert len(tr1) + len(te1) == len(df)
    # Estratificación: el test conserva la proporción de clases del dataset completo
    total = df["clase"].value_counts(normalize=True)
    test = df.loc[te1, "clase"].value_counts(normalize=True)
    assert (total - test).abs().max() < 0.02


def test_calcular_metricas_clasificador_perfecto():
    y = ["negativo", "neutro", "positivo", "positivo"]
    m = calcular_metricas(y, y)
    assert m["reporte"]["accuracy"] == 1.0
    assert m["kappa"] == pytest.approx(1.0)
    assert m["matriz_confusion"] == [[1, 0, 0], [0, 1, 0], [0, 0, 2]]
    assert m["clases"] == CLASES


def test_calcular_metricas_incluye_clase_ausente_sin_romper():
    # Si el test no tiene "neutro", el reporte igual debe traer las 3 clases (la app las lee)
    m = calcular_metricas(["negativo", "positivo"], ["negativo", "positivo"])
    assert set(CLASES) <= set(m["reporte"])
    assert m["reporte"]["neutro"]["f1-score"] == 0


def test_entrenar_guarda_modelo_y_metricas(tmp_path, csv_sintetico):
    ruta_modelo = tmp_path / "out" / "modelo.joblib"
    resultado = train.entrenar(csv_sintetico, ruta_modelo)

    assert ruta_modelo.exists()
    art = joblib.load(ruta_modelo)
    assert set(art) == {"vectorizador", "modelo"}
    assert set(art["modelo"].classes_) == set(CLASES)

    # Las métricas se guardan junto al modelo (no pisan las reales del proyecto)
    metricas = json.loads((tmp_path / "out" / "metricas.json").read_text(encoding="utf-8"))
    assert metricas["clases"] == CLASES
    assert metricas["n_train"] + metricas["n_test"] == 150
    assert 0.0 <= resultado["reporte"]["accuracy"] <= 1.0
    # El dataset sintético es trivialmente separable: el pipeline debe aprenderlo
    assert resultado["reporte"]["accuracy"] > 0.8
