"""Pruebas del baseline con Transformer SIN torch ni descargas: el modelo se reemplaza por un
predictor falso. Lo que se prueba es todo lo demás: muestreo, métricas, bootstrap y salida."""
import json

import numpy as np
import pytest

from resenas import train
from resenas.baseline_transformer import (bootstrap_diferencia_f1, ejecutar, mapear_etiqueta,
                                          submuestra_prueba)
from resenas.train import CLASES, cargar_datos, dividir


@pytest.mark.parametrize("etiqueta,esperada", [
    ("NEG", "negativo"), ("NEU", "neutro"), ("POS", "positivo"),
    ("pos", "positivo"), ("Negative", "negativo"), (" NEUTRAL ", "neutro"),
])
def test_mapear_etiqueta(etiqueta, esperada):
    assert mapear_etiqueta(etiqueta) == esperada


def test_mapear_etiqueta_desconocida_falla_con_mensaje_claro():
    with pytest.raises(ValueError, match="LABEL_0"):
        mapear_etiqueta("LABEL_0")


def test_submuestra_es_del_test_estratificada_y_reproducible(csv_sintetico):
    df = cargar_datos(csv_sintetico)
    _, idx_test = dividir(df)
    sub = submuestra_prueba(df, idx_test, 15)
    assert len(sub) == 15 and set(sub) <= set(idx_test)          # nunca sale de test
    assert np.array_equal(sub, submuestra_prueba(df, idx_test, 15))  # reproducible
    assert df.loc[sub, "clase"].nunique() == 3                       # estratificada
    assert np.array_equal(submuestra_prueba(df, idx_test, 10_000), np.sort(idx_test))


def test_bootstrap_sin_diferencia():
    y = ["negativo", "neutro", "positivo"] * 20
    r = bootstrap_diferencia_f1(y, y, y, n_boot=100)
    assert r["diferencia_f1_macro"] == 0 and r["ic95"] == [0, 0] and not r["significativa"]


def test_bootstrap_detecta_una_mejora_clara():
    y = ["negativo", "neutro", "positivo"] * 40
    malo = ["positivo"] * len(y)
    r = bootstrap_diferencia_f1(y, malo, y, n_boot=200)          # B perfecto vs A pésimo
    assert r["diferencia_f1_macro"] > 0.5 and r["significativa"] and r["ic95"][0] > 0


def test_ejecutar_de_extremo_a_extremo_con_predictor_falso(tmp_path, csv_sintetico):
    ruta_modelo, salida = tmp_path / "modelo.joblib", tmp_path / "baseline.json"
    train.entrenar(csv_sintetico, ruta_modelo)
    siempre_neutro = lambda textos: ["neutro"] * len(textos)    # noqa: E731
    siempre_neutro.info = {"dispositivo": "falso"}

    r = ejecutar(n=30, ruta_datos=csv_sintetico, ruta_modelo=ruta_modelo, salida=salida,
                 predictor_transformer=siempre_neutro, n_boot=50)

    assert r["n"] == 30 and r["transformer"]["dispositivo"] == "falso"
    assert r["clasico"]["ms_por_resena"] >= 0
    # el "Transformer" falso solo acierta la clase neutro -> peor que el modelo clásico
    assert r["transformer"]["reporte"]["accuracy"] < r["clasico"]["reporte"]["accuracy"]
    assert r["comparacion"]["diferencia_f1_macro"] < 0
    assert set(r["clasico"]["clases"]) == set(CLASES)
    assert json.loads(salida.read_text(encoding="utf-8"))["n"] == 30   # JSON válido y guardado