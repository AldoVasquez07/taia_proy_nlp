import json

import pandas as pd
import pytest

from resenas import evaluar
from resenas.train import CLASES, cargar_datos, dividir


@pytest.fixture(scope="module")
def entorno(tmp_path_factory, df_sintetico):
    ruta = tmp_path_factory.mktemp("datos") / "reviews.csv"
    df_sintetico.to_csv(ruta, index=False)
    df = cargar_datos(ruta)
    i_tr, i_te = dividir(df)
    return df, evaluar.preparar_textos(df), i_tr, i_te


def test_preparar_textos_devuelve_todas_las_variantes(entorno):
    df, textos, _, _ = entorno
    assert set(textos) == {"normalizado", "negaciones", "lemas", "lemas_con_stop",
                           "neg_emojis", "neg_alarg", "neg_tildes", "neg_todo"}
    assert all(len(s) == len(df) for s in textos.values())
    assert all(t == t.lower() for t in textos["normalizado"])
    # Las stopwords solo se quitan en "lemas"
    n = lambda k: textos[k].str.split().str.len().sum()
    assert n("lemas") < n("lemas_con_stop")
    # Solo "negaciones" agrega marcadores NEG_
    assert textos["negaciones"].str.contains("neg_").any()
    assert not textos["normalizado"].str.contains("neg_").any()


def test_configuracion_final_existe_y_usa_negaciones():
    assert evaluar.FINAL in evaluar.CONFIGURACIONES
    assert evaluar.CONFIGURACIONES[evaluar.FINAL][0] == "negaciones"


def test_comparar_evalua_todas_las_configuraciones(entorno):
    df, textos, i_tr, i_te = entorno
    filas = evaluar.comparar(df, textos, i_tr, i_te)
    assert [f["configuracion"] for f in filas] == list(evaluar.CONFIGURACIONES)
    for f in filas:
        assert 0 <= f["accuracy"] <= 1
        assert 0 <= f["f1_macro"] <= 1
        assert set(f["f1_por_clase"]) == set(CLASES)


def test_validacion_cruzada_devuelve_media_y_desviacion(entorno):
    df, textos, _, _ = entorno
    cv = evaluar.validacion_cruzada(df, textos, k=3)
    assert cv["k"] == 3
    assert 0 <= cv["f1_macro_media"] <= 1
    assert cv["f1_macro_std"] >= 0


def test_analisis_errores_es_consistente(entorno):
    df, textos, i_tr, i_te = entorno
    resumen, peores = evaluar.analisis_errores(df, textos, i_tr, i_te, top=5)
    assert set(resumen["por_estrellas"]) <= {1, 2, 3, 4, 5}
    assert resumen["n_errores_graves"] <= resumen["n_errores"]
    assert 0 <= resumen["pct_errores_que_involucran_neutro"] <= 1
    assert len(peores) <= 5
    assert (peores["clase"] != peores["prediccion"]).all()          # solo errores
    assert peores["confianza"].is_monotonic_decreasing              # los más seguros primero


def test_main_escribe_los_archivos_de_salida(tmp_path, monkeypatch, csv_sintetico):
    monkeypatch.setattr(evaluar, "SALIDA_JSON", tmp_path / "comparacion.json")
    monkeypatch.setattr(evaluar, "SALIDA_ERRORES", tmp_path / "errores.csv")

    salida = evaluar.main(k_cv=3, ruta_datos=csv_sintetico)

    guardado = json.loads((tmp_path / "comparacion.json").read_text(encoding="utf-8"))
    assert guardado["n_total"] == 150 == salida["n_total"]
    assert len(guardado["comparacion"]) == len(evaluar.CONFIGURACIONES)
    assert guardado["validacion_cruzada"]["k"] == 3
    assert (tmp_path / "errores.csv").exists()
    pd.read_csv(tmp_path / "errores.csv")  # es un CSV válido (aunque sea vacío de filas)