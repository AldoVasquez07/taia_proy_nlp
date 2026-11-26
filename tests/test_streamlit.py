"""Pruebas de la interfaz Streamlit con AppTest (sin navegador ni servidor).

AppTest ejecuta `streamlit_app.py` en el mismo proceso, así que usamos los mismos trucos que
en test_api.py: un modelo mínimo y `api.MODELO` apuntando a él.
"""
from pathlib import Path

import joblib
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from streamlit.testing.v1 import AppTest

from resenas import api
from resenas.preprocess import normalizar_lote

APP = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")  # ruta absoluta

TEXTOS = ["excelente producto muy bueno", "muy buena calidad lo recomiendo",
          "me encantó funciona genial", "está bien cumple nada especial",
          "regular ni bueno ni malo normal", "cumple pero poco más",
          "llegó roto y no funciona", "no lo recomiendo pésimo producto",
          "mala calidad no sirve nunca más"]
CLASES = ["positivo"] * 3 + ["neutro"] * 3 + ["negativo"] * 3


@pytest.fixture
def con_modelo(tmp_path, monkeypatch):
    vec = TfidfVectorizer(ngram_range=(1, 2))
    clf = LogisticRegression(max_iter=1000)
    clf.fit(vec.fit_transform(normalizar_lote(TEXTOS)), CLASES)
    ruta = tmp_path / "modelo.joblib"
    joblib.dump({"vectorizador": vec, "modelo": clf}, ruta)
    monkeypatch.setattr(api, "MODELO", ruta)
    api._cargar.cache_clear()
    yield
    api._cargar.cache_clear()


@pytest.fixture
def sin_modelo(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "MODELO", tmp_path / "no_existe.joblib")
    api._cargar.cache_clear()
    yield
    api._cargar.cache_clear()


def _app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=30).run()


def _boton(at: AppTest, etiqueta: str):
    return next(b for b in at.button if b.label == etiqueta)


def _metricas_de_prediccion(at: AppTest) -> dict:
    """Solo las métricas de la pestaña Predecir. La pestaña «Métricas del modelo» también dibuja
    métricas (Accuracy, F1, Kappa) cuando existe models/metricas.json, y eso depende de la
    máquina: las pruebas no deben asumir que `at.metric` contiene únicamente las de Predecir."""
    return {m.label: m.value for m in at.metric if m.label in ("Sentimiento", "Confianza")}


def _clasificar(at: AppTest, texto: str) -> AppTest:
    at.text_area(key="texto").set_value(texto).run()
    return _boton(at, "Clasificar").click().run()


# ------------------------------------------------------------------ carga
def test_la_app_carga_sin_errores_y_con_sus_cuatro_pestanas():
    # Con o sin modelo/metricas en models/, ninguna pestaña debe romperse
    at = _app()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Predecir", "Lote (CSV)", "Métricas del modelo",
                                          "Comparación y errores"]


# ------------------------------------------------------------------ pestaña Predecir
def test_ejemplo_rapido_rellena_el_texto():
    at = _app()
    at.button(key="ej0").click().run()
    assert at.text_area(key="texto").value == "El producto llegó roto y no funciona"


def test_clasificar_muestra_sentimiento_y_confianza(con_modelo):
    at = _clasificar(_app(), "excelente producto, muy bueno")
    assert not at.exception
    metricas = _metricas_de_prediccion(at)
    assert set(metricas) == {"Sentimiento", "Confianza"}
    assert any(c in metricas["Sentimiento"] for c in ("negativo", "neutro", "positivo"))
    assert metricas["Confianza"].endswith("%")


def test_texto_demasiado_corto_muestra_advertencia(con_modelo):
    at = _clasificar(_app(), "ok")
    assert at.warning and not _metricas_de_prediccion(at)


def test_texto_sin_contenido_lexico_muestra_error(con_modelo):
    at = _clasificar(_app(), "???")
    assert any("letra" in e.value for e in at.error)
    assert not _metricas_de_prediccion(at)


def test_sin_modelo_muestra_instrucciones(sin_modelo):
    at = _clasificar(_app(), "producto excelente")
    assert any("resenas.train" in e.value for e in at.error)


# ------------------------------------------------------------------ pestaña Lote (CSV)
def test_lote_descarta_filas_invalidas_y_clasifica_el_resto(con_modelo):
    csv = ("texto\nexcelente producto muy bueno\n???\nllegó roto y no funciona\n   \n"
           "no lo recomiendo pésimo\n").encode("utf-8")
    at = _app()
    at.get("file_uploader")[0].upload("resenas.csv", csv, "text/csv").run()
    _boton(at, "Clasificar lote").click().run()

    assert not at.exception and not at.error
    lote = at.session_state["lote"]
    assert len(lote) == 3                                  # 5 filas, 2 inválidas
    assert list(lote.columns) == ["texto", "etiqueta", "confianza"]
    assert set(lote["etiqueta"]) <= {"negativo", "neutro", "positivo"}


def test_lote_sin_textos_validos_avisa(con_modelo):
    at = _app()
    at.get("file_uploader")[0].upload("resenas.csv", b"texto\n???\n   \n", "text/csv").run()
    _boton(at, "Clasificar lote").click().run()
    assert at.warning and "lote" not in at.session_state