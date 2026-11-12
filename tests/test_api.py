import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from resenas import api
from resenas.preprocess import preprocesar_lote

TEXTOS = [
    "excelente producto llegó rápido muy bueno",
    "muy buena calidad lo recomiendo perfecto",
    "me encantó funciona genial",
    "está bien cumple nada especial",
    "regular ni bueno ni malo normal",
    "cumple pero poco más",
    "llegó roto y no funciona",
    "no lo recomiendo pésimo producto",
    "mala calidad no sirve nunca más",
]
CLASES = ["positivo"] * 3 + ["neutro"] * 3 + ["negativo"] * 3


@pytest.fixture
def modelo_falso(tmp_path, monkeypatch):
    """Entrena un modelo mínimo y hace que la API lo use."""
    vec = TfidfVectorizer(ngram_range=(1, 2))
    clf = LogisticRegression(max_iter=1000)
    clf.fit(vec.fit_transform(preprocesar_lote(TEXTOS)), CLASES)

    ruta = tmp_path / "modelo.joblib"
    joblib.dump({"vectorizador": vec, "modelo": clf}, ruta)
    monkeypatch.setattr(api, "MODELO", ruta)
    api._cargar.cache_clear()
    yield ruta
    api._cargar.cache_clear()


@pytest.fixture
def sin_modelo(tmp_path, monkeypatch):
    """Simula que aún no se ha entrenado el modelo."""
    monkeypatch.setattr(api, "MODELO", tmp_path / "no_existe.joblib")
    api._cargar.cache_clear()
    yield
    api._cargar.cache_clear()


@pytest.fixture
def cliente():
    return TestClient(api.app)


def test_salud_con_modelo(cliente, modelo_falso):
    r = cliente.get("/salud")
    assert r.status_code == 200
    assert r.json() == {"estado": "ok", "modelo_entrenado": True}


def test_salud_sin_modelo(cliente, sin_modelo):
    r = cliente.get("/salud")
    assert r.status_code == 200
    assert r.json()["modelo_entrenado"] is False


def test_predecir_devuelve_estructura_esperada(cliente, modelo_falso):
    r = cliente.post("/predecir", json={"texto": "El producto llegó roto y no funciona"})
    assert r.status_code == 200
    datos = r.json()
    assert set(datos) == {"etiqueta", "probabilidades", "terminos_influyentes",
                          "texto_preprocesado"}
    assert datos["etiqueta"] in {"negativo", "neutro", "positivo"}


def test_probabilidades_suman_uno_y_coinciden_con_etiqueta(cliente, modelo_falso):
    datos = cliente.post("/predecir", json={"texto": "excelente producto, muy bueno"}).json()
    probs = datos["probabilidades"]
    assert set(probs) == {"negativo", "neutro", "positivo"}
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-3)
    assert datos["etiqueta"] == max(probs, key=probs.get)


def test_terminos_influyentes_son_validos(cliente, modelo_falso):
    datos = cliente.post("/predecir", json={"texto": "llegó roto y no funciona"}).json()
    terminos = datos["terminos_influyentes"]
    assert 0 < len(terminos) <= 5
    assert all(t["peso"] > 0 for t in terminos)
    pesos = [t["peso"] for t in terminos]
    assert pesos == sorted(pesos, reverse=True)


def test_texto_preprocesado_conserva_negacion(cliente, modelo_falso):
    datos = cliente.post("/predecir", json={"texto": "El producto no funciona"}).json()
    assert "no" in datos["texto_preprocesado"].split()


def test_texto_muy_corto_es_rechazado(cliente, modelo_falso):
    r = cliente.post("/predecir", json={"texto": "ok"})
    assert r.status_code == 422


def test_falta_el_campo_texto(cliente, modelo_falso):
    assert cliente.post("/predecir", json={}).status_code == 422


def test_predecir_sin_modelo_devuelve_503(cliente, sin_modelo):
    r = cliente.post("/predecir", json={"texto": "producto excelente"})
    assert r.status_code == 503
    assert "resenas.train" in r.json()["detail"]