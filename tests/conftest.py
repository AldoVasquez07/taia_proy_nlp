"""Fixtures compartidas: un dataset sintético mínimo para no depender de la descarga real."""
import pandas as pd
import pytest

_PLANTILLAS = {
    1: ["producto malo llegó roto no funciona", "pésima calidad no lo recomiendo nunca",
        "horrible se rompió el primer día no sirve"],
    2: ["mala calidad regular no cumple lo prometido", "decepcionante no es lo que esperaba",
        "tardó mucho y vino con fallas"],
    3: ["está bien cumple pero nada especial", "regular ni bueno ni malo normal",
        "cumple lo básico poco más"],
    4: ["buen producto llegó rápido lo recomiendo", "muy buena calidad cumple bien",
        "me gustó funciona bien buen precio"],
    5: ["excelente producto muy bueno perfecto", "me encantó calidad increíble lo recomiendo",
        "genial llegó rápido muy bien empacado"],
}


@pytest.fixture(scope="session")
def df_sintetico() -> pd.DataFrame:
    """150 reseñas (30 por estrella): sin duplicados exactos gracias al sufijo numérico."""
    filas = []
    for estrellas, frases in _PLANTILLAS.items():
        for i in range(30):
            filas.append({"texto": f"{frases[i % len(frases)]} unidad {i}", "estrellas": estrellas})
    return pd.DataFrame(filas)


@pytest.fixture
def csv_sintetico(tmp_path, df_sintetico):
    ruta = tmp_path / "reviews.csv"
    df_sintetico.to_csv(ruta, index=False)
    return ruta
