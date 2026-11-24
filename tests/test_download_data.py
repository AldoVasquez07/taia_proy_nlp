import pandas as pd
import pytest

from scripts import download_data as dd


def _bruto(n=10):
    return pd.DataFrame({dd.COL_TEXTO: [f"reseña {i}" for i in range(n)],
                         dd.COL_ESTRELLAS: [1 + i % 5 for i in range(n)],
                         "otra_columna": ["x"] * n})


def test_preparar_renombra_y_descarta_columnas_extra():
    df = dd.preparar(_bruto(), max_filas=100)
    assert list(df.columns) == ["texto", "estrellas"]


def test_preparar_quita_nulos_y_duplicados():
    bruto = _bruto(4)
    bruto.loc[1, dd.COL_TEXTO] = None
    bruto.loc[2, dd.COL_TEXTO] = bruto.loc[3, dd.COL_TEXTO]   # duplicado
    assert len(dd.preparar(bruto, max_filas=100)) == 2


def test_preparar_submuestrea_de_forma_reproducible():
    a = dd.preparar(_bruto(50), max_filas=10)
    b = dd.preparar(_bruto(50), max_filas=10)
    assert len(a) == 10
    assert a.equals(b)


def test_preparar_falla_claro_si_faltan_columnas():
    with pytest.raises(KeyError, match="Faltan columnas"):
        dd.preparar(pd.DataFrame({"a": [1]}), max_filas=10)
