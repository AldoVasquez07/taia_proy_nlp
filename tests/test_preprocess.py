from resenas.preprocess import limpiar, preprocesar


def test_limpiar_quita_urls_y_espacios():
    assert limpiar("mira   https://ejemplo.com  esto") == "mira esto"


def test_preprocesar_devuelve_lista_de_minusculas():
    tokens = preprocesar("¡Excelente PRODUCTO!")
    assert tokens and all(t == t.lower() for t in tokens)


def test_conserva_negaciones():
    # Decisión de diseño: «no» cambia el sentimiento, no debe eliminarse
    assert "no" in preprocesar("El producto no funciona")


def test_quita_articulos():
    assert "el" not in preprocesar("el producto llegó rápido")
