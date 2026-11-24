from resenas.preprocess import (limpiar, marcar_negaciones, normalizar, normalizar_lote,
                                preprocesar, preprocesar_lote)


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


def test_preprocesar_lote_coincide_con_preprocesar():
    textos = ["El producto no funciona", "Excelente calidad, muy rápido", "Llegó tarde y sin caja"]
    esperado = [" ".join(preprocesar(t)) for t in textos]
    assert preprocesar_lote(textos) == esperado


def test_preprocesar_lote_conserva_orden_y_longitud():
    textos = ["uno", "dos tres", "cuatro cinco seis"]
    assert len(preprocesar_lote(textos)) == len(textos)


def test_texto_vacio_o_solo_simbolos_no_rompe():
    assert preprocesar("") == []
    assert preprocesar("!!! ??? 123") == []
    assert preprocesar_lote([""]) == [""]


def test_limpiar_acepta_valores_no_string():
    assert limpiar(12345) == "12345"


def test_negaciones_se_conservan_en_lote():
    assert "no" in preprocesar_lote(["no me gustó nada"])[0].split()


# ------------------------------------------------------- marcado de negaciones
def test_marcar_negaciones_basico():
    assert marcar_negaciones("no funciona") == "no neg_funciona"
    assert marcar_negaciones("funciona") == "funciona"


def test_ambito_de_negacion_limitado_por_ventana():
    assert marcar_negaciones("no es muy bueno de verdad", ventana=3) ==         "no neg_es neg_muy neg_bueno de verdad"


def test_la_negacion_termina_en_puntuacion_y_conjunciones():
    assert marcar_negaciones("no funciona, llegó rápido") == "no neg_funciona llegó rápido"
    assert marcar_negaciones("no es caro pero sirve") == "no neg_es neg_caro pero sirve"


def test_nueva_negacion_reinicia_la_ventana():
    assert marcar_negaciones("no no funciona") == "no no neg_funciona"


def test_marcar_negaciones_pasa_a_minusculas_y_maneja_acentos():
    assert marcar_negaciones("NUNCA más") == "nunca neg_más"
    assert marcar_negaciones("Jamás volvería") == "jamás neg_volvería"


def test_normalizar_limpia_urls_y_marca_negaciones():
    # «nada» es a su vez un negador: no se marca, solo reinicia el ámbito
    assert normalizar("No sirve https://x.com  nada") == "no neg_sirve nada"
    assert normalizar("") == ""
    assert normalizar("!!!") == ""


def test_normalizar_lote_equivale_a_normalizar():
    textos = ["No me gustó", "Excelente producto", ""]
    assert normalizar_lote(textos) == [normalizar(t) for t in textos]


def test_preprocesar_puede_conservar_stopwords():
    con = preprocesar("el producto llegó rápido", quitar_stop=False)
    sin = preprocesar("el producto llegó rápido", quitar_stop=True)
    assert "el" in con and "el" not in sin
