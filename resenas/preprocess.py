import re
import warnings
from functools import lru_cache

# Palabras que NO se eliminan aunque sean stopwords
CONSERVAR = {"no", "nunca", "ni", "sin", "muy", "poco", "mucho", "mas", "más",
             "nada", "tampoco", "jamás", "jamas", "bien", "mal"}

# Lista mínima de respaldo si spaCy no está instalado (solo para pruebas rápidas)
_STOP_BASICAS = {"el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del",
                 "al", "y", "o", "que", "en", "con", "por", "para", "es", "son",
                 "se", "lo", "su", "sus", "me", "te", "mi", "a", "e"}

_RE_URL = re.compile(r"https?://\S+|www\.\S+")
_RE_ESPACIOS = re.compile(r"\s+")


# Marcadores de negación y fin de su ámbito (puntuación o conjunciones adversativas)
_NEGADORES = {"no", "nunca", "jamás", "jamas", "ni", "sin", "tampoco", "nada", "ningún",
              "ninguno", "ninguna"}
_FIN_AMBITO = {"pero", "aunque", "sino", "y", "que"}
_RE_TOKEN = re.compile(r"[\wáéíóúüñÁÉÍÓÚÜÑ]+|[.,;:!?¡¿]", re.UNICODE)


def marcar_negaciones(texto: str, ventana: int = 3) -> str:
    """Prefija con `NEG_` las hasta `ventana` palabras que siguen a una negación, hasta
    la siguiente puntuación o conjunción adversativa (Das & Chen, 2001). Así «no funciona»
    y «funciona» generan rasgos distintos (`neg_funciona` vs `funciona`).
    Devuelve el texto en minúsculas con tokens separados por espacio."""
    salida, restantes = [], 0
    for tok in _RE_TOKEN.findall(texto.lower()):
        if not tok[0].isalnum() and tok[0] != "_":      # puntuación: cierra el ámbito
            restantes = 0
            continue
        if tok in _NEGADORES:
            salida.append(tok)
            restantes = ventana
        elif tok in _FIN_AMBITO:
            salida.append(tok)
            restantes = 0
        elif restantes > 0:
            salida.append("neg_" + tok)
            restantes -= 1
        else:
            salida.append(tok)
    return " ".join(salida)


def limpiar(texto: str) -> str:
    """Normalización mínima: URLs fuera y espacios colapsados.
    NO pasa a minúsculas aquí: se hace después de lematizar (ver Laboratorio 1)."""
    texto = _RE_URL.sub(" ", str(texto))
    return _RE_ESPACIOS.sub(" ", texto).strip()


def normalizar(texto: str) -> str:
    """Preprocesamiento del MODELO FINAL: limpieza + minúsculas + marcado de negaciones.
    No usa spaCy: en la ablación (ver evaluar.py) la lematización y el filtrado de
    stopwords empeoraron el F1, así que se conserva el texto casi tal cual."""
    return marcar_negaciones(limpiar(texto))


def normalizar_lote(textos) -> list[str]:
    return [normalizar(t) for t in textos]


@lru_cache(maxsize=1)
def _cargar_nlp():
    try:
        import spacy
        return spacy.load("es_core_news_sm", disable=["parser", "ner"])
    except Exception:
        warnings.warn("spaCy/es_core_news_sm no disponible: uso tokenizador básico "
                      "SIN lematización. Instala el modelo para resultados reales.")
        return None


def _basico(texto: str, quitar_stop: bool = True) -> list[str]:
    tokens = re.findall(r"[a-záéíóúüñ]+", texto.lower())
    if not quitar_stop:
        return tokens
    return [t for t in tokens if t not in _STOP_BASICAS or t in CONSERVAR]


def preprocesar(texto: str, quitar_stop: bool = True) -> list[str]:
    """Texto crudo -> lista de lemas en minúscula.
    Con `quitar_stop=True` (por defecto) elimina stopwords salvo negaciones/cuantificadores."""
    texto = limpiar(texto)
    nlp = _cargar_nlp()
    if nlp is None:
        return _basico(texto, quitar_stop)
    return _lemas(nlp(texto), quitar_stop)


def _lemas(doc, quitar_stop: bool = True) -> list[str]:
    salida = []
    for t in doc:
        if not t.is_alpha:
            continue
        lema = t.lemma_.lower()
        if (quitar_stop and t.is_stop and lema not in CONSERVAR
                and t.text.lower() not in CONSERVAR):
            continue
        salida.append(lema)
    return salida


def preprocesar_lote(textos, n_process: int = 1, quitar_stop: bool = True) -> list[str]:
    """Versión rápida para entrenamiento: devuelve cada texto como string de lemas."""
    textos = [limpiar(t) for t in textos]
    nlp = _cargar_nlp()
    if nlp is None:
        return [" ".join(_basico(t, quitar_stop)) for t in textos]
    return [" ".join(_lemas(d, quitar_stop))
            for d in nlp.pipe(textos, batch_size=256, n_process=n_process)]
