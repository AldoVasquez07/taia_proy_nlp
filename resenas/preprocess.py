"""Preprocesamiento de texto en español (Laboratorio 1, con decisiones propias).

Decisión clave: se CONSERVAN las negaciones y cuantificadores («no», «nunca»,
«sin», «muy», «poco»...) aunque spaCy los marque como stopwords, porque cambian
el sentimiento (ver diapositiva 11).
"""
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


def limpiar(texto: str) -> str:
    """Normalización mínima: URLs fuera y espacios colapsados.
    NO pasa a minúsculas aquí: se hace después de lematizar (ver Laboratorio 1)."""
    texto = _RE_URL.sub(" ", str(texto))
    return _RE_ESPACIOS.sub(" ", texto).strip()


@lru_cache(maxsize=1)
def _cargar_nlp():
    try:
        import spacy
        return spacy.load("es_core_news_sm", disable=["parser", "ner"])
    except Exception:
        warnings.warn("spaCy/es_core_news_sm no disponible: uso tokenizador básico "
                      "SIN lematización. Instala el modelo para resultados reales.")
        return None


def _basico(texto: str) -> list[str]:
    tokens = re.findall(r"[a-záéíóúüñ]+", texto.lower())
    return [t for t in tokens if t not in _STOP_BASICAS or t in CONSERVAR]


def preprocesar(texto: str) -> list[str]:
    """Texto crudo -> lista de lemas en minúscula, sin stopwords (salvo negaciones)."""
    texto = limpiar(texto)
    nlp = _cargar_nlp()
    if nlp is None:
        return _basico(texto)
    return _lemas(nlp(texto))


def _lemas(doc) -> list[str]:
    salida = []
    for t in doc:
        if not t.is_alpha:
            continue
        lema = t.lemma_.lower()
        if t.is_stop and lema not in CONSERVAR and t.text.lower() not in CONSERVAR:
            continue
        salida.append(lema)
    return salida


def preprocesar_lote(textos, n_process: int = 1) -> list[str]:
    """Versión rápida para entrenamiento: devuelve cada texto como string de lemas."""
    textos = [limpiar(t) for t in textos]
    nlp = _cargar_nlp()
    if nlp is None:
        return [" ".join(_basico(t)) for t in textos]
    return [" ".join(_lemas(d)) for d in nlp.pipe(textos, batch_size=256, n_process=n_process)]
