# Clasificador de reseñas en español

Proyecto del curso de NLP. Clasifica reseñas de productos en **negativo / neutro / positivo**
usando preprocesamiento en español (spaCy) + **TF-IDF** + **Regresión Logística**. El modelo se
expone como una **API REST (FastAPI)** y como una **interfaz web (Streamlit)**; ambas devuelven
además los términos que más influyeron en la predicción.

**Autores:** _Aldo Ray Vasquez Lopez_ y _Alvaro Andre Machaca Melendez_ · **Estado:** Evidencia 4 (demo al 100 %)

## Demo

| Componente | Comando | URL |
|---|---|---|
| Interfaz web (Streamlit) | `streamlit run streamlit_app.py` | http://localhost:8501 |
| API REST (FastAPI) | `uvicorn resenas.api:app --reload` | http://127.0.0.1:8000/docs |

La interfaz tiene cuatro pestañas:
- **Predecir:** ejemplos rápidos o texto libre; muestra sentimiento, confianza, probabilidades por clase, términos influyentes y el texto preprocesado.
- **Lote (CSV):** sube un CSV de reseñas, clasifícalas todas, mira la distribución de sentimientos y descarga los resultados.
- **Métricas del modelo:** accuracy, F1 macro, kappa, métricas por clase y matriz de confusión (`models/metricas.json`).
- **Comparación y errores:** ablación de preprocesamiento/representación/clasificador, validación cruzada y los fallos más «seguros» del modelo (`models/comparacion.json`, `models/errores.csv`).

## Datos

Dataset **MeLiSA** (reseñas de Mercado Libre en español), descargado con script:
https://huggingface.co/datasets/lpsc-fiuba/melisa

- Licencia: acuerdo de Amazon (https://docs.opendata.aws/amazon-reviews-ml/license.txt).
- Uso exclusivamente académico. Los datos **no** se suben al repositorio.
- La rama `main` del dataset solo contiene una versión demo; por eso el script descarga una **revisión fija** (commit) con los datos completos (`es/train.csv`).
- Por defecto se usa una muestra de 60 000 reseñas (`--max-filas` para cambiarla), sin duplicados ni valores nulos.

Mapeo de etiquetas: 1–2 estrellas → negativo · 3 → neutro · 4–5 → positivo.

## Instalación
Requiere **Python 3.12 o superior** (las versiones fijadas de NumPy y SciPy no soportan versiones anteriores). Verifica la tuya con `python --version`.

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` ya incluye el modelo de spaCy `es_core_news_sm`. Si esa instalación falla en tu sistema, instala el modelo manualmente:

```bash
python -m spacy download es_core_news_sm
```

> Si el modelo de spaCy no está disponible, el preprocesamiento usa un tokenizador básico **sin lematización** (y emite una advertencia). Sirve para pruebas rápidas, pero no para obtener resultados reales.

## Uso

Ejecutar en este orden desde la raíz del proyecto:

```bash
python scripts/download_data.py        # descarga y guarda data/raw/reviews.csv
python -m resenas.train                # entrena y guarda models/modelo.joblib y models/metricas.json
python -m resenas.evaluar              # ablación + validación cruzada + análisis de errores
streamlit run streamlit_app.py         # interfaz web
uvicorn resenas.api:app --reload       # API (opcional, en otra terminal)
pytest                                 # pruebas automatizadas
```

El entrenamiento lematiza todas las reseñas con spaCy, así que puede tardar varios minutos.

Ejemplo de llamada a la API:

```bash
curl -X POST http://127.0.0.1:8000/predecir -H "Content-Type: application/json" \
     -d '{"texto": "El producto llegó roto y no funciona"}'
```

Respuesta (forma):

```json
{
  "etiqueta": "negativo",
  "probabilidades": {"negativo": 0.0, "neutro": 0.0, "positivo": 0.0},
  "terminos_influyentes": [{"termino": "no funciona", "peso": 0.0}],
  "texto_preprocesado": "producto llegar roto no funcionar"
}
```

Endpoints:

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/salud` | estado y si el modelo está entrenado |
| POST | `/predecir` | una reseña → etiqueta, probabilidades, términos influyentes |
| POST | `/predecir_lote` | hasta 1000 reseñas (`{"textos": [...]}`) → etiqueta y confianza de cada una |
| GET | `/metricas` | métricas del modelo (`404` si aún no se entrenó) |

Si el modelo no existe, `/predecir` y `/predecir_lote` responden `503` con instrucciones; textos demasiado cortos o listas vacías responden `422`.

## Estructura

```
scripts/download_data.py   obtención del dataset (MeLiSA)
resenas/preprocess.py      normalización, tokenización, lematización, stopwords
resenas/train.py           TF-IDF + regresión logística, métricas (accuracy, F1, kappa, MCC)
resenas/evaluar.py         ablación, validación cruzada y análisis de errores
resenas/api.py             endpoints /salud, /predecir, /predecir_lote y /metricas
streamlit_app.py           dashboard web (predicción, lote CSV, métricas, comparación)
docs/informe.md            informe de la Evidencia 4 (máx. 4 páginas)
tests/                     pruebas automatizadas (pytest)
data/raw/                  datos descargados (ignorados por git)
models/                    modelo y métricas entrenados (ignorados por git)
```

## Técnicas de NLP utilizadas

**En el modelo final:**
- **Normalización y tokenización** del texto (URLs fuera, minúsculas, espacios).
- **Marcado de ámbito de negación** (Das & Chen, 2001): las 3 palabras tras «no», «nunca», «sin»… se marcan con `neg_` (`no funciona` → `no neg_funciona`) hasta la siguiente puntuación o «pero/aunque/y/que».
- **Representación TF-IDF** con unigramas y bigramas (`sublinear_tf`).
- **Clasificación** con regresión logística (`class_weight="balanced"`), con explicabilidad a partir de los coeficientes (peso TF-IDF × coeficiente).

**Evaluadas en la ablación (`resenas/evaluar.py`):** lematización y filtrado de stopwords con spaCy, Bag of Words, unigramas vs. bigramas, Naive Bayes y SVM lineal.

## Decisiones de diseño
- **El preprocesamiento final NO lematiza ni quita stopwords.** La Evidencia 3 usaba spaCy (lemas, sin stopwords). La ablación separó los dos efectos: **quitar stopwords empeoró el F1 macro en 3.3 puntos** (0.683 → 0.716), probablemente porque se pierden conectores con información («pero», «para»); en cambio, **lematizar no cambió el resultado** (0.716 con lemas frente a 0.717 con texto normalizado). Como no aporta mejora, se descartó por simplicidad y velocidad: la API no depende de spaCy y predice más rápido. spaCy quedó como línea base de la ablación.
- **Marcado de negaciones** en lugar de depender solo de bigramas: captura negaciones a distancia («no es nada bueno»).
- **Bigramas (1–2) en TF-IDF** y `sublinear_tf`; un único TF-IDF para todo el texto.
- **`class_weight="balanced"`** para compensar que «neutro» (solo 3 estrellas) tiene la mitad de ejemplos que las otras clases.
- Partición 80/20 estratificada con semilla fija (`random_state=42`); todas las configuraciones de la ablación comparten el mismo conjunto de prueba, y la configuración final se valida además con K-Fold (k=5).
- **Las pruebas no dependen de la descarga real**: usan un dataset sintético mínimo (`tests/conftest.py`).

## Limitaciones conocidas

- **Ruido de etiquetas:** el mapeo por estrellas es ruidoso. Los errores más «seguros» del modelo suelen ser reseñas como «Todo excelente» con 2 estrellas: el modelo acierta el sentimiento del texto pero no la etiqueta. La clase **neutro** (3 estrellas) es la más difícil: concentra la gran mayoría de los errores.
- Modelo de bolsa de palabras: no captura bien ironía, sarcasmo ni contexto largo; una reseña con opiniones mezcladas se resuelve por las palabras dominantes.
- Entrenado solo con reseñas de Mercado Libre (varias categorías de producto); puede generalizar peor a otros dominios.
- La descarga depende de una revisión fija del dataset en Hugging Face; se usa una muestra de 60 000 reseñas.
- No se probó un modelo neuronal (p. ej., transformers) por costo de cómputo; sería la siguiente mejora natural.

## Pruebas automatizadas

```bash
pytest          # ~60 pruebas, unos segundos, no necesitan datos reales ni el modelo entrenado
```

Cubren: preprocesamiento y marcado de negaciones, limpieza del script de datos, particiones y métricas, entrenamiento end-to-end sobre datos sintéticos, la ablación/validación cruzada/análisis de errores, y la API (estructura de respuesta, probabilidades, lote, validaciones `422`, `503` sin modelo, `/metricas`).

## Estado

- [x] Repositorio, script de datos, línea base, endpoint `/predecir`, interfaz web (Evidencia 3)
- [x] Comparación de preprocesamiento (lema vs. normalizado vs. negaciones) y BoW vs. TF-IDF
- [x] Evaluación completa: hold-out, validación cruzada y análisis de errores
- [x] Endpoints `/predecir_lote` y `/metricas`; dashboard con carga de CSV y comparación de modelos
- [x] Pruebas automatizadas de preprocesamiento, entrenamiento, evaluación y API
- [x] Informe (máx. 4 páginas): `docs/informe.md`
