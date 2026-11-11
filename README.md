# Clasificador de reseñas en español

Proyecto del curso de NLP. Clasifica reseñas de productos en **negativo / neutro / positivo**
usando preprocesamiento en español (spaCy) + **TF-IDF** + **Regresión Logística**. El modelo se
expone como una **API REST (FastAPI)** y como una **interfaz web (Streamlit)**; ambas devuelven
además los términos que más influyeron en la predicción.

**Autor:** _Aldo Ray Vasquez Lopez_ · **Estado:** Evidencia 3 (demo al 50 %)

## Demo

| Componente | Comando | URL |
|---|---|---|
| Interfaz web (Streamlit) | `streamlit run streamlit_app.py` | http://localhost:8501 |
| API REST (FastAPI) | `uvicorn resenas.api:app --reload` | http://127.0.0.1:8000/docs |

La interfaz tiene dos pestañas:
- **Predecir:** ejemplos rápidos o texto libre; muestra sentimiento, confianza, probabilidades por clase, términos influyentes y el texto preprocesado.
- **Métricas del modelo:** accuracy, F1 macro, métricas por clase y matriz de confusión (leídas de `models/metricas.json`).

## Datos

Dataset **MeLiSA** (reseñas de Mercado Libre en español), descargado con script:
https://huggingface.co/datasets/lpsc-fiuba/melisa

- Licencia: acuerdo de Amazon (https://docs.opendata.aws/amazon-reviews-ml/license.txt).
- Uso exclusivamente académico. Los datos **no** se suben al repositorio.
- La rama `main` del dataset solo contiene una versión demo; por eso el script descarga una **revisión fija** (commit) con los datos completos (`es/train.csv`).
- Por defecto se usa una muestra de 60 000 reseñas (`--max-filas` para cambiarla), sin duplicados ni valores nulos.

Mapeo de etiquetas: 1–2 estrellas → negativo · 3 → neutro · 4–5 → positivo.

## Instalación

Requiere Python 3.10+.

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` ya incluye el modelo de spaCy `es_core_news_sm`. Si su instalación falla en tu sistema, instálalo manualmente:

```bash
python -m spacy download es_core_news_sm
```

> Si el modelo de spaCy no está disponible, el preprocesamiento usa un tokenizador básico **sin lematización** (y emite una advertencia). Sirve para pruebas rápidas, pero no para obtener resultados reales.

## Uso

Ejecutar en este orden desde la raíz del proyecto:

```bash
python scripts/download_data.py        # descarga y guarda data/raw/reviews.csv
python -m resenas.train                # entrena y guarda models/modelo.joblib y models/metricas.json
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

Endpoints: `GET /salud` (estado y si el modelo está entrenado) · `POST /predecir`.
Si el modelo no existe, `/predecir` responde `503` con instrucciones.

## Estructura

```
scripts/download_data.py   obtención del dataset (MeLiSA)
resenas/preprocess.py      normalización, tokenización, lematización, stopwords
resenas/train.py           TF-IDF + regresión logística, métricas
resenas/api.py             endpoints /salud y /predecir
streamlit_app.py           interfaz web (predicción + métricas)
tests/                     pruebas automatizadas (pytest)
data/raw/                  datos descargados (ignorados por git)
models/                    modelo y métricas entrenados (ignorados por git)
```

## Técnicas de NLP utilizadas

- **Preprocesamiento en español con spaCy:** limpieza (URLs, espacios), tokenización, lematización y filtrado de stopwords.
- **Representación TF-IDF** con unigramas y bigramas.
- **Clasificación** con regresión logística (`class_weight="balanced"`), con explicabilidad a partir de los coeficientes (peso TF-IDF × coeficiente).

## Decisiones de diseño

- **Se conservan las negaciones y cuantificadores** («no», «nunca», «sin», «muy», «poco»…) aunque sean stopwords: cambian el sentimiento.
- **Se lematiza antes de pasar a minúsculas**, porque el modelo pequeño de spaCy lematiza peor con minúsculas.
- **Bigramas (1–2) en TF-IDF** para capturar casos como «no funciona».
- **`class_weight="balanced"`** para compensar el desbalance entre clases (la clase neutra suele ser minoritaria).
- Partición 80/20 estratificada con semilla fija (`random_state=42`) para reproducibilidad.

## Limitaciones conocidas

- El mapeo por estrellas es ruidoso: una reseña de 3 estrellas no siempre es realmente neutra, por lo que la clase **neutro** es la más difícil de predecir.
- Modelo clásico (bolsa de palabras): no captura bien ironía, sarcasmo ni contexto largo.
- Entrenado solo con reseñas de Mercado Libre; puede generalizar peor a otros dominios.
- La descarga depende de una revisión fija del dataset en Hugging Face.

## Estado y pendientes

- [x] Repositorio, script de datos, preprocesamiento, línea base, endpoint `/predecir`
- [x] Interfaz web mínima (Streamlit) con pestaña de métricas
- [x] Pruebas del preprocesamiento
- [ ] Comparar preprocesamiento (stopwords / lema vs. stem) y BoW vs. TF-IDF
- [ ] Evaluación completa y análisis de errores
- [ ] Pruebas de la API y del entrenamiento
- [ ] Informe (máx. 4 páginas)