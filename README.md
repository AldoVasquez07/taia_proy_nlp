# Clasificador de reseñas en español

Proyecto del curso de NLP. Clasifica reseñas de productos en **negativo / neutro / positivo**
usando preprocesamiento en español (spaCy) + **TF-IDF** + **Regresión Logística**, y lo expone
como una **API REST (FastAPI)** que además devuelve los términos que más influyeron.

**Autor:** _Aldo Ray Vasquez Lopez_ · **Estado:** Evidencia 3 (demo al 50 %)

## Datos
Dataset **MeLiSA** (reseñas de Mercado Libre en español), descargado con script:
https://huggingface.co/datasets/lpsc-fiuba/melisa
Licencia: acuerdo de Amazon (https://docs.opendata.aws/amazon-reviews-ml/license.txt).
Uso exclusivamente académico. Los datos **no** se suben al repositorio.

Mapeo de etiquetas: 1–2 estrellas → negativo · 3 → neutro · 4–5 → positivo.

## Instalación
```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download es_core_news_sm
```

## Uso
```bash
python scripts/download_data.py        # descarga y guarda data/raw/reviews.csv
python -m resenas.train                # entrena y guarda models/modelo.joblib
uvicorn resenas.api:app --reload       # API en http://127.0.0.1:8000/docs
pytest                                 # pruebas
```

Ejemplo:
```bash
curl -X POST http://127.0.0.1:8000/predecir -H "Content-Type: application/json" \
     -d '{"texto": "El producto llegó roto y no funciona"}'
```

## Estructura
```
scripts/download_data.py   obtención del dataset
resenas/preprocess.py      normalización, tokenización, lematización, stopwords
resenas/train.py           TF-IDF + regresión logística, métricas
resenas/api.py             endpoints /salud y /predecir
tests/                     pruebas automatizadas
```

## Decisiones de diseño
- **Se conservan las negaciones** («no», «nunca», «sin»…) aunque sean stopwords: cambian el sentimiento.
- **Se lematiza antes de pasar a minúsculas**, porque el modelo pequeño de spaCy lematiza peor con minúsculas.
- Bigramas (1–2) en TF-IDF para capturar casos como «no funciona».

## Estado y pendientes
- [x] Repositorio, script de datos, preprocesamiento, línea base, endpoint `/predecir`
- [ ] Interfaz web mínima (Streamlit)
- [ ] Comparar preprocesamiento (stopwords / lema vs. stem) y BoW vs. TF-IDF
- [ ] Evaluación completa y análisis de errores · informe · pruebas ampliadas
