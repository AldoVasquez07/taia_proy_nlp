# Clasificador de reseñas en español

![tests](https://github.com/AldoVasquez07/taia_proy_nlp/actions/workflows/tests.yml/badge.svg)

Proyecto del curso de NLP. Clasifica reseñas de productos en **negativo / neutro / positivo**
con normalización de texto en español, **marcado de negaciones**, **TF-IDF** (unigramas y bigramas)
y **regresión logística**. El modelo se expone como una **API REST (FastAPI)** y como una
**interfaz web (Streamlit)**; ambas devuelven además los términos que más influyeron en la predicción.

**Autores:** Aldo Ray Vasquez Lopez y Alvaro Andre Machaca Melendez · **Estado:** Evidencia 4 (demo al 100 %)

## Demo

| Componente | Comando | URL |
|---|---|---|
| Interfaz web (Streamlit) | `streamlit run streamlit_app.py` | http://localhost:8501 |
| API REST (FastAPI) | `uvicorn resenas.api:app --reload` | http://127.0.0.1:8000/docs |

La interfaz tiene cuatro pestañas:
- **Predecir:** ejemplos rápidos o texto libre; muestra sentimiento, confianza, probabilidades por clase, términos influyentes y el texto preprocesado.
- **Lote (CSV):** sube un CSV de reseñas, clasifícalas todas, mira la distribución de sentimientos y descarga los resultados. Las filas inválidas (vacías, solo signos, demasiado largas) se descartan.
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

Si tienes varias versiones de Python instaladas, crea el entorno con la 3.12 explícitamente:

```bash
python3.12 -m venv .venv        # Windows: py -3.12 -m venv .venv
```

`requirements.txt` ya incluye el modelo de spaCy `es_core_news_sm`. Si esa instalación falla en tu sistema, instala el modelo manualmente:

```bash
python -m spacy download es_core_news_sm
```

> spaCy solo se usa en la ablación (`resenas.evaluar`), no en el modelo final. Si el modelo de spaCy no está disponible, el preprocesamiento de la ablación usa un tokenizador básico **sin lematización** (y emite una advertencia); sirve para pruebas rápidas, pero no para obtener resultados reales.

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

El entrenamiento solo normaliza el texto (sin spaCy). `resenas.evaluar` sí lematiza con spaCy para la línea base de la ablación, por lo que tarda varios minutos.

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
  "texto_preprocesado": "el producto llegó roto y no neg_funciona"
}
```

Endpoints:

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/salud` | estado y si el modelo está entrenado |
| POST | `/predecir` | una reseña → etiqueta, probabilidades, términos influyentes |
| POST | `/predecir_lote` | hasta 1000 reseñas (`{"textos": [...]}`) → etiqueta y confianza de cada una |
| GET | `/metricas` | métricas del modelo (`404` si aún no se entrenó) |

Si el modelo no existe, `/predecir` y `/predecir_lote` responden `503` con instrucciones. Responden `422` los textos inválidos (menos de 3 o más de 5000 caracteres, o sin ninguna letra ni número, p. ej. `"???"`) y las listas vacías o de más de 1000 textos.

## Resultados

Partición 80/20 estratificada (48 000 reseñas de entrenamiento, 12 000 de prueba). Cifras de `python -m resenas.train` y `python -m resenas.evaluar`.

**Modelo final** (normalización + negaciones + TF-IDF 1–2 + regresión logística): exactitud **0,761**, F1 macro **0,725**. Validación cruzada de 5 particiones: exactitud 0,759 ± 0,002, F1 macro 0,721 ± 0,003.

| Clase | Precisión | Recall | F1 |
|---|---|---|---|
| negativo | 0,88 | 0,79 | 0,83 |
| neutro | 0,48 | 0,57 | 0,52 |
| positivo | 0,82 | 0,83 | 0,83 |

**Ablación** (mismo conjunto de prueba para todas las configuraciones):

| Configuración | Exactitud | F1 macro |
|---|---|---|
| Lemas sin stopwords + TF-IDF(1–2) + LogReg *(base de la Evidencia 3)* | 0,723 | 0,683 |
| Lemas con stopwords + TF-IDF(1–2) + LogReg | 0,752 | 0,716 |
| Normalizado + BoW(1–2) + LogReg | 0,751 | 0,700 |
| Normalizado + TF-IDF(1) + LogReg | 0,727 | 0,694 |
| Normalizado + TF-IDF(1–2) + LogReg | 0,754 | 0,717 |
| Normalizado + negaciones + TF-IDF(1) + LogReg | 0,742 | 0,709 |
| **Normalizado + negaciones + TF-IDF(1–2) + LogReg (final)** | **0,761** | **0,725** |
| Normalizado + negaciones + TF-IDF(1–2) + ComplementNB | 0,724 | 0,567 |
| Normalizado + negaciones + TF-IDF(1–2) + LinearSVC | 0,755 | 0,696 |

Lectura: quitar stopwords cuesta 3,3 puntos de F1 (0,683 → 0,716); lematizar no cambia nada (0,716 frente a 0,717); los bigramas suman 2,3 puntos y las negaciones 0,8.

**Errores:** de 12 000 reseñas de prueba hay 2 870 errores, pero solo 342 son graves (positivo ↔ negativo); el 88 % involucra la clase neutro.

## Comparación con un Transformer (offline)

`resenas/baseline_transformer.py` compara el modelo final con **RoBERTuito** (`pysentimiento/robertuito-sentiment-analysis`) sobre una submuestra estratificada del **mismo conjunto de prueba**. El Transformer **no se reentrena** con MeLiSA (fue ajustado con tweets): la comparación mide la transferencia de dominio, no un ajuste fino. No forma parte de la API ni del dashboard.

```bash
pip install -r requirements-baseline.txt          # descarga PyTorch: pesado, no está en requirements.txt
python -m resenas.baseline_transformer --n 50     # prueba rápida
python -m resenas.baseline_transformer --n 2000   # corrida real → models/baseline_transformer.json
```

Resultado con 2 000 reseñas de prueba (las mismas para ambos modelos):

| Modelo | Exactitud | F1 macro | F1 neutro | ms/reseña |
|---|---|---|---|---|
| TF-IDF + regresión logística (final) | 0,769 | 0,734 | 0,534 | 0,2 |
| RoBERTuito, sin ajuste | 0,702 | 0,648 | 0,380 | 106,6 |

Diferencia de F1 macro (Transformer − clásico): −0,086, IC 95 % por bootstrap pareado [−0,112; −0,061] → significativa. Los tiempos son por lotes de 16 en el equipo de los autores y dependen del hardware.

## Estructura

```
scripts/download_data.py        obtención del dataset (MeLiSA)
resenas/preprocess.py           normalización, negaciones; lematización/stopwords (solo ablación);
                                opciones de emojis, alargamientos y tildes (apagadas por defecto)
resenas/train.py                TF-IDF + regresión logística, métricas (accuracy, F1, kappa, MCC)
resenas/evaluar.py              ablación, validación cruzada y análisis de errores
resenas/baseline_transformer.py comparación offline con un Transformer preentrenado
resenas/api.py                  endpoints /salud, /predecir, /predecir_lote y /metricas
streamlit_app.py                dashboard web (predicción, lote CSV, métricas, comparación)
docs/informe.pdf, informe.tex   informe de la Evidencia 4 (máx. 4 páginas)
tests/                          pruebas automatizadas (pytest)
.github/workflows/tests.yml     CI: ejecuta pytest en cada push y pull request
requirements.txt                dependencias del proyecto
requirements-baseline.txt       dependencias opcionales (PyTorch + transformers)
data/raw/                       datos descargados (ignorados por git)
models/                         modelo y métricas entrenados (ignorados por git)
```

## Técnicas de NLP utilizadas

**En el modelo final:**
- **Normalización y tokenización** del texto (URLs fuera, minúsculas, espacios).
- **Marcado de ámbito de negación** (Das & Chen, 2001): las 3 palabras tras «no», «nunca», «sin»… se marcan con `neg_` (`no funciona` → `no neg_funciona`) hasta la siguiente puntuación o «pero/aunque/y/que».
- **Representación TF-IDF** con unigramas y bigramas (`sublinear_tf`).
- **Clasificación** con regresión logística (`class_weight="balanced"`), con explicabilidad a partir de los coeficientes (peso TF-IDF × coeficiente).

**Evaluadas pero no adoptadas:** lematización y filtrado de stopwords con spaCy, Bag of Words, unigramas vs. bigramas, Naive Bayes, SVM lineal y un Transformer preentrenado sin ajuste (RoBERTuito).

**Opciones de preprocesamiento disponibles (apagadas por defecto):** emojis convertidos en palabras, colapso de alargamientos («buenísimooo» → «buenísimo») y eliminación de tildes salvo la ñ. Se activan con `normalizar(texto, emojis=True, alargamientos=True, tildes=True)`; la ablación de `resenas.evaluar` mide su efecto.

## Decisiones de diseño

- **El preprocesamiento final NO lematiza ni quita stopwords.** La Evidencia 3 usaba spaCy (lemas, sin stopwords). La ablación separó los dos efectos: **quitar stopwords empeoró el F1 macro en 3,3 puntos** (0,683 → 0,716), probablemente porque se pierden conectores con información («pero», «para»); en cambio, **lematizar no cambió el resultado** (0,716 con lemas frente a 0,717 con texto normalizado). Como no aporta mejora, se descartó por simplicidad y velocidad: la API no depende de spaCy y predice más rápido. spaCy quedó como línea base de la ablación.
- **Marcado de negaciones** en lugar de depender solo de bigramas: captura negaciones a distancia («no es nada bueno»).
- **Bigramas (1–2) en TF-IDF** y `sublinear_tf`; un único TF-IDF para todo el texto.
- **`class_weight="balanced"`** para compensar que «neutro» (solo 3 estrellas) tiene la mitad de ejemplos que las otras clases.
- **Se mantiene el modelo clásico en producción:** un Transformer preentrenado sin ajuste rindió peor y fue cientos de veces más lento.
- **Las entradas se validan al predecir** (3–5000 caracteres y alguna letra o número), de modo que la API y el dashboard rechazan textos como `"???"` en lugar de devolver una clase sin sentido.
- Partición 80/20 estratificada con semilla fija (`random_state=42`); todas las configuraciones de la ablación comparten el mismo conjunto de prueba, y la configuración final se valida además con K-Fold (k=5).
- **Las pruebas no dependen de la descarga real**: usan un dataset sintético mínimo (`tests/conftest.py`).

## Limitaciones conocidas

- **Ruido de etiquetas:** el mapeo por estrellas es ruidoso. Los errores más «seguros» del modelo suelen ser reseñas como «Todo excelente» con 2 estrellas: el modelo acierta el sentimiento del texto pero no la etiqueta. La clase **neutro** (3 estrellas) es la más difícil: concentra la gran mayoría de los errores.
- Modelo de bolsa de palabras: no captura bien ironía, sarcasmo ni contexto largo; una reseña con opiniones mezcladas se resuelve por las palabras dominantes. El marcado de negación es una heurística con ventana fija de 3 palabras.
- Entrenado solo con reseñas de Mercado Libre (varias categorías de producto); puede generalizar peor a otros dominios.
- Las configuraciones de la ablación se comparan y se reporta la ganadora sobre el mismo conjunto de prueba (leve optimismo). Con una desviación de ≈0,003 en validación cruzada, diferencias menores a 0,01 no son concluyentes.
- El modelo final descarta los emojis y no trata alargamientos ni tildes (existen opciones para ello, apagadas por defecto).
- Solo se evaluó un Transformer preentrenado **sin ajuste** (entrenado con tweets), sobre 2 000 reseñas y en un único equipo. No se hizo ajuste fino sobre MeLiSA (requiere GPU), por lo que se desconoce lo que lograría un Transformer entrenado en el dominio; sería la siguiente mejora natural.
- No se midió la latencia de la API bajo carga; el modelo entrenado no se versiona en el repositorio (`models/` está ignorado), así que hay que entrenarlo para reproducir la demo.
- La descarga depende de una revisión fija del dataset en Hugging Face; se usa una muestra de 60 000 reseñas.

## Pruebas automatizadas

```bash
pytest          # 102 pruebas, menos de un minuto, no necesitan datos reales, modelo entrenado ni PyTorch
```

Cubren: preprocesamiento y marcado de negaciones (incluidas las opciones de emojis, alargamientos y tildes), limpieza del script de datos, particiones y métricas, entrenamiento end-to-end sobre datos sintéticos, la ablación/validación cruzada/análisis de errores, la API (estructura de respuesta, probabilidades, lote, validaciones `422`, `503` sin modelo, `/metricas`), el dashboard con `AppTest` de Streamlit y el baseline con Transformer (con un predictor simulado). La prueba del camino de spaCy se omite si el modelo no está instalado. Un workflow de GitHub Actions (`.github/workflows/tests.yml`) ejecuta `pytest` en cada push y pull request.

## Estado

- [x] Repositorio, script de datos, línea base, endpoint `/predecir`, interfaz web (Evidencia 3)
- [x] Comparación de preprocesamiento (lema vs. normalizado vs. negaciones) y BoW vs. TF-IDF
- [x] Evaluación completa: hold-out, validación cruzada y análisis de errores
- [x] Endpoints `/predecir_lote` y `/metricas`; dashboard con carga de CSV y comparación de modelos
- [x] Validación de entradas en la API y el dashboard
- [x] Comparación offline con un Transformer preentrenado
- [x] Pruebas automatizadas (preprocesamiento, entrenamiento, evaluación, API, dashboard) y CI
- [x] Informe (máx. 4 páginas): `docs/informe.pdf`