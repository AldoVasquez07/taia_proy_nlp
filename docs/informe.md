# Clasificador de reseñas en español — Informe Evidencia 4

**Curso:** Tópicos Avanzados en IA (PLN) · **Autores:** Aldo Ray Vasquez Lopez, Alvaro Andre Machaca Melendez
**Repositorio:** https://github.com/AldoVasquez07/taia_proy_nlp

## 1. Problema y entregable
Clasificar reseñas de productos en **negativo / neutro / positivo**. Entregables: API REST (FastAPI: `/predecir`, `/predecir_lote`, `/metricas`, `/salud`) y dashboard web (Streamlit) con predicción individual, clasificación por lote de CSV, métricas y comparación de modelos. Cada predicción incluye los términos que más influyeron.

## 2. Datos
Dataset **MeLiSA** (reseñas de Mercado Libre, Hugging Face, revisión fija), muestra de 60 000 reseñas sin nulos ni duplicados, distribuidas casi uniformemente en 1–5 estrellas. Mapeo: 1–2★ → negativo (23 937), 3★ → neutro (11 968), 4–5★ → positivo (24 095). Partición 80/20 estratificada (semilla 42): 48 000 entrenamiento / 12 000 prueba. Uso solo académico; los datos no se suben al repositorio.

## 3. Técnicas de NLP y decisiones
1. **Normalización y tokenización** (URLs fuera, minúsculas).
2. **Marcado de ámbito de negación** (Das & Chen, 2001): las 3 palabras tras «no/nunca/sin/ni…» se prefijan con `neg_` hasta puntuación o «pero/aunque/y/que» (`no funciona` → `no neg_funciona`).
3. **TF-IDF** con unigramas + bigramas (`sublinear_tf`, `min_df=2`, 200 000 rasgos).
4. **Regresión logística** con `class_weight="balanced"` y explicabilidad por coeficientes × peso TF-IDF.
5. **spaCy** (lematización y stopwords) se evaluó como línea base de la ablación.

**Decisión clave:** la Evidencia 3 lematizaba y quitaba stopwords. La ablación mostró que eso *perjudica* (F1 macro 0.683 frente a 0.716 conservando stopwords), porque se pierde información («pero», «para») y el lematizador pequeño introduce errores. El modelo final usa texto normalizado + negaciones, que además es más rápido en producción.

## 4. Evaluación
Todas las configuraciones usan el mismo conjunto de prueba (12 000 reseñas).

| Configuración | Accuracy | F1 macro |
|---|---|---|
| Lemas sin stopwords + TF-IDF(1-2) + LogReg (base Evid. 3) | 0.723 | 0.683 |
| Lemas con stopwords + TF-IDF(1-2) + LogReg | 0.752 | 0.716 |
| Normalizado + BoW(1-2) + LogReg | 0.751 | 0.700 |
| Normalizado + TF-IDF(1) + LogReg | 0.727 | 0.694 |
| Normalizado + TF-IDF(1-2) + LogReg | 0.754 | 0.717 |
| Normalizado + negaciones + TF-IDF(1) + LogReg | 0.742 | 0.709 |
| **Normalizado + negaciones + TF-IDF(1-2) + LogReg (final)** | **0.761** | **0.725** |
| … + ComplementNB | 0.724 | 0.567 |
| … + LinearSVC | 0.755 | 0.696 |

**Modelo final (hold-out):** accuracy 0.761, F1 macro 0.72. Por clase (precisión/recall/F1): negativo 0.88/0.79/0.83 · neutro 0.48/0.57/0.52 · positivo 0.82/0.83/0.83.
**Validación cruzada 5-fold:** accuracy 0.759 ± 0.002, F1 macro 0.721 ± 0.003 (estable, sin sobreajuste a la partición).

**Conclusiones de la ablación:** (a) bigramas > unigramas (+2.3 F1); (b) TF-IDF > BoW (+1.7); (c) el marcado de negaciones suma +0.8 con bigramas y +1.5 con unigramas; (d) regresión logística supera a NB y SVM lineal en F1 macro.

**Análisis de errores:** de 12 000 reseñas hay 2 870 errores, pero solo 342 son graves (positivo↔negativo); el **88 %** involucra la clase neutro. Los fallos más «seguros» son casi siempre etiquetas ruidosas (p. ej. «Todo excelente» con 2★): el modelo acierta el sentimiento del texto, pero no la etiqueta derivada de estrellas.

## 5. Pruebas automatizadas
Suite `pytest` (~60 pruebas, segundos, sin datos reales): preprocesamiento y negaciones, script de datos, partición y métricas, entrenamiento end-to-end sobre datos sintéticos, ablación/validación cruzada/errores y API (estructura, probabilidades, lote, validaciones 422, 503 sin modelo, métricas). Las pruebas detectaron un bug real (porcentaje `NaN` cuando no hay errores, que habría generado JSON inválido).

## 6. Limitaciones
- Etiquetas derivadas de estrellas: ruido inevitable, sobre todo en «neutro» (techo práctico ≈ 0.72 F1 macro).
- Bolsa de palabras: no captura ironía ni opiniones mezcladas.
- Un solo dominio (Mercado Libre) y muestra de 60 000 reseñas.
- No se probaron modelos neuronales (Transformers) por costo de cómputo: siguiente mejora natural.
