# Recomendador de compras de supermercado

Sistema de recomendación para un e-commerce de supermercado a domicilio, construido sobre el dataset público de **Instacart** (3,4 millones de órdenes). El objetivo es responder dos preguntas de negocio distintas:

1. **"Compra de nuevo"**: ¿qué va a volver a comprar el cliente?
2. **"Descubre algo nuevo"**: ¿qué producto que nunca ha comprado podría interesarle?

> Todas las métricas de este README salen de ejecuciones reales del notebook. Ninguna cifra se escribe a mano.

---

## Resultados principales (módulo 1: ALS)

**Muestra:** 20.000 usuarios (semilla 42). **Validación temporal:** se entrena con las órdenes 1 a n−2 de cada usuario y se valida con la orden n−1. La orden n (test) queda reservada para la medición final.

### Predicción de la próxima canasta (validación)

| Modelo | Precision@10 | Recall@10 | NDCG@10 | Aciertos nuevos por usuario |
|---|---|---|---|---|
| Popularidad | 0,0698 | 0,0680 | 0,0953 | 0,129 |
| ALS inicial | 0,1081 | 0,1416 | 0,1564 | 0,123 |
| ALS ajustado (Optuna) | 0,2106 | 0,2648 | 0,2905 | 0,024 |
| **Recompra** | **0,2622** | **0,3201** | **0,3838** | 0,000 |

- Una regla simple de **recompra** (los 10 productos que cada cliente más compra) supera a ALS ajustado en NDCG@10 (0,384 contra 0,291).
- Al optimizar NDCG, Optuna llevó a ALS a imitar la recompra: el NDCG subió un 86 %, pero los aciertos en productos nuevos cayeron un 80 %. **Se obtiene lo que se optimiza.**

### Descubrimiento de productos nuevos (validación)

Las recomendaciones excluyen lo que el cliente ya compró y se evalúan solo contra los productos **nuevos** de su orden siguiente. El 83 % de los clientes (16.636 de 20.000) compró al menos un producto nuevo.

| Modelo | Precision@10 | Recall@10 | NDCG@10 |
|---|---|---|---|
| Popularidad nueva para ti | 0,0199 | 0,0386 | 0,0331 |
| **ALS descubrimiento** | **0,0240** | **0,0500** | **0,0403** |

ALS supera a la popularidad en un **22 %** de NDCG@10 en productos nuevos: de cada 1.000 clientes, acierta unos 240 productos nuevos frente a unos 199 de la popularidad.

### Conclusión de diseño

| Carrusel | Modelo | Por qué |
|---|---|---|
| "Compra de nuevo" | Recompra | Mejor NDCG@10 en la próxima canasta |
| "Descubre algo nuevo" | ALS (filtrando lo ya comprado) | +22 % sobre popularidad en productos nuevos |

---

## Detalle del módulo ALS

### Preparación de datos
- **Filtro de cola larga** (productos con menos de 5 compradores): el catálogo baja un 46,4 % (41.248 → 22.123 productos) perdiendo solo el 3,0 % de las interacciones.
- **Densidad de la matriz** usuario × producto: 0,157 % → 0,284 %.
- **Cobertura del test:** el 97,2 % de lo comprado en la última orden existe en el catálogo de ALS. Es el techo del modelo.
- **Fuga de información:** 0 órdenes de validación en el entrenamiento.
- **Cold-start real:** 1 usuario quedó sin perfil (todo su historial era de cola larga) y recibe popularidad como respaldo.

### Diagnóstico de ALS inicial (validación)
- Clientes con al menos 1 acierto: **59,9 %** con ALS frente a 43,9 % con popularidad.
- Tasa de acierto en la posición 1: 17,2 %; en la posición 10: 7,9 %. **El modelo ordena bien**: lo que pone arriba es más probable.
- Precision por tamaño de canasta: 0,061 (1–5 productos) → 0,182 (21+); el recall se mueve en sentido inverso (0,221 → 0,070).
- Composición de los aciertos: **89 % habituales, 11 % nuevos**.

### Ajuste de hiperparámetros (Optuna, 40 pruebas, sampler TPE)

| Objetivo | Mejores parámetros |
|---|---|
| NDCG@10 (próxima canasta) | BM25, alpha 1,02, 256 factores, reg 0,37, 10 iteraciones, K1 54,4, B 0,54 |
| NDCG@10 (solo productos nuevos) | BM25, alpha 0,44, 32 factores, reg 0,002, 15 iteraciones, K1 68,9, B 0,33 |

Importancia de los parámetros (objetivo próxima canasta): alpha 41 %, K1 28 %, B 18 %, iteraciones 9 %, método 2 %, regularización 1 %, factores 1 %. **La forma de ponderar cada compra importa mucho más que el tamaño del modelo.**

---

## Hoja de ruta

- [x] **ALS** sobre feedback implícito, ajuste con Optuna, líneas base de popularidad y recompra
- [ ] Medición final de ALS en test (orden n)
- [ ] Factorización matricial (SVD): elección de k y detección de patrones atípicos
- [ ] Reglas de asociación (FP-Growth): "comprados juntos" en el carrito
- [ ] Cold-start con segmentación (K-Means, HDBSCAN, GMM) y modelo basado en contenido
- [ ] Two-Tower con embeddings: "productos similares"
- [ ] Recomendación secuencial (SASRec)
- [ ] Ranker que combine candidatos de todos los modelos
- [ ] API en FastAPI con Docker, desplegada en Google Cloud Run, con latencia medida

---

## Cómo reproducir

```bash
git clone <url-del-repo>
cd recomendador-supermercado
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

1. Crea un token de API en Kaggle (Settings → API) y guárdalo según las instrucciones de Kaggle. **Nunca lo subas al repositorio.**
2. Abre `notebooks/01_als.ipynb` y ejecútalo en orden. La primera celda descarga los datos con `kagglehub`.

## Datos

- **Fuente:** Instacart Market Basket Analysis (Kaggle, espejo `psparks/instacart-market-basket-analysis`).
- **Licencia:** uso no comercial. Los datos **no** se incluyen en este repositorio; se descargan al ejecutar el notebook.
- Diccionario de datos completo en [`docs/diccionario_datos.md`](docs/diccionario_datos.md).

## Stack

Python · pandas · NumPy · SciPy (matrices dispersas) · implicit (ALS) · Optuna · Matplotlib
