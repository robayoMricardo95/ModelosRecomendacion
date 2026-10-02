# Diccionario de datos — Instacart Market Basket Analysis

Dataset público de Instacart (2017): más de 3 millones de órdenes de supermercado de unos 200.000 usuarios anónimos, cada uno con entre 4 y 100 órdenes. Es lo más parecido a Tipti que existe en datos públicos: compras de supermercado por app, hechas por shoppers y entregadas a domicilio.

- **Fuente:** Kaggle, espejo `psparks/instacart-market-basket-analysis`
- **Licencia:** uso no comercial. Los datos crudos **no** se suben al repositorio (`data/` está en `.gitignore`).

---

## 1. Modelo de datos

El dataset es relacional, como un modelo estrella de BI:

- **Hechos:** `order_products__prior` y `order_products__train` (qué producto va en qué orden).
- **Dimensiones:** `orders` (quién y cuándo), `products`, `aisles`, `departments`.

```
departments ──┐
              ├── products ──── order_products__prior / __train ──── orders
aisles ───────┘   (product_id)        (order_id)                     (user_id)
```

### Cabecera y detalle

`orders` y `order_products` funcionan como la cabecera y el detalle de una factura:

- **`orders` (cabecera):** una fila por orden. Dice quién compró y cuándo, pero no qué compró.
- **`order_products__*` (detalle):** una fila por cada producto dentro de una orden. Dice qué se compró.

Se unen por `order_id`.

### `prior` y `train` no son muestras: separan el tiempo

```
Usuario 1 — sus órdenes en orders:
order_number:  1      2      3   ...   10     11
eval_set:      prior  prior  prior ... prior  train
               └──── detalle en __prior ────┘  └ detalle en __train
                     (su historial)              (su última orden)
```

| eval_set | Qué contiene | Uso en el proyecto |
|---|---|---|
| `prior` | Todas las órdenes de cada usuario excepto la última | **Entrenar** |
| `train` | La última orden de ~131.000 usuarios | **Evaluar (test)**: el modelo nunca la ve al entrenar |
| `test` | La última orden de ~75.000 usuarios, sin productos (respuesta de la competencia) | No se usa |

Esto simula la vida real: con todo lo que el cliente compró antes, ¿adivinamos qué pedirá la próxima vez?

### Jerarquía del catálogo

```
departamento (21)  →  pasillo (134)  →  producto (~50.000)
```

Es la misma estructura de categorías de una app de supermercado, y permite recomendar o segmentar en el nivel que convenga.

---

## 2. Las 6 tablas

| Tabla | Filas aprox. | Qué es | Para qué la usamos |
|---|---|---|---|
| `orders` | 3,4 M | Cabecera de cada orden: quién, cuál de sus órdenes, día, hora y días desde la anterior | Ordenar el historial en el tiempo (validación temporal, SASRec), contexto (hora/día), separar train/test (`eval_set`) |
| `order_products__prior` | 32 M | Detalle del historial: productos de cada orden pasada | **Entrenar**: matriz de ALS, reglas de asociación, popularidad, recompra |
| `order_products__train` | 1,4 M | Detalle de la última orden de ~131 mil usuarios | **Evaluar** (test) |
| `products` | ~50 mil | Catálogo: nombre del producto y códigos de pasillo y departamento | Traducir IDs a nombres, cold-start, modelos de contenido |
| `aisles` | 134 | Diccionario de pasillos: código → nombre ("fresh fruits") | Análisis por pasillo, segmentación de usuarios |
| `departments` | 21 | Diccionario de departamentos: código → nombre ("produce", "dairy eggs") | Nivel más alto de la jerarquía del catálogo |

---

## 3. Columnas

### `orders` (cabecera)

| Columna | Tipo | Qué es | Para qué nos sirve |
|---|---|---|---|
| `order_id` | int32 | ID único de la orden | Llave para unir con el detalle |
| `user_id` | int32 | ID anónimo del cliente | Construir el historial de cada usuario |
| `eval_set` | texto | `prior`, `train` o `test` | Separar pasado y futuro |
| `order_number` | int16 | 1ª, 2ª, 3ª… orden del usuario | Ordenar en el tiempo; validación temporal y modelos secuenciales |
| `order_dow` | int8 | Día de la semana (0 a 6) | Contexto: fin de semana frente a entre semana |
| `order_hour_of_day` | int8 | Hora del pedido (0 a 23) | Contexto: compra de mañana frente a la de noche |
| `days_since_prior_order` | float32 | Días desde la orden anterior (máximo 30; vacío en la 1ª orden) | Frecuencia del cliente, ciclo de recompra |

### `order_products__prior` / `order_products__train` (detalle)

| Columna | Tipo | Qué es | Para qué nos sirve |
|---|---|---|---|
| `order_id` | int32 | A qué orden pertenece | Unir con `orders` para saber quién y cuándo |
| `product_id` | int32 | Qué producto | El "ítem" de la matriz usuario × producto |
| `add_to_cart_order` | int16 | Posición en que se agregó al carrito (1º, 2º…) | Secuencia dentro de la canasta; recomendaciones en el carrito |
| `reordered` | int8 | 1 si ya lo había comprado antes, 0 si es nuevo | Tasa de recompra; distinguir hábito de descubrimiento |

### `products`, `aisles`, `departments` (catálogo)

| Columna | Tabla | Qué es |
|---|---|---|
| `product_id` | products | ID único del producto |
| `product_name` | products | Nombre en texto ("Banana", "Organic Whole Milk") |
| `aisle_id` | products, aisles | Código del pasillo |
| `department_id` | products, departments | Código del departamento |
| `aisle` | aisles | Nombre del pasillo ("fresh fruits") |
| `department` | departments | Nombre del departamento ("produce") |

`products` solo trae los **códigos** de pasillo y departamento; los **nombres** están en `aisles` y `departments`. Por eso se unen:

```
products:     24852 | Banana | 24 | 4
aisles:       24 | fresh fruits
departments:  4  | produce
resultado:    24852 | Banana | 24 | fresh fruits | 4 | produce
```

Para los algoritmos bastan los códigos. Los nombres sirven para el análisis exploratorio, para leer las recomendaciones y para los embeddings de texto en cold-start.

---

## 4. Limitaciones a tener en cuenta

- **No hay fechas reales**, solo `order_number` y `days_since_prior_order`. La validación temporal se hace por número de orden.
- **`days_since_prior_order` está topado en 30.** Un cliente que volvió a los 90 días aparece como 30; no sirve para medir abandono largo.
- **No hay precios**, así que no se pueden calcular métricas de ingreso.
- **No hay marca ni demografía.** El cold-start de usuario se resuelve por comportamiento (primeras compras, segmentos), no por atributos personales.
- **No es una muestra aleatoria** de los usuarios de Instacart; los resultados ilustran el método, no el negocio real.

---

## 5. Decisiones de carga

- **Tipos de datos livianos** (`int32`, `int16`, `int8` en lugar de `int64`): bajan la memoria de ~1 GB a ~300 MB y permiten trabajar en una laptop. Es una decisión de costo, igual que el costo de inferencia en producción.
- **Controles de integridad referencial** antes de modelar: todo `product_id` vendido existe en el catálogo y todo `order_id` del detalle existe en `orders`. Los nulos de `days_since_prior_order` deben coincidir con el número de usuarios (una primera orden por usuario).
