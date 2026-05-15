# 📘 Documentación PriceIQ

Guía detallada del proyecto paso a paso. Pensada para explicar la lógica,
las decisiones y los resultados a evaluadores académicos, clientes o
colegas técnicos.

---

## 0. Resumen ejecutivo

**Problema:** una tienda de retail recibe avisos de cambio de costo de su
proveedor con ~24 horas de anticipación. No tiene un proceso sistemático
para decidir el nuevo precio de venta — generalmente "subir el mismo %",
lo cual ignora la elasticidad real del producto y deja dinero en la mesa.

**Solución:** herramienta de recomendación que, dado un SKU y su nuevo
costo, devuelve:

1. **Rango recomendado de precio** (no un punto único).
2. **Demanda esperada** con intervalo de confianza calibrado.
3. **Margen esperado** y comparación contra el statu quo.
4. **Canibalización esperada** en productos sustitutos.
5. **Benchmark** de promociones históricas de la misma familia.
6. **Tarjeta de confianza** con score 0-100 y texto explicativo.

**Datos:** 22 meses (abril 2024 a febrero 2026) de ventas de un supermercado
de abarrotes mexicano:

- 4M líneas de venta válidas en ~7,300 SKUs
- 150K eventos de recepción del proveedor (historia de costos)
- 5,166 eventos de promoción detectables

**Métrica de éxito:** MAPE 17.3% mediano en la predicción de demanda diaria
(holdout de 60 días), cobertura empírica del IC80% = 81.1% (target 80%),
y validación quasi-experimental sobre 1,289 cambios reales de precio con
MAPE 17.7%.

---

## 1. Arquitectura general

El proyecto está construido en **9 fases** que se ejecutan en cascada. Cada
fase produce parquet(s) que alimentan a las siguientes.

```
   ┌─────────────────────────────────────────────────────────────┐
   │  Fase 1: ETL                                                │
   │     tablas_merksyst → panel_diario + costos_eventos + ...   │
   └─────────────────────────────────────────────────────────────┘
                                  │
       ┌──────────────────────────┼──────────────────────────┐
       ▼                          ▼                          ▼
   ┌─────────────┐         ┌──────────────┐         ┌────────────────┐
   │ Fase 2:     │         │ Fase 3:      │         │ Fase 5:        │
   │ Elasticidad │         │ Canastas     │         │ Promociones    │
   │ propia      │         │ + vecinos    │         │ históricas     │
   │ (4 iter.)   │         │ (lift)       │         │ (descriptivo)  │
   └─────────────┘         └──────────────┘         └────────────────┘
       │                          │                          │
       └──────────────┬───────────┘                          │
                      ▼                                      │
              ┌──────────────────┐                           │
              │ Fase 4:          │                           │
              │ Cruzadas         │                           │
              │ (β_ij + Bayes)   │                           │
              └──────────────────┘                           │
                      │                                      │
                      └────────────┬─────────────────────────┘
                                   │
                                   ▼
         ┌──────────────────────────────────────────────┐
         │  Fase 6: ML moderno                          │
         │    Item2Vec embeddings → LightGBM/XGB/CAT    │
         │    → Conformal Prediction                    │
         └──────────────────────────────────────────────┘
                                   │
                                   ▼
         ┌──────────────────────────────────────────────┐
         │  Fase 7: Validación honesta                  │
         │    Backtest naive + Backtest quasi-experim.  │
         └──────────────────────────────────────────────┘
                                   │
                                   ▼
         ┌──────────────────────────────────────────────┐
         │  Fase 8: Recomendador + Dashboard            │
         │    Integra TODO en una interfaz Streamlit    │
         └──────────────────────────────────────────────┘
```

---

## 2. Fase 1 — ETL: construir el dataset analítico

**Script:** `build_dataset.py`

### 2.1 Punto de partida

Recibimos 21 archivos parquet exportados del sistema Merksyst del cliente.
Los principales:

| Archivo | Contenido | Filas |
|---|---|---|
| `pdvdet.parquet` | Líneas de venta (1 por SKU vendido en un ticket) | 4,013,897 |
| `pdvhdr.parquet` | Headers de transacciones (ticket completo) | 693,199 |
| `invart.parquet` | Catálogo de artículos con costos y precios actuales (snapshot) | 72,093 |
| `inviar.parquet` | Clasificación (línea, familia, marca) | 30,035 |
| `cprrde.parquet` | Recepciones de mercancía (cuando llega del proveedor) | 150,337 |
| `cprrme.parquet` | Headers de recepciones (fecha, proveedor) | 20,726 |
| `cprdet.parquet` | Detalle de órdenes de compra (con factor_uds caja↔pieza) | 212,696 |

### 2.2 Decisiones clave durante el ETL

**1. Filtros de validez:**
- `status='01'` en pdvdet (líneas válidas, no canceladas)
- `cantidad > 0` (excluir devoluciones)
- `tipo_promocion=' '` con espacio para línea sin promoción (descubrimos
  que el campo no estaba vacío sino con un espacio — gotcha del legacy)

**2. Reconstrucción del precio histórico:**

`invart` es un *snapshot* — solo tiene precios actuales, no historial.
Pero `pdvdet.precio_normal` sí registra el precio de lista al momento
de cada venta. Solución: **mediana diaria de `precio_normal` filtrando
promos** = el precio efectivo que el cliente cobró cada día.

**3. Reconstrucción del costo del proveedor:**

Joineamos `cprrde` (líneas de recepción con `costo_rec`) con `cprrme`
(headers con `fec_rec`) por `(ord, n_rec)`. Cada recepción = un evento
de costo del proveedor. Luego **forward-fill** del último costo por
SKU para tener el costo vigente cada día.

**4. Gotcha del padding:**

`cve_art` viene con 15 caracteres paddeados a la derecha con espacios.
Aplicamos `.str.strip()` siempre antes de joins. (Detectamos esto cuando
las búsquedas devolvían 0 resultados.)

**5. Factor de unidades (caja ↔ pieza):**

El costo en `cprrde` está por **caja** (factor_uds = 24 para Coca, 12
para otros, hasta 1050 para algunos productos a granel). El precio en
`pdvdet` está por **pieza**. Hay que convertir uno u otro. Decisión:
guardar `sku_factor_uds.parquet` y aplicar la conversión en inferencia
(el modelo se entrenó con la escala original; convertimos input/output
en el recommender).

### 2.3 Outputs de la Fase 1

| Archivo | Contenido | Tamaño |
|---|---|---|
| `data/panel_diario.parquet` | ⭐ Dataset canónico: (fecha × SKU) con precio, costo, qty, calendario, jerarquía | 7 MB |
| `data/ventas_diarias.parquet` | Agregado diario de ventas con/sin promo | 9 MB |
| `data/costos_eventos.parquet` | Log de eventos de cambio de costo (un row por recepción) | 1.6 MB |
| `data/sku_catalog.parquet` | Catálogo limpio con clasificación | 1.4 MB |

---

## 3. Fase 2 — Elasticidad propia (4 iteraciones)

### 3.1 La pregunta y el concepto

**Elasticidad** = cuánto cambia la demanda cuando cambia el precio,
expresado en porcentaje.

```
ε = % cambio en cantidad / % cambio en precio
```

Se estima con una regresión **log-log**:

```
log(qty) = α + β · log(precio) + controles + ε
```

Donde **β es directamente la elasticidad**.

### 3.2 Por qué no es trivial — los confusores

Si haces solo `log(qty) ~ log(precio)` sin más, te equivocas:

1. **Estacionalidad de oferta** (clásico en frescos): el plátano en
   temporada cuesta poco Y se vende mucho. Una regresión naive diría
   "precio bajo → demanda alta", confundiendo causalidad con co-movimiento
   estacional.
2. **Promociones**: 20% de los días son con descuento; si no separas, la
   "demanda alta a precio bajo" se confunde con efecto puro de precio.
3. **Endogeneidad (simultaneidad)**: el gerente sube el precio *porque*
   detecta demanda alta. Eso sesga el β hacia menos negativo (subestima).
4. **Stockouts**: días sin inventario aparecen como "qty=0 a precio X".
   No es efecto de precio, es falta de mercancía.

### 3.3 Iteración 1 — Baseline OLS

**Script:** `baseline_elasticity.py`

```
log(qty_no_promo) = α + β · log(precio_normal_sinpromo)
                       + γ_dow + γ_mes
                       + δ · log(costo)
                       + ε
```

Una regresión por SKU con OLS robusto (HC3 standard errors).

**Resultado:**
- 1,690 SKUs estimados
- Solo **187 (11.1%)** con β<0 significativo
- A nivel familia, mejor: 30 de 184 familias bien identificadas

**Conclusión honesta:** baseline débil. La mayoría de SKUs no tiene
suficiente variación de precio histórica para identificar elasticidad
individualmente.

### 3.4 Iteración 2 — IV-2SLS + splines + filtro de stockouts

**Script:** `elasticity_v2.py`

#### Variable instrumental con el costo

El gran problema era la **endogeneidad**: precio y demanda se mueven a la
vez. La solución es la **variable instrumental** (IV):

> Necesitamos una variable que **afecta el precio** pero **NO afecta la
> demanda directamente**.

El **costo del proveedor** califica perfecto:
- Pass-through: si el costo sube, la tienda sube el precio ✓
- Exclusión: al consumidor le importa solo el precio final, no el costo
  interno de la tienda ✓

Implementación con `linearmodels.IV2SLS`:

```
Primera etapa:  log(precio) = π₀ + π₁·log(costo) + controles + v
Segunda etapa:  log(qty)    = α  + β·log(precio_predicho) + controles + u
```

El β de la segunda etapa es la elasticidad **purificada de simultaneidad**.

#### Splines temporales finos

Agregamos dummies por **semana del año** (52 dummies) además de dow y mes.
Crítico para frescos donde la estacionalidad es semanal, no mensual.

#### Filtro de stockouts

Por cada SKU, removemos días donde `log(qty)` está en el percentil bajo
(z-score < -2). Quita los outliers sospechosos de falta de inventario.

#### Holdout temporal

Reservamos los **últimos 60 días** del panel como test set. No los usa el
modelo. Sirve para medir si el modelo predice o solo describe.

**Resultados v2:**

| Métrica | OLS | IV |
|---|---|---|
| SKUs con β<0, p<0.05, IC excluye 0 | 78 | **147** |
| First-stage F mediano | — | 143 (instrumento muy fuerte) |
| Mediana de diferencia IV − OLS | — | **-1.03** |

**Lectura clave:** OLS subestimaba la magnitud de la elasticidad en
**casi 1 unidad** — exactamente el sesgo de simultaneidad que la teoría
predice. El IV lo corrige.

**Pero el holdout reveló un problema:** RMSE in-sample = 0.41, RMSE
out-of-sample = 2.80 → el modelo no generaliza. Solo 15% de SKUs predicen
razonablemente. Esto motivó la siguiente iteración.

### 3.5 Iteración 3 — Mixed Effects jerárquico

**Script:** `elasticity_v3_hierarchical.py`

#### El concepto: partial pooling

Asumimos que los β de SKUs dentro de la misma familia **se parecen**:

```
β_SKU_i ~ Normal(β_familia, σ_familia²)
```

Implementación: regresión de efectos mixtos (`statsmodels.MixedLM`) una
por familia con:

- **Efectos fijos:** log(precio), log(costo), dow, mes
- **Efectos aleatorios por SKU:** intercepto α_i + pendiente b_i sobre log(precio)
- **β del SKU:** `β_i = β_familia + b_i` (BLUP, Best Linear Unbiased Predictor)

#### Por qué esto resuelve el problema

- SKU con MUCHA data → mantiene su β individual (poco pooling).
- SKU con POCA data → hereda el β de la familia (mucho pooling).
- TODOS los SKUs obtienen una estimación con IC que refleja honestamente
  la incertidumbre.

**Resultados v3:**

| Versión | SKUs con β confiable |
|---|---|
| OLS naive (v1) | 141 |
| IV-2SLS (v2) | 147 |
| **Jerárquico (v3)** | **420** |

| Holdout MAPE mediano | v2 OLS | **v3 jerárquico** |
|---|---|---|
| | 99.7% | **51.7%** |

El holdout MAPE bajó **a la mitad** porque el partial pooling reduce el
sobreajuste agudo de las regresiones por SKU.

### 3.6 Resumen de las iteraciones de elasticidad

```
v1: OLS naive               →  no controla endogeneidad         (11% SKUs)
   ↓
v2: + IV con costo          →  corrige simultaneidad            (147 SKUs)
    + splines semanales     →  estacionalidad fina
    + filtro stockouts      →  quita outliers
    + holdout 60 días       →  detecta sobreajuste
   ↓
v3: + Mixed Effects         →  cubre TODOS los SKUs vía pooling (420 SKUs)
                            →  IC honesto por SKU
                            →  MAPE holdout 99.7% → 51.7%
```

---

## 4. Fase 3 — Canastas y vecinos (lift)

**Script:** `build_baskets.py`

### 4.1 Objetivo

Identificar para cada SKU sus **sustitutos** (productos rivales que
compiten por la misma compra) y **complementos** (productos que se
compran juntos como bundle).

### 4.2 Método: lift de mercado

Reconstruimos cada ticket joineando `pdvdet` con `pdvhdr` por
`(Suc, FecOpe, Caja, NumTra)`. Cada ticket = una "canasta" del cliente.

Para cada par de SKUs (A, B):

```
lift(A,B) = P(A y B en mismo ticket) / [P(A) · P(B)]
```

Interpretación:
- `lift > 1` → **complementos** (aparecen juntos más que al azar)
- `lift < 1` dentro de la misma familia → **sustitutos** (rara vez juntos)

### 4.3 Resultados

- 551,821 tickets reconstruidos
- 15,964 relaciones de vecindad detectadas
- 3,445 SKUs tienen al menos un sustituto identificado

**Ejemplos textbook detectados:**
- Coca 600ml ↔ Pepsi 600ml: lift bajo, misma familia → sustitutos
- Tomate Saladet ↔ Tomate Bola: idem
- Pan Bimbo ↔ Jamón Pavo: lift alto → complementos (sándwich)
- Yogurt sabor A ↔ Yogurt sabor B (misma marca): lift alto → bundle

### 4.4 Output

`data/sku_neighbors.parquet` — un row por par (foco, vecino, tipo, lift,
score_sustituto)

---

## 5. Fase 4 — Elasticidad cruzada

**Scripts:** `cross_elasticity.py` (v1), `cross_elasticity_v2.py` (v2)

### 5.1 El concepto

Mientras la elasticidad propia mide *qty_A ~ precio_A*, la cruzada mide
*qty_A ~ precio_B*:

```
log(qty_A_t) = α + β_AA · log(p_A_t)
                  + Σ_{B ∈ vecinos(A)} β_AB · log(p_B_t)
                  + controles
```

- `β_AA < 0` (propia, sabido)
- `β_AB > 0` → **sustituto** (precio de B sube → demanda de A sube)
- `β_AB < 0` → **complemento** (precio de B sube → ambas caen)

### 5.2 Por qué no se estima ingenuamente

Con 7,800 SKUs, una matriz completa tiene **60M parámetros**. Imposible
con 22 meses de datos. Por eso solo modelamos las relaciones que **el
lift identificó previamente** — top 4 sustitutos + 6 complementos por
SKU focal.

### 5.3 Empirical Bayes para mejorar identificación

Implementación v2:

1. **Regresión OLS por SKU focal** con sus 10 vecinos como covariables.
2. **Empirical Bayes shrinkage por grupo** `(fam_A, fam_B, tipo)`:
   - Para cada grupo de pares similares, estimar μ_g y τ_g
   - Cada β_ij se "encoge" hacia μ_g con peso inverso a su precisión
   - Pares ruidosos heredan del grupo; pares bien medidos mantienen su valor

### 5.4 Resultados v2

- 3,178 pares estimados (top-10 vecinos por focal)
- 131 pares con CI estricto (signo + significancia + CI excluye 0)
- 980 grupos `(fam_A, fam_B, tipo)` con μ_g estimado

**Casos identificados:**

| Si sube precio de... | ...sube demanda de... | β_ij |
|---|---|---|
| Coca 600ml | Coca 1.25L | +3.84 |
| Pepsi 355ml | Coca 1.25L | +2.42 |
| Tomate Saladet | Tomate Bola | +0.61 |
| Manzana Golden | Manzana Golden Escolar | +0.46 |

**Limitación honesta:** la magnitud cruzada es difícil. Solo 131 pares
están "bien identificados". Para el resto reportamos solo la dirección.

---

## 6. Fase 5 — Promociones históricas (descriptivo)

**Script:** `promo_descriptive.py`

### 6.1 Motivación

Si tienes 700K líneas con `tipo_promocion='1'` y `NumProm` poblado,
**tienes un experimento natural masivo**. Cada promoción histórica es
un evento donde el precio cambió mucho — podemos medir directamente
qué pasó.

### 6.2 Algoritmo

**Para cada (cve_art, NumProm):**

1. Definir un **evento** = período contiguo con esa promoción.
2. **Baseline pre-promo**: 14 días anteriores sin promo en ese SKU.
3. **Métricas del evento:**
   - `descuento_pct` = 1 − precio_medio_durante / precio_normal
   - `uplift_pct` = (qty/día durante) / (qty/día baseline) − 1
4. **Canibalización en sustitutos:**
   - Para cada sustituto del SKU promovido, comparar qty del sustituto
     durante el evento vs su baseline pre-promo (controlando que el
     sustituto NO esté él mismo en promo).

### 6.3 Resultados clave

- **5,166 eventos** de promoción detectados en 1,300 SKUs
- Mediana descuento: 20%
- **Mediana uplift propio: +30.6%**
- Mediana canibalización en sustitutos: **−17%**
- **50.5% de promociones NO fueron rentables** (ingreso bruto < baseline)

**Hallazgo de negocio importante:** el cliente está dejando dinero en
la mesa cuando arma promos a ojo — la mitad pierde plata cuando se
descuenta el efecto-baseline. Esto valida la propuesta de valor del
proyecto.

### 6.4 Outputs

- `data/promo_events.parquet` — un row por evento con uplift propio
- `data/promo_canibalization.parquet` — un row por (evento × sustituto)

---

## 7. Fase 6 — ML moderno

### 7.1 Item2Vec: embeddings densos de SKU

**Script:** `item2vec.py`

#### Idea

Tratar cada ticket como una "oración" de SKUs y entrenar Word2Vec
(Skip-Gram) para aprender un **vector denso de 32 dimensiones** por
cada SKU.

```python
gensim.Word2Vec(
    sentences=tickets,   # 551,821 listas de SKUs
    vector_size=32,
    window=8,            # contexto en la canasta
    sg=1,                # skip-gram
    negative=10, epochs=25
)
```

#### Por qué funciona

Productos que se compran en contextos similares (mismas canastas) acaban
con vectores similares — captura **similitud semántica continua** que
el lift binario perdía.

#### Validación cualitativa

| SKU foco | Top vecinos por similitud cosine |
|---|---|
| Coca 600ml | Pepsi 600ml, Joya 600ml, Fresca 600ml, chips |
| Aguacate Hass | Tomate Bola, Limón, Cebolla, Chile Serrano |
| Pan Bimbo | Jamón Pavo, Queso Americano, Pan Integral |
| Pechuga Pollo | Pechuga sin Piel, Milanesa Pollo, Tilapia |

Aguacate → tomate/limón/cebolla es ORO: el embedding aprendió "ingredientes
de salsa" sin que se lo digamos.

#### Métrica

37.4% de vecinos top-10 están en la misma familia (más alto sería forzado;
muchos vecinos válidos son inter-familia como Aguacate→Tomate).

**Output:** `data/sku_embeddings.parquet` (7,158 SKUs × 32 dim).

### 7.2 LightGBM con quantile regression

**Script:** `lgbm_demand.py`

#### Features (53 totales)

- **Numéricos (15):** log_p, log_c, margen_unit, margen_pct,
  dias_desde_cambio_costo, frac_promo, log_qty_lag1, log_qty_lag7,
  log_qty_ma7 (media móvil 7d), log_p_lag1, margen_unit_lag1, dow, mes,
  año, woy.
- **Categóricos label-encoded (6):** lin, fam, marca, tipo_articulo,
  cve_pro, proveedor_ult.
- **Embeddings (32):** los vectores Item2Vec.

#### Tres modelos por cuantil

Tres LightGBM independientes con `objective="quantile"` y `alpha = {0.1, 0.5, 0.9}`:

- q=0.5 → predicción central (mediana de la demanda)
- q=0.1 → cuantil bajo del IC
- q=0.9 → cuantil alto del IC

#### Hiperparámetros principales

```python
n_estimators=1500, learning_rate=0.05,
num_leaves=63, max_depth=-1,
min_child_samples=20,
feature_fraction=0.85, bagging_fraction=0.85,
early_stopping_rounds=50
```

Con early stopping sobre el split de validación (30 días).

#### Resultados

| Métrica | LightGBM | Estructural v3 |
|---|---|---|
| RMSE log (test) | 0.601 | 0.683 |
| **MAPE qty mediano** | **17.3%** | 51.7% |
| Cobertura IC80% | **79.4%** | — |

**3x mejor MAPE que el estructural** — gracias a los lags y embeddings.

#### Feature importance (qué importa realmente)

| Grupo de feature | % del poder predictivo |
|---|---|
| Lags de qty (autocorrelación) | 63.2% |
| **Embeddings Item2Vec** | **17.0%** |
| Precio y costo | 7.8% |
| Calendario (dow, mes, woy) | 7.1% |
| Categóricos (familia, marca, etc.) | 4.9% |

**Los embeddings valen ~17%** — validación cuantitativa de que el
Item2Vec NO es decorativo.

### 7.3 Comparación con otros modelos

**Script:** `model_comparison.py`

Entrenamos también XGBoost, CatBoost, MLP (3 capas) y un Stacking
(meta-learner Ridge):

| Modelo | RMSE log | MAPE | Cobertura IC80% |
|---|---|---|---|
| LightGBM | 0.601 | **29.1%** | **79.4%** ✓ |
| XGBoost | 0.591 | 29.5% | 78.9% |
| CatBoost | 0.597 | 29.4% | 54.4% ⚠️ |
| MLP | 0.565 | 35.5% | — |
| Stacking | **0.561** | 34.6% | — |

**Hallazgo contra-intuitivo:** el MLP gana RMSE pero pierde MAPE.
El Stacking lo combinó con peso 0.72 para MLP — los modelos se
complementan. CatBoost tiene IC mal calibrados (cobertura 54%).

**Decisión final:** mantener LightGBM como backbone (mejor IC) y usar
todos los otros como evidencia de robustez en la tesis.

### 7.4 Per-SKU fine-tuning (intento que no resultó tan bien)

**Script:** `per_sku_finetune.py`

Probamos entrenar un modelo LGBM pequeño por SKU sobre los **residuales**
del modelo global. Idea: el local captura patrones idiosincráticos.

**Resultado:** mejora de solo ~0.3pp en MAPE global. El modelo global ya
usa `cve_pro`, `marca`, embeddings — esos features ya capturan la mayoría
de la especificidad por SKU.

**Decisión honesta:** mantener los 164 modelos locales (no estorban) pero
no es la palanca grande que esperaba.

### 7.5 Conformal Prediction adaptativo

**Script:** `conformal_adaptive.py`

#### El problema

El IC del LightGBM cubre 79.4% empírico pero **por SKU varía mucho** —
algunos SKUs tienen cobertura del 50%, otros del 95%. Eso es injusto
para el cliente.

#### CQR Mondrian (Conformalized Quantile Regression con grupos)

1. En el set de validación, calcular para cada observación:
   ```
   score_i = max(qhi_i − y_i, y_i − qlo_i, 0)
   ```
   Mide cuánto "se pasó" el IC original.

2. Para cada **SKU con ≥30 obs en val**, calcular el cuantil 80% de sus
   scores. Eso es el "ajuste" para ese SKU.

3. Para **familias con ≥100 obs**, calcular cuantil grupal (fallback).

4. **Para predecir en test:**
   ```
   q10_calibrado = q10_modelo − q_ajuste_grupo
   q90_calibrado = q90_modelo + q_ajuste_grupo
   ```

#### Resultados

| Métrica | Original LGBM | Calibrado CQR |
|---|---|---|
| Cobertura global | 79.4% | **81.1%** |
| % SKUs con cobertura <70% | 18.7% | **15.0%** |
| Caso extremo (Refresco Manzanada) | 6.7% | **100%** |

**Garantía teórica:** la cobertura marginal ≥ 80% se mantiene por grupo
(propiedad de intercambiabilidad).

---

## 8. Fase 7 — Validación honesta

### 8.1 Backtest naive (`backtest.py`)

Para cada (SKU × día) en los últimos 60 días, ejecuta el recomendador
con el costo de ese día y compara su sugerencia vs el precio que el
gerente realmente usó.

**Resultado:**
- Ganancia teórica agregada: **+$709K (+28.7%)** según el modelo
- PERO: solo 20.6% de SKUs tienen ganancia > 0
- La ganancia se concentra en **frescos y aves** (productos inelásticos
  de alto volumen donde subir 5-10% sí da margen)

**Lectura honesta:** el agregado se ve atractivo pero está concentrado.
Para el cliente, vendemos "alertas en SKUs específicos donde hay
oportunidad", no "subir todo".

### 8.2 Backtest quasi-experimental (`backtest_quasi.py`)

**Esto es la validación más importante.**

#### Concepto

En los últimos 60 días, detectamos **1,289 cambios reales de precio**
(≥5% sobre la mediana móvil de 7 días previos). Para cada cambio:

- Comparamos qty predicha al precio nuevo vs qty real observada después
  del cambio.
- El precio nuevo es un valor "no visto" por el modelo en entrenamiento
  — exactamente lo que el caso de uso de la herramienta requiere.

#### Resultados

| Métrica | Valor | Lectura |
|---|---|---|
| **MAPE en cambios reales** | **17.7%** | El modelo predice ±18% cuando el precio se mueve |
| Bias mediano | -0.87% | Sin sesgo sistemático |
| **Cobertura IC80% empírica** | **81.9%** | El IC calibrado funciona en eventos reales ✓ |
| Acierto de dirección | 53% | Bueno en magnitudes, débil en dirección |

#### Por familia

| Familia | n_eventos | MAPE | Dirección |
|---|---|---|---|
| VERD (verduras) | 447 | 14.9% | 54% |
| FRU1 (frutas) | 219 | 15.4% | 61% |
| AVES (pollo) | 16 | 14.3% | 56% |
| REFR (refrescos) | 206 | 18.8% | 48% |

**Esto es la métrica defendible:** el modelo no es perfecto, pero
predice cantidad ±15-18% sobre cambios reales históricos, con IC
empíricamente calibrado.

---

## 9. Fase 8 — Recomendador y dashboard

### 9.1 Recommender (`recommender.py`)

Módulo Python que integra todos los artefactos:

- Carga LGBM q10/q50/q90 + modelos locales + cuantiles conformal
- Carga embeddings Item2Vec + vecindarios
- Carga cross-elasticities + promociones históricas + factor_uds

Funciones principales:

```python
get_sku_state(sku)              # estado actual: precio, costo, demanda
predict_demand_curve(sku, cost) # curva precio→demanda con IC
recommend_price(sku, cost)      # precio óptimo + IC + credibilidad
recommended_range(rec)           # rango ≥95% del óptimo
confidence_card(sku, rec)       # score 0-100 + texto explicativo
get_substitutes(sku, k=5)       # top-k sustitutos
estimate_canibalization(sku, p) # efecto en sustitutos
get_promo_benchmarks(sku)        # promos históricas
```

### 9.2 Algoritmo de recomendación paso a paso

Cuando el cliente pide una recomendación para (SKU, nuevo_costo):

```
1. Construir grilla de 50 precios candidatos
   entre [costo·1.05, max(p_actual·1.3, p_hist_max·1.25)]

2. Para cada precio en la grilla:
   a. Construir feature vector usando la última fila del panel
   b. Predecir log(qty) con LightGBM q10, q50, q90
   c. Si SKU tiene modelo local, aplicar corrección residual
   d. Aplicar ajuste conformal: q10 -= q_adj, q90 += q_adj
   e. Convertir de log a unidades: qty = exp(yhat)

3. Calcular margen esperado para cada precio:
   margen = (precio − costo) × qty_mediana

4. Penalización por extrapolación:
   credibilidad(p) = gaussian(distancia al rango histórico)
                    × gaussian(distancia al precio actual, peso=conservatism)

5. Optimizar:
   precio_óptimo = argmax(margen × credibilidad)

6. Definir rango seguro:
   zona = precios donde margen×credibilidad ≥ 95% del óptimo
          Y credibilidad ≥ 60%
```

### 9.3 Tarjeta de confianza

Score 0-100 que combina 5 señales:

```
score = 0.30 · credibilidad_modelo
      + 0.25 · precisión_familia        (1 − MAPE_familia)
      + 0.20 · cobertura_IC_familia
      + 0.15 · acierto_dirección_familia
      + 0.10 · suficiencia_datos_SKU
```

Etiqueta: ALTA (≥75) / MEDIA (≥55) / BAJA (<55).

**Texto auto-generado** en español que el cliente puede leer:

> "Confianza ALTA (82/100). Modelo entrenado con 680 días de historia
> del producto. En la familia REFR, evaluamos 206 cambios de precio
> reales: el modelo predijo cantidad con error mediano de 18.8%, acertó
> la dirección del cambio en 48% de los casos, y su intervalo de
> confianza cubrió la realidad en 83%."

### 9.4 Dashboard (`dashboard.py`)

Streamlit con 4 secciones:

1. **Sidebar**: selector de SKU, input de nuevo costo (% o valor),
   objetivo (margen/ingreso/cantidad), slider de conservadurismo.
2. **Sección 2 — Recomendación**: tarjeta de confianza prominente,
   rango seguro + precio óptimo en métricas grandes, gráficos
   precio→demanda/margen/ingreso con bandas de IC y banda verde del
   rango recomendado, expander con desglose de la confianza, expander
   con la elasticidad estructural.
3. **Sección 3 — Canibalización**: top-5 sustitutos por embedding con
   sus β_ij cruzados y Δ demanda esperada.
4. **Sección 4 — Promociones**: tabla de promos históricas del SKU y de
   su familia con uplift y descuento.

Al final, **expander de validación**: métricas del backtest
quasi-experimental para que el cliente sepa por qué confiar.

---

## 10. Hallazgos y limitaciones honestas

### 10.1 Hallazgos clave

✅ **El modelo funciona donde más importa:** frescos y aves, productos de
alto volumen con elasticidad moderada. Ahí está la oportunidad real.

✅ **50% de promociones históricas no fueron rentables** — insight de
negocio sólido que justifica la propuesta de valor del proyecto.

✅ **Los embeddings Item2Vec aportan 17%** del poder predictivo — el ML
moderno no es decorativo.

✅ **El IC empírico cubre 82%** en eventos reales — el cliente puede
confiar en los intervalos.

✅ **El sesgo IV-OLS de -1.03** valida que el costo es un instrumento
sólido (libro de texto).

### 10.2 Limitaciones honestas

⚠️ **Acierto de dirección del 53%** — el modelo es bueno en magnitudes,
mediocre en saber si la demanda sube o baja con un cambio de precio.
Esto es porque hay confusores no observados (eventos de competencia,
clima, etc.).

⚠️ **Solo 420 SKUs con elasticidad propia estricta** — los demás se
modelan con pooling de familia, lo cual es honesto pero menos
específico.

⚠️ **El backtest naive sugiere ganancia agregada +28%** pero está
concentrada en ~20% de SKUs. No es uniforme.

⚠️ **Per-SKU fine-tuning solo dio 0.3pp** — la palanca esperada de 12-18%
fue una sobre-expectativa que el experimento honestamente refutó.

⚠️ **Una sola tienda, 22 meses** — la generalización a otras tiendas
o periodos requiere validación adicional.

### 10.3 Próximos pasos (post-MVP)

- **Validación en producción A/B testing**: deployar a un subconjunto de
  SKUs y medir margen ganado vs control.
- **Modelo bayesiano completo (PyMC)** para cruzadas: mejor cuantificación
  de incertidumbre conjunta.
- **TabNet / FT-Transformer**: deep learning para tabular si quieres
  pushear más en ML.
- **Penalización de competencia**: si tienes datos de precios del rival,
  agregarlo como feature.

---

## 11. Cómo correr el proyecto

Ver sección detallada en `README.md`. Resumen:

```bash
# Setup
pip install -r requirements.txt

# Colocar tablas_merksyst_parquet/ con los datos del cliente

# Pipeline en orden (~45 min total)
python3 build_dataset.py
python3 baseline_elasticity.py
python3 elasticity_v2.py
python3 elasticity_v3_hierarchical.py
python3 build_baskets.py
python3 cross_elasticity_v2.py
python3 promo_descriptive.py
python3 item2vec.py
python3 lgbm_demand.py
python3 conformal_adaptive.py
python3 backtest_quasi.py

# Lanzar dashboard
python3 -m streamlit run dashboard.py
```

---

## 12. Diccionario de archivos

### Scripts (en orden de ejecución)

| Script | Propósito | Tiempo aprox |
|---|---|---|
| `build_dataset.py` | ETL desde Merksyst → panel analítico | 2 min |
| `baseline_elasticity.py` | OLS naive baseline | 3 min |
| `elasticity_v2.py` | IV-2SLS + splines + stockout | 5 min |
| `elasticity_v3_hierarchical.py` | Mixed Effects jerárquico | 10 min |
| `build_baskets.py` | Tickets, lift, vecinos | 3 min |
| `cross_elasticity_v2.py` | Cruzadas + Empirical Bayes | 5 min |
| `promo_descriptive.py` | Eventos históricos de promo | 15 min |
| `item2vec.py` | Embeddings Word2Vec | 3 min |
| `lgbm_demand.py` | LightGBM quantile | 5 min |
| `model_comparison.py` | XGB/CatBoost/MLP/Stacking (opcional) | 15 min |
| `per_sku_finetune.py` | Correcciones locales (opcional) | 10 min |
| `conformal_adaptive.py` | CQR Mondrian | 1 min |
| `backtest.py` | Backtest naive (opcional) | 20 min |
| `backtest_quasi.py` | Backtest sobre cambios reales | 1 min |
| `recommender.py` | Módulo de inferencia | — |
| `dashboard.py` | Streamlit | — |

### Datos derivados principales (carpeta `data/`)

| Archivo | Generado por | Contenido |
|---|---|---|
| `panel_diario.parquet` | build_dataset | Dataset analítico canónico |
| `costos_eventos.parquet` | build_dataset | Log de eventos de costo del proveedor |
| `sku_catalog.parquet` | build_dataset | Catálogo limpio |
| `sku_factor_uds.parquet` | build_dataset | Factor caja↔unidad por SKU |
| `elasticidades_jerarquico_sku.parquet` | elasticity_v3 | β_i por SKU (jerárquico) |
| `sku_neighbors.parquet` | build_baskets | Sustitutos/complementos por lift |
| `cross_elasticities_v2.parquet` | cross_elasticity_v2 | β_ij con shrinkage |
| `promo_events.parquet` | promo_descriptive | Eventos de promoción |
| `sku_embeddings.parquet` | item2vec | Vectores Item2Vec |
| `item2vec_model.kv` | item2vec | Modelo gensim para nearest-neighbor |
| `lgbm_predictions.parquet` | lgbm_demand | Predicciones en test |
| `conformal_quantiles.parquet` | conformal_adaptive | Ajustes CQR por SKU/familia |
| `backtest_quasi_events.parquet` | backtest_quasi | Eventos de cambio real |
| `credibilidad_por_familia.parquet` | backtest_quasi | Métricas para tarjeta de confianza |

### Modelos (carpeta `models/`)

| Archivo | Modelo |
|---|---|
| `lgbm_q10.txt`, `lgbm_q50.txt`, `lgbm_q90.txt` | LightGBM quantile (3 modelos) |
| `xgb_q{10,50,90}.json` | XGBoost quantile |
| `cat_q{10,50,90}.cbm` | CatBoost quantile |
| `mlp.pkl` | MLP + scaler |
| `stack.pkl` | Stacking Ridge |
| `local_lgbm/{sku}.txt` | 164 modelos LGBM locales por SKU top |

---

## 13. Glosario rápido

| Término | Definición |
|---|---|
| **Elasticidad propia (β_ii)** | % cambio en demanda / % cambio en precio del mismo producto |
| **Elasticidad cruzada (β_ij)** | % cambio en demanda de A / % cambio en precio de B |
| **Endogeneidad / simultaneidad** | Sesgo cuando dos variables se determinan mutuamente |
| **Variable instrumental (IV)** | Variable que afecta el regresor pero no la variable dependiente directamente |
| **2SLS** | Two-Stage Least Squares — método estándar para estimar IV |
| **Partial pooling** | Compartir información entre grupos (en Bayes o Mixed Effects) |
| **BLUP** | Best Linear Unbiased Predictor — el random effect estimado para cada grupo |
| **Empirical Bayes** | Estimar la prior desde los mismos datos (no fully Bayesian) |
| **Conformal Prediction** | Técnica que garantiza cobertura del IC sin asumir distribución |
| **CQR** | Conformalized Quantile Regression — variante para cuantiles |
| **Mondrian** | Versión adaptativa de Conformal por grupos |
| **MAPE** | Mean Absolute Percentage Error |
| **Pinball loss** | Función de pérdida propia para cuantiles |
| **Item2Vec** | Aplicar Word2Vec a tickets de retail para aprender embeddings de SKU |
| **Skip-gram** | Variante de Word2Vec que predice el contexto desde el item central |
| **Holdout temporal** | Reservar los últimos N días para test, sin contaminar entrenamiento |
| **Quasi-experimental** | Validación usando variación natural (no aleatorización deliberada) |

---

*Documentación generada para el proyecto PriceIQ.
Última actualización: mayo 2026.*
