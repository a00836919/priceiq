# 💰 PriceIQ

Herramienta de recomendación de precio de venta para retail cuando el proveedor
cambia el costo. Combina **modelos causales** (elasticidad jerárquica con IV) y
**ML moderno** (LightGBM + Item2Vec + Conformal Prediction) para producir
precios óptimos con intervalo de confianza calibrado.

Datos: 22 meses de ventas (2024-04 a 2026-02) de un supermercado de abarrotes,
~7,300 SKUs, ~4M líneas de venta, ~150K eventos de recepción de proveedor.

---

## 🎯 Problema

Cuando el proveedor anuncia un cambio de costo (típicamente 1 día de
anticipación), la tienda necesita decidir el nuevo precio de venta. PriceIQ:

1. **Predice la demanda** esperada a distintos precios (con IC).
2. **Recomienda un rango y un precio óptimo** que maximiza margen esperado.
3. **Cuantifica la confianza** de cada recomendación usando validación
   quasi-experimental sobre cambios reales históricos.
4. **Alerta sobre canibalización** en sustitutos identificados por embeddings.
5. **Aporta benchmarks de promociones** históricas de la misma familia.

---

## 🏗️ Arquitectura del pipeline

```
   tablas_merksyst_parquet/  (datos crudos del cliente, NO incluidos)
                │
                ▼
    build_dataset.py        →  data/panel_diario.parquet
                │              data/costos_eventos.parquet
                │              data/sku_catalog.parquet
                ▼
   ┌──────────────────────────────────────────────┐
   │ Modelos causales (interpretables)            │
   │ ─────────────────────────────────────────    │
   │ baseline_elasticity.py    (OLS naive)        │
   │ elasticity_v2.py          (IV-2SLS + splines)│
   │ elasticity_v3_hierarchical.py                │
   │   (Mixed Effects → β jerárquico por SKU)     │
   │                                              │
   │ build_baskets.py    (tickets, lift, vecinos) │
   │ cross_elasticity_v2.py                       │
   │   (OLS por focal + Empirical Bayes)          │
   │ promo_descriptive.py    (eventos históricos) │
   └──────────────────────────────────────────────┘
                │
                ▼
   ┌──────────────────────────────────────────────┐
   │ Modelos ML (predicción punto)                │
   │ ─────────────────────────────────────────    │
   │ item2vec.py              (embeddings de SKU) │
   │ lgbm_demand.py           (GBM + quantile)    │
   │ model_comparison.py      (XGB/CatBoost/MLP)  │
   │ per_sku_finetune.py      (correc. residual)  │
   │ conformal_adaptive.py    (CQR Mondrian)      │
   └──────────────────────────────────────────────┘
                │
                ▼
   ┌──────────────────────────────────────────────┐
   │ Validación honesta                           │
   │ ─────────────────────────────────────────    │
   │ backtest.py                                  │
   │ backtest_quasi.py (sobre cambios REALES)     │
   └──────────────────────────────────────────────┘
                │
                ▼
   recommender.py   ←  módulo unificado de inferencia
                │
                ▼
   dashboard.py     ←  Streamlit con 4 secciones
```

---

## 🚀 Ejecutar

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Colocar datos del cliente en tablas_merksyst_parquet/
#    (parquets exportados de Merksyst: pdvdet, pdvhdr, invart, inviar, etc.)

# 3. Construir dataset analítico
python build_dataset.py

# 4. Entrenar modelos (en orden)
python baseline_elasticity.py
python elasticity_v2.py
python elasticity_v3_hierarchical.py
python build_baskets.py
python cross_elasticity_v2.py
python promo_descriptive.py
python item2vec.py
python lgbm_demand.py
python conformal_adaptive.py
python backtest_quasi.py

# 5. Lanzar dashboard
streamlit run dashboard.py
```

El dashboard abre en `http://localhost:8501`.

---

## 📊 Resultados clave

| Componente | Métrica |
|---|---|
| Predicción de demanda (LightGBM + embeddings) | MAPE mediano **17.3%** en holdout |
| Cobertura IC80% (Conformal calibrado) | **81.1%** (target 80%) |
| Validación quasi-experimental (cambios reales de precio) | MAPE **17.7%** en 1,289 eventos |
| SKUs con elasticidad propia confiable (β identificada) | 420 con CI estricto |
| Vecinos por embedding (Item2Vec) | 91.9% cobertura, agrupación semántica |
| Eventos históricos de promoción analizados | 5,166 |

**Hallazgo de negocio:** ~50% de las promociones históricas no fueron rentables
(ingreso bruto < baseline sin promo). La herramienta sí detecta este patrón.

---

## 🧠 Decisiones técnicas notables

- **factor_uds**: el costo en `cprrde` viene por caja (factor 24 para Coca, etc.);
  se convierte a unitario en inferencia.
- **Conformal Prediction Mondrian**: cuantiles de calibración por SKU (≥30 obs en
  val) o por familia (fallback), garantizando cobertura ≥80%.
- **Penalización por extrapolación**: el optimizador maximiza `margen × credibilidad`
  donde la credibilidad decae fuera del rango histórico de precios.
- **Rango recomendado en vez de punto único**: zona donde `margen ≥ 95% óptimo` y
  `credibilidad ≥ 60%` — evita falsa precisión.

---

## 📁 Estructura

```
priceIQ/
├── build_dataset.py            # ETL desde Merksyst
├── baseline_elasticity.py      # Baseline OLS
├── elasticity_v2.py            # IV-2SLS
├── elasticity_v3_hierarchical.py # Mixed Effects
├── build_baskets.py            # Canastas + lift
├── cross_elasticity_v2.py      # Cruzadas + Empirical Bayes
├── promo_descriptive.py        # Análisis de promos
├── item2vec.py                 # Embeddings de SKU
├── lgbm_demand.py              # LightGBM quantile
├── model_comparison.py         # XGB/CatBoost/MLP/Stacking
├── per_sku_finetune.py         # Correcciones locales
├── conformal_adaptive.py       # CQR Mondrian
├── backtest.py                 # Backtest naive
├── backtest_quasi.py           # Backtest quasi-experimental
├── recommender.py              # Módulo de inferencia
├── dashboard.py                # Streamlit
├── requirements.txt
└── README.md
```

---

## 🛡️ Datos

Los datos crudos del cliente (`tablas_merksyst_parquet/`) NO están incluidos en
el repo por privacidad y tamaño (~180 MB). Los modelos entrenados y los
parquets derivados grandes tampoco — todo se regenera ejecutando los scripts
en orden.

---

## 📜 Licencia

Proyecto académico. Ver con el autor antes de uso comercial.
