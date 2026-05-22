# Graph Report - .  (2026-05-22)

## Corpus Check
- Corpus is ~35,751 words - fits in a single context window. You may not need a graph.

## Summary
- 214 nodes · 258 edges · 27 communities (15 shown, 12 thin omitted)
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 25 edges (avg confidence: 0.85)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Log-Log Elasticity Baseline|Log-Log Elasticity Baseline]]
- [[_COMMUNITY_Cross-Elasticity & Backtest Core|Cross-Elasticity & Backtest Core]]
- [[_COMMUNITY_Price Recommender Engine|Price Recommender Engine]]
- [[_COMMUNITY_Elasticity Diagnostic Dashboard|Elasticity Diagnostic Dashboard]]
- [[_COMMUNITY_Conformal Prediction & Uncertainty|Conformal Prediction & Uncertainty]]
- [[_COMMUNITY_Dataset Builder & Model Comparison|Dataset Builder & Model Comparison]]
- [[_COMMUNITY_EDA Price-Demand Visualization|EDA Price-Demand Visualization]]
- [[_COMMUNITY_Demand Model Artifacts & Finetune|Demand Model Artifacts & Finetune]]
- [[_COMMUNITY_Basket Analysis & SKU Embeddings|Basket Analysis & SKU Embeddings]]
- [[_COMMUNITY_Quasi-Experimental Validation|Quasi-Experimental Validation]]
- [[_COMMUNITY_Cross-Elasticity V2 EB Shrinkage|Cross-Elasticity V2 EB Shrinkage]]
- [[_COMMUNITY_Elasticity V2 IV-OLS|Elasticity V2 IV-OLS]]
- [[_COMMUNITY_Market Basket Co-occurrence|Market Basket Co-occurrence]]
- [[_COMMUNITY_Panel Dataset Construction|Panel Dataset Construction]]
- [[_COMMUNITY_Adaptive Conformal Quantiles|Adaptive Conformal Quantiles]]
- [[_COMMUNITY_Cross-Elasticity V1 OLS|Cross-Elasticity V1 OLS]]
- [[_COMMUNITY_Hierarchical Mixed-Effects Model|Hierarchical Mixed-Effects Model]]
- [[_COMMUNITY_Backtest Recommendation Logic|Backtest Recommendation Logic]]
- [[_COMMUNITY_Log-Log OLS Baseline|Log-Log OLS Baseline]]
- [[_COMMUNITY_Dev Tooling Config|Dev Tooling Config]]
- [[_COMMUNITY_Model Evaluation Metrics|Model Evaluation Metrics]]
- [[_COMMUNITY_Per-SKU Demand Finetune|Per-SKU Demand Finetune]]
- [[_COMMUNITY_Quasi-Experimental Backtest|Quasi-Experimental Backtest]]
- [[_COMMUNITY_Streamlit Dashboard|Streamlit Dashboard]]
- [[_COMMUNITY_Promo Profitability Analysis|Promo Profitability Analysis]]
- [[_COMMUNITY_LightGBM Demand Model|LightGBM Demand Model]]
- [[_COMMUNITY_Item2Vec Embeddings|Item2Vec Embeddings]]

## God Nodes (most connected - your core abstractions)
1. `Hierarchical Mixed-Effects Elasticity v3 (Partial Pooling)` - 11 edges
2. `LightGBM Demand Prediction with Quantile Regression` - 10 edges
3. `Identification Rate by Line (% SKUs with beta < -0.05)` - 10 edges
4. `Daily Panel Parquet (data/panel_diario.parquet)` - 9 edges
5. `PriceIQ Project Documentation` - 9 edges
6. `_load_all()` - 8 edges
7. `_load_all Cache Function` - 8 edges
8. `get_sku_state()` - 7 edges
9. `Elasticity v2 with IV-2SLS and Stockout Trim` - 7 edges
10. `Per-SKU Fine-Tuning (Residual Correction over Global LightGBM)` - 7 edges

## Surprising Connections (you probably didn't know these)
- `Canibalization Measurement Loop` --semantically_similar_to--> `estimate_canibalization Function`  [INFERRED] [semantically similar]
  promo_descriptive.py → recommender.py
- `Stacking Ridge Meta-Learner` --semantically_similar_to--> `Recommender Module`  [INFERRED] [semantically similar]
  model_comparison.py → recommender.py
- `Cost as Instrumental Variable for Price` --conceptually_related_to--> `Cross Elasticity Script`  [INFERRED]
  DOCUMENTACION.md → cross_elasticity.py
- `Quasi-Experimental Validation on Real Price Changes` --rationale_for--> `Backtest Script`  [INFERRED]
  DOCUMENTACION.md → backtest.py
- `CQR Mondrian Conformal Calibration` --rationale_for--> `predict_demand_curve Function`  [INFERRED]
  DOCUMENTACION.md → recommender.py

## Hyperedges (group relationships)
- **Demand Prediction Pipeline (LGBM + Item2Vec Embeddings + Conformal Calibration)** — lgbm_demand_script, item2vec_sku_embeddings_parquet, conformal_adaptive_script, conformal_predictions_conformal_parquet [EXTRACTED 0.95]
- **Versioned Elasticity Modeling Pipeline (v1 OLS → v2 IV-2SLS → v3 Hierarchical Mixed Effects)** — baseline_elasticity_script, elasticity_v2_script, elasticity_v3_script, elasticity_v3_jerarquico_sku_parquet [EXTRACTED 0.95]
- **Basket Analysis → SKU Embeddings → Cross-Elasticity Identification Chain** — build_baskets_script, item2vec_script, cross_elasticity_v2_script, build_baskets_sku_neighbors_parquet [INFERRED 0.85]
- **Demand Prediction Pipeline: Panel + Embeddings + LightGBM** — build_dataset_panel_diario, recommender_load_all, recommender_predict_demand_curve [INFERRED 0.95]
- **Canibalization Analysis Flow: Cross Elasticity + Promo Descriptive + Recommender** — cross_elasticity_ols_per_focal, promo_descriptive_canibalization_measurement, recommender_estimate_canibalization [INFERRED 0.85]
- **Model Ensemble Comparison: LightGBM + XGBoost + CatBoost + MLP + Stacking** — model_comparison_xgboost_quantile, model_comparison_catboost_quantile, model_comparison_mlp_model, model_comparison_stacking_meta [EXTRACTED 1.00]

## Communities (27 total, 12 thin omitted)

### Community 0 - "Log-Log Elasticity Baseline"
Cohesion: 0.10
Nodes (28): Elasticidades Familia Parquet v1 (data/elasticidades_fam.parquet), Elasticidades SKU Parquet v1 (data/elasticidades_sku.parquet), Baseline Own-Price Elasticity (OLS log-log), fit_loglog OLS Estimator Function, Empirical Bayes Shrinkage (Partial Pooling across Groups), Log-Log Demand Model (log_qty ~ log_price), Daily Panel Parquet (data/panel_diario.parquet), Temporal Holdout Split Strategy (train/val/test by date) (+20 more)

### Community 1 - "Cross-Elasticity & Backtest Core"
Cohesion: 0.12
Nodes (22): Backtest Script, Cross Elasticity Script, OLS Estimation per Focal SKU, top_neighbors Selection Function, trim_low Stockout Filter Function, 50% Historical Promotions Were Unprofitable Finding, Elasticity Confounders (Seasonality Promos Endogeneity Stockouts), Item2Vec Semantic Embedding of SKUs (+14 more)

### Community 2 - "Price Recommender Engine"
Cohesion: 0.16
Nodes (20): confidence_card(), estimate_canibalization(), get_promo_benchmarks(), get_sku_state(), get_substitutes(), list_skus(), _load_all(), predict_demand_curve() (+12 more)

### Community 3 - "Elasticity Diagnostic Dashboard"
Cohesion: 0.17
Nodes (17): ABAR Retail Line, Distribution of Beta (Elasticity) by SKU, CARN Retail Line, CONG Retail Line, Family-level Pooled Elasticity (CRS model), Goodness of Fit vs Elasticity Scatter Plot, FRUT Retail Line, Identification Rate by Line (% SKUs with beta < -0.05) (+9 more)

### Community 4 - "Conformal Prediction & Uncertainty"
Cohesion: 0.18
Nodes (15): Counterfactual Margin Comparison, recommend_at_cost Function, CQR Mondrian Conformal Calibration, confidence_card Function, estimate_canibalization Function, Extrapolation Penalty via Gaussian Weight, get_promo_benchmarks Function, get_sku_state Function (+7 more)

### Community 5 - "Dataset Builder & Model Comparison"
Cohesion: 0.24
Nodes (11): attach_costo Function, Build Dataset Script, Forward-Fill Cost Strategy, Panel Diario Dataset (ETL Output), strip_str Helper Function, CatBoost Quantile Model, metrics Helper Function, MLP Neural Network Model (+3 more)

### Community 6 - "EDA Price-Demand Visualization"
Cohesion: 0.31
Nodes (10): Category: SVERDURA (fresh produce / verdura), EDA Price vs Demand Time Series Chart, Inverse Price-Demand Pattern (price up, volume down), Price Time Series (Red Line, right axis), High Price Volatility Observed Across SKUs, Sales Volume (Blue Bars, left axis), SKU 006931 — Aguacate Hass KG, SKU 006941 — Cebolla Blanca KG (+2 more)

### Community 7 - "Demand Model Artifacts & Finetune"
Cohesion: 0.24
Nodes (10): Conformal Prediction Adaptive Calibration, Conformalized Quantile Regression (CQR) Mondrian, Conformal Quantiles Parquet (data/conformal_quantiles.parquet), SKU Embeddings Parquet (data/sku_embeddings.parquet), Trained LGBM Booster Model Files (models/lgbm_q{10,50,90}.txt), Hybrid Prediction Pattern (Global + Local Residual Correction), Per-SKU Local LGBM Model Files (models/local_lgbm/{sku}.txt), Local SKU Metadata Parquet (data/local_sku_metadata.parquet) (+2 more)

### Community 8 - "Basket Analysis & SKU Embeddings"
Cohesion: 0.25
Nodes (9): Build Baskets Co-occurrence Analysis, SKU Neighbors Parquet (data/sku_neighbors.parquet), SKU Pairs Parquet (data/sku_pairs.parquet), sub_score Substitutability Scoring Function, Tickets Parquet (data/tickets.parquet), Embedding Neighbors Parquet (data/embedding_neighbors.parquet), Item2Vec Gensim KeyedVectors (data/item2vec_model.kv), Item2Vec SKU Embedding Trainer (+1 more)

### Community 9 - "Quasi-Experimental Validation"
Cohesion: 0.29
Nodes (8): Credibilidad por Familia Parquet (data/credibilidad_por_familia.parquet), Backtest Quasi Events Parquet (data/backtest_quasi_events.parquet), Quasi-Experimental Backtest on Real Price Changes, Backtest Quasi Summary Parquet (data/backtest_quasi_summary.parquet), Predictions Conformal Parquet (data/predictions_conformal.parquet), boot() Cache-Resource Function, PriceIQ Streamlit Dashboard, LGBM Predictions Parquet (data/lgbm_predictions.parquet)

### Community 10 - "Cross-Elasticity V2 EB Shrinkage"
Cohesion: 0.33
Nodes (3): eb_shrink(), Elasticidad cruzada v2: top-10 vecinos + pooling jerárquico empírico-Bayesiano., Pooling parcial con piso de heterogeneidad para evitar shrinkage degenerado.

### Community 11 - "Elasticity V2 IV-OLS"
Cohesion: 0.40
Nodes (4): build_controls(), fit_pair(), Elasticidad propia v2 — mejoras sobre baseline:    1. Filtro de stockouts: trim, Devuelve dict con OLS y IV para el mismo grupo.

### Community 12 - "Market Basket Co-occurrence"
Cohesion: 0.40
Nodes (3): long_pairs(), Análisis de canastas: co-ocurrencia, lift, sustitutos y complementos.  Pipeline:, Devuelve pares en formato largo: una fila por (foco, vecino).

### Community 17 - "Backtest Recommendation Logic"
Cohesion: 0.50
Nodes (3): Backtesting: ¿qué hubiera pasado si la herramienta se hubiera usado en los últim, Recomendación rápida sin recargar artefactos., recommend_at_cost()

### Community 18 - "Log-Log OLS Baseline"
Cohesion: 0.50
Nodes (3): fit_loglog(), Baseline de elasticidad precio propia.  Modelo: log(qty_no_promo) = α + β·log(pr, Devuelve dict con β, SE, IC95, R², n para log(qty)~log(p)+dow+mes(+log_c).

## Knowledge Gaps
- **39 isolated node(s):** `allow`, `Embedding Neighbors Parquet (data/embedding_neighbors.parquet)`, `Item2Vec Gensim KeyedVectors (data/item2vec_model.kv)`, `Conformal Quantiles Parquet (data/conformal_quantiles.parquet)`, `Backtest Quasi Summary Parquet (data/backtest_quasi_summary.parquet)` (+34 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **12 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Daily Panel Parquet (data/panel_diario.parquet)` connect `Log-Log Elasticity Baseline` to `Basket Analysis & SKU Embeddings`, `Quasi-Experimental Validation`, `Demand Model Artifacts & Finetune`?**
  _High betweenness centrality (0.038) - this node is a cross-community bridge._
- **Why does `Panel Diario Dataset (ETL Output)` connect `Dataset Builder & Model Comparison` to `Cross-Elasticity & Backtest Core`, `Conformal Prediction & Uncertainty`?**
  _High betweenness centrality (0.018) - this node is a cross-community bridge._
- **What connects `Item2Vec: embeddings densos de SKU desde canastas.  Cada ticket (transacción de`, `Conformal Prediction Adaptativo — calibra los IC del LightGBM por SKU/familia.`, `Backtest quasi-experimental: validar el modelo en CAMBIOS REALES de precio.  Est` to the rest of the system?**
  _77 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Log-Log Elasticity Baseline` be split into smaller, more focused modules?**
  _Cohesion score 0.09788359788359788 - nodes in this community are weakly interconnected._
- **Should `Cross-Elasticity & Backtest Core` be split into smaller, more focused modules?**
  _Cohesion score 0.11688311688311688 - nodes in this community are weakly interconnected._