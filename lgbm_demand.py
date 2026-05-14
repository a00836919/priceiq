"""LightGBM para predicción de demanda con quantile regression para IC.

Features:
  - Numéricos: log_p, log_c, log(margen+1), woy, dow, mes, año, días-desde-cambio-costo
  - Lags: log_qty_lag1, log_qty_lag7, log_qty_ma7 (media móvil 7 días)
  - Embeddings: 32-dim de Item2Vec por SKU
  - Categóricos: lin, fam, marca (label-encoded)

Tres modelos LGBM independientes (quantile regression):
  - q=0.5 → predicción central (mediana)
  - q=0.1 → cuantil bajo (IC inferior)
  - q=0.9 → cuantil alto (IC superior)

Comparación contra modelo estructural v3 jerárquico en holdout temporal.

Salidas:
  models/lgbm_q{10,50,90}.txt    — modelos entrenados
  data/lgbm_predictions.parquet   — predicciones en test set
  data/lgbm_vs_structural.parquet — comparación de métricas
  data/lgbm_feature_importance.parquet
"""
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

DATA = Path("data")
MODELS = Path("models"); MODELS.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Cargar panel + embeddings + structural
# ---------------------------------------------------------------------------
print("[1/6] Cargando panel y embeddings…")
panel = pd.read_parquet(DATA / "panel_diario.parquet")
panel["qty_no_promo"] = panel.qty_total - panel.qty_promo
panel = panel[(panel.precio_normal_sinpromo.notna())
              & (panel.precio_normal_sinpromo > 0)
              & (panel.qty_no_promo > 0)
              & panel.costo_vigente.notna()
              & (panel.costo_vigente > 0)].copy()
panel["log_qty"] = np.log(panel.qty_no_promo)
panel["log_p"] = np.log(panel.precio_normal_sinpromo)
panel["log_c"] = np.log(panel.costo_vigente)
panel = panel.sort_values(["cve_art","fecha"]).reset_index(drop=True)

# Embeddings
emb = pd.read_parquet(DATA / "sku_embeddings.parquet")
emb_cols = [c for c in emb.columns if c.startswith("emb_")]
print(f"   panel: {len(panel):,} filas | embeddings: {len(emb):,} SKUs × {len(emb_cols)} dim")

# ---------------------------------------------------------------------------
# 2. Construir features con lags y embeddings
# ---------------------------------------------------------------------------
print("\n[2/6] Construyendo features (lags + embeddings + categóricos)…")
# Lag features por SKU
g = panel.groupby("cve_art", group_keys=False)
panel["log_qty_lag1"] = g.log_qty.shift(1)
panel["log_qty_lag7"] = g.log_qty.shift(7)
panel["log_qty_ma7"]  = g.log_qty.shift(1).rolling(7, min_periods=3).mean().reset_index(0, drop=True)
panel["log_p_lag1"]   = g.log_p.shift(1)
panel["margen_unit_lag1"] = g.margen_unit.shift(1)

# Adjuntar embeddings
panel = panel.merge(emb[["cve_art"] + emb_cols], on="cve_art", how="left")
print(f"   con embedding: {panel[emb_cols[0]].notna().sum():,} / {len(panel):,}")
# Rellenar embeddings ausentes con 0 (SKUs no en vocab Item2Vec)
panel[emb_cols] = panel[emb_cols].fillna(0.0)

# Categóricos label-encoded
for c in ["lin","fam","marca","tipo_articulo","cve_pro","proveedor_ult"]:
    panel[c] = panel[c].fillna("__NA__").astype(str)
    le = LabelEncoder()
    panel[c + "_le"] = le.fit_transform(panel[c])

# Lista final de features
features_num = ["log_p","log_c","margen_unit","margen_pct",
                "dias_desde_cambio_costo","frac_promo",
                "log_qty_lag1","log_qty_lag7","log_qty_ma7","log_p_lag1","margen_unit_lag1",
                "dow","mes","año"]
features_cat = ["lin_le","fam_le","marca_le","tipo_articulo_le","cve_pro_le","proveedor_ult_le"]
# woy
panel["woy"] = panel.fecha.dt.isocalendar().week.astype(int)
features_num.append("woy")

features = features_num + features_cat + emb_cols
print(f"   features totales: {len(features)} "
      f"(num={len(features_num)}, cat={len(features_cat)}, emb={len(emb_cols)})")

# Drop filas con lag NaN (primer día de cada SKU)
panel_m = panel.dropna(subset=["log_qty_lag1","log_qty_lag7"]).copy()
print(f"   filas válidas tras lags: {len(panel_m):,}")

# ---------------------------------------------------------------------------
# 3. Split temporal (train / val / test)
# ---------------------------------------------------------------------------
fecha_max = panel_m.fecha.max()
fecha_test = fecha_max - pd.Timedelta(days=60)
fecha_val  = fecha_test - pd.Timedelta(days=30)

train = panel_m[panel_m.fecha < fecha_val]
val   = panel_m[(panel_m.fecha >= fecha_val) & (panel_m.fecha < fecha_test)]
test  = panel_m[panel_m.fecha >= fecha_test]

print(f"\n[3/6] Splits temporales:")
print(f"   train: {train.fecha.min().date()} → {train.fecha.max().date()}  n={len(train):,}")
print(f"   val:   {val.fecha.min().date()} → {val.fecha.max().date()}  n={len(val):,}")
print(f"   test:  {test.fecha.min().date()} → {test.fecha.max().date()}  n={len(test):,}")

X_tr, y_tr = train[features], train.log_qty
X_va, y_va = val[features], val.log_qty
X_te, y_te = test[features], test.log_qty

# ---------------------------------------------------------------------------
# 4. Entrenar tres modelos LGBM (cuantiles 10, 50, 90)
# ---------------------------------------------------------------------------
print(f"\n[4/6] Entrenando LightGBM en 3 cuantiles…")
common_params = dict(
    n_estimators=1500,
    learning_rate=0.05,
    num_leaves=63,
    max_depth=-1,
    min_child_samples=20,
    feature_fraction=0.85,
    bagging_fraction=0.85,
    bagging_freq=5,
    reg_alpha=0.0,
    reg_lambda=0.1,
    verbose=-1,
    seed=42,
)

models = {}
for q in [0.1, 0.5, 0.9]:
    print(f"   entrenando cuantil {q}…")
    m = lgb.LGBMRegressor(
        objective="quantile", alpha=q, **common_params)
    m.fit(X_tr, y_tr,
          eval_set=[(X_va, y_va)],
          eval_metric="quantile",
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    print(f"      best iter: {m.best_iteration_}, best score: {m.best_score_}")
    models[q] = m
    m.booster_.save_model(str(MODELS / f"lgbm_q{int(q*100):02d}.txt"))

# ---------------------------------------------------------------------------
# 5. Predicciones y métricas
# ---------------------------------------------------------------------------
print(f"\n[5/6] Predicciones en TEST y métricas…")
pred = test[["fecha","cve_art","descripcion","fam","lin","log_qty","qty_no_promo",
             "log_p","log_c","margen_unit"]].copy()
pred["yhat_q10"] = models[0.1].predict(X_te)
pred["yhat_q50"] = models[0.5].predict(X_te)
pred["yhat_q90"] = models[0.9].predict(X_te)
pred["qty_hat"]  = np.exp(pred.yhat_q50)
pred["qty_hat_lo"] = np.exp(pred.yhat_q10)
pred["qty_hat_hi"] = np.exp(pred.yhat_q90)
pred["residual_log"] = pred.yhat_q50 - pred.log_qty
pred["in_ci"] = (pred.log_qty >= pred.yhat_q10) & (pred.log_qty <= pred.yhat_q90)
pred["mape_qty"] = np.abs(pred.qty_hat - pred.qty_no_promo) / np.maximum(pred.qty_no_promo, 1) * 100

pred.to_parquet(DATA / "lgbm_predictions.parquet", index=False)

rmse = np.sqrt(np.mean(pred.residual_log**2))
mape = pred.mape_qty.median()
cov  = pred.in_ci.mean() * 100  # cobertura del IC (target: ~80% para q10-q90)
print(f"   RMSE_log (test): {rmse:.3f}")
print(f"   MAPE_qty median: {mape:.1f}%")
print(f"   Cobertura IC80%: {cov:.1f}% (target: ~80%)")

# ---------------------------------------------------------------------------
# Comparación con modelo estructural v3 jerárquico
# ---------------------------------------------------------------------------
holdv3 = pd.read_parquet(DATA / "holdout_v3.parquet")
# v3 RMSE/MAPE están a nivel SKU sobre últimos 60 días — agregamos LGBM al mismo nivel
lgb_by_sku = pred.groupby("cve_art").agg(
    rmse_lgb=("residual_log", lambda x: np.sqrt(np.mean(x**2))),
    mape_lgb=("mape_qty", "median"),
    cov_lgb=("in_ci", "mean"),
    n_test=("log_qty", "size"),
).reset_index()

cmp = holdv3[["cve_art","n_test","rmse_out","mape_out"]].rename(
    columns={"rmse_out":"rmse_struct","mape_out":"mape_struct"})
cmp = cmp.merge(lgb_by_sku, on="cve_art", how="inner")
cmp.to_parquet(DATA / "lgbm_vs_structural.parquet", index=False)

print(f"\n— COMPARACIÓN LightGBM vs Estructural v3 (mismos {len(cmp):,} SKUs en test) —")
print(f"  RMSE log mediano   estructural: {cmp.rmse_struct.median():.3f}")
print(f"  RMSE log mediano   LightGBM:    {cmp.rmse_lgb.median():.3f}")
print(f"  MAPE qty mediano   estructural: {cmp.mape_struct.median():.1f}%")
print(f"  MAPE qty mediano   LightGBM:    {cmp.mape_lgb.median():.1f}%")
print(f"  SKUs donde LightGBM gana RMSE: {(cmp.rmse_lgb < cmp.rmse_struct).sum():,} "
      f"({(cmp.rmse_lgb < cmp.rmse_struct).mean()*100:.1f}%)")
print(f"  SKUs donde LightGBM gana MAPE: {(cmp.mape_lgb < cmp.mape_struct).sum():,} "
      f"({(cmp.mape_lgb < cmp.mape_struct).mean()*100:.1f}%)")

# ---------------------------------------------------------------------------
# 6. Feature importance
# ---------------------------------------------------------------------------
print(f"\n[6/6] Feature importance (modelo q=0.5)…")
m50 = models[0.5]
fi = pd.DataFrame({"feature": features, "gain": m50.booster_.feature_importance(importance_type="gain")})
fi = fi.sort_values("gain", ascending=False).reset_index(drop=True)
# Agrupar embeddings
fi["grupo"] = fi.feature.apply(lambda x: "embedding" if x.startswith("emb_")
                                else "lag" if "lag" in x or "ma7" in x
                                else "categ" if x.endswith("_le")
                                else "precio_costo" if x in ["log_p","log_c","margen_unit","margen_pct"]
                                else "calendario" if x in ["dow","mes","año","woy","dias_desde_cambio_costo","frac_promo"]
                                else "otro")
fi_grp = fi.groupby("grupo").gain.sum().sort_values(ascending=False)
fi_grp_pct = (fi_grp / fi_grp.sum() * 100).round(1)
fi.to_parquet(DATA / "lgbm_feature_importance.parquet", index=False)

print(f"\n— Importancia por GRUPO de feature (% del gain total) —")
print(fi_grp_pct.to_string())

print(f"\n— TOP 20 features individuales —")
print(fi.head(20).to_string(index=False))
