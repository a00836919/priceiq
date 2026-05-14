"""Comparación de modelos de predicción de demanda.

Modelos:
  1. LightGBM (ya entrenado, reusamos predicciones)
  2. XGBoost
  3. CatBoost
  4. MLP (Multi-Layer Perceptron, sklearn)
  5. Stacking: meta-learner Ridge sobre las predicciones de val de los 4 anteriores

Métricas comparadas en TEST:
  - RMSE log
  - MAPE qty (mediana)
  - Pinball loss q=0.5 (proper scoring rule)
  - Cobertura IC80% (para los que soportan quantile)

Salidas:
  data/model_comparison.parquet      — tabla por modelo con métricas globales
  data/model_comparison_by_sku.parquet — métricas por SKU para análisis fino
  models/{xgb,cat,mlp,stack}.{json,cbm,pkl}
"""
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import RidgeCV
import joblib
import warnings
warnings.filterwarnings("ignore")

DATA = Path("data")
MODELS = Path("models"); MODELS.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Cargar y preparar (replica lgbm_demand.py)
# ---------------------------------------------------------------------------
print("[1/7] Cargando panel + embeddings…")
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

emb = pd.read_parquet(DATA / "sku_embeddings.parquet")
emb_cols = [c for c in emb.columns if c.startswith("emb_")]

g = panel.groupby("cve_art", group_keys=False)
panel["log_qty_lag1"] = g.log_qty.shift(1)
panel["log_qty_lag7"] = g.log_qty.shift(7)
panel["log_qty_ma7"]  = g.log_qty.shift(1).rolling(7, min_periods=3).mean().reset_index(0, drop=True)
panel["log_p_lag1"]   = g.log_p.shift(1)
panel["margen_unit_lag1"] = g.margen_unit.shift(1)
panel = panel.merge(emb[["cve_art"] + emb_cols], on="cve_art", how="left")
panel[emb_cols] = panel[emb_cols].fillna(0.0)

for c in ["lin","fam","marca","tipo_articulo","cve_pro","proveedor_ult"]:
    panel[c] = panel[c].fillna("__NA__").astype(str)
    panel[c + "_le"] = LabelEncoder().fit_transform(panel[c])

features_num = ["log_p","log_c","margen_unit","margen_pct",
                "dias_desde_cambio_costo","frac_promo",
                "log_qty_lag1","log_qty_lag7","log_qty_ma7","log_p_lag1","margen_unit_lag1",
                "dow","mes","año"]
panel["woy"] = panel.fecha.dt.isocalendar().week.astype(int)
features_num.append("woy")
features_cat = ["lin_le","fam_le","marca_le","tipo_articulo_le","cve_pro_le","proveedor_ult_le"]
features = features_num + features_cat + emb_cols
panel_m = panel.dropna(subset=["log_qty_lag1","log_qty_lag7"]).copy()

fecha_max = panel_m.fecha.max()
fecha_test = fecha_max - pd.Timedelta(days=60)
fecha_val  = fecha_test - pd.Timedelta(days=30)
train = panel_m[panel_m.fecha < fecha_val]
val   = panel_m[(panel_m.fecha >= fecha_val) & (panel_m.fecha < fecha_test)]
test  = panel_m[panel_m.fecha >= fecha_test]

X_tr, y_tr = train[features].values, train.log_qty.values
X_va, y_va = val[features].values, val.log_qty.values
X_te, y_te = test[features].values, test.log_qty.values
print(f"   train={len(X_tr):,}  val={len(X_va):,}  test={len(X_te):,}  features={len(features)}")


# ---------------------------------------------------------------------------
# Helpers de evaluación
# ---------------------------------------------------------------------------
def metrics(y_true, y_pred, label=""):
    qty_true = np.exp(y_true)
    qty_pred = np.exp(y_pred)
    rmse = float(np.sqrt(np.mean((y_pred - y_true)**2)))
    mape = float(np.median(np.abs(qty_pred - qty_true) / np.maximum(qty_true, 1) * 100))
    pinball = float(np.mean(np.maximum(0.5*(y_true - y_pred), 0.5*(y_pred - y_true))))
    return {"modelo": label, "rmse_log": rmse, "mape_qty_med": mape, "pinball_q50": pinball}


# ---------------------------------------------------------------------------
# 2. Reusar predicciones de LightGBM
# ---------------------------------------------------------------------------
print("\n[2/7] Cargando predicciones de LightGBM (ya entrenado)…")
lgb_pred = pd.read_parquet(DATA / "lgbm_predictions.parquet")
# Alinear índice con test
lgb_pred = lgb_pred.set_index(["fecha","cve_art"])
test_idx = test.set_index(["fecha","cve_art"]).index
lgb_yhat_test = lgb_pred.loc[test_idx, "yhat_q50"].values
lgb_lo = lgb_pred.loc[test_idx, "yhat_q10"].values
lgb_hi = lgb_pred.loc[test_idx, "yhat_q90"].values
# Para stacking: necesitamos predicciones de VAL también — re-predict
lgb_q50 = lgb.Booster(model_file=str(MODELS / "lgbm_q50.txt"))
lgb_yhat_val = lgb_q50.predict(X_va)
print(f"   LightGBM cargado")


# ---------------------------------------------------------------------------
# 3. XGBoost (quantile)
# ---------------------------------------------------------------------------
print("\n[3/7] Entrenando XGBoost (quantile q=0.5, q=0.1, q=0.9)…")
xgb_preds_val = {}; xgb_preds_test = {}
for q in [0.1, 0.5, 0.9]:
    print(f"   q={q}…", end=" ", flush=True)
    m = xgb.XGBRegressor(
        objective="reg:quantileerror", quantile_alpha=q,
        n_estimators=2000, learning_rate=0.05,
        max_depth=8, min_child_weight=20,
        subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.0, reg_lambda=0.1,
        early_stopping_rounds=50,
        tree_method="hist", verbosity=0, seed=42,
    )
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    xgb_preds_val[q]  = m.predict(X_va)
    xgb_preds_test[q] = m.predict(X_te)
    print(f"best_iter={m.best_iteration}")
    m.save_model(str(MODELS / f"xgb_q{int(q*100):02d}.json"))


# ---------------------------------------------------------------------------
# 4. CatBoost (quantile)
# ---------------------------------------------------------------------------
print("\n[4/7] Entrenando CatBoost (quantile q=0.5)…")
# CatBoost se entrena 1 modelo por cuantil; objetivo "Quantile:alpha=0.5"
cat_preds_val = {}; cat_preds_test = {}
for q in [0.1, 0.5, 0.9]:
    print(f"   q={q}…", end=" ", flush=True)
    m = CatBoostRegressor(
        loss_function=f"Quantile:alpha={q}",
        iterations=2000, learning_rate=0.05,
        depth=8, l2_leaf_reg=3,
        early_stopping_rounds=50,
        verbose=False, random_seed=42,
    )
    m.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
    cat_preds_val[q]  = m.predict(X_va)
    cat_preds_test[q] = m.predict(X_te)
    print(f"best_iter={m.tree_count_}")
    m.save_model(str(MODELS / f"cat_q{int(q*100):02d}.cbm"))


# ---------------------------------------------------------------------------
# 5. MLP (red neuronal pequeña con sklearn) — point prediction (no quantile)
# ---------------------------------------------------------------------------
print("\n[5/7] Entrenando MLP (3 capas) con scaler…")
scaler = StandardScaler().fit(X_tr)
X_tr_s = scaler.transform(X_tr)
X_va_s = scaler.transform(X_va)
X_te_s = scaler.transform(X_te)
mlp = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32),
    activation="relu",
    solver="adam",
    alpha=1e-4,
    learning_rate_init=1e-3,
    batch_size=4096,
    max_iter=60,
    early_stopping=True, validation_fraction=0.1, n_iter_no_change=5,
    random_state=42,
    verbose=False,
)
mlp.fit(X_tr_s, y_tr)
mlp_yhat_val  = mlp.predict(X_va_s)
mlp_yhat_test = mlp.predict(X_te_s)
joblib.dump({"mlp": mlp, "scaler": scaler}, MODELS / "mlp.pkl")
print(f"   epochs={mlp.n_iter_}  loss final={mlp.loss_:.4f}")


# ---------------------------------------------------------------------------
# 6. Stacking: Ridge sobre predicciones de val
# ---------------------------------------------------------------------------
print("\n[6/7] Stacking (Ridge meta-learner sobre val)…")
P_val = np.column_stack([lgb_yhat_val, xgb_preds_val[0.5], cat_preds_val[0.5], mlp_yhat_val])
P_test = np.column_stack([lgb_yhat_test, xgb_preds_test[0.5], cat_preds_test[0.5], mlp_yhat_test])

meta = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0]).fit(P_val, y_va)
stack_yhat_test = meta.predict(P_test)
joblib.dump(meta, MODELS / "stack.pkl")
print(f"   pesos del stacking: LGB={meta.coef_[0]:.3f} | XGB={meta.coef_[1]:.3f} | "
      f"CAT={meta.coef_[2]:.3f} | MLP={meta.coef_[3]:.3f} | intercept={meta.intercept_:.3f}")


# ---------------------------------------------------------------------------
# 7. Métricas comparadas
# ---------------------------------------------------------------------------
print("\n[7/7] Métricas en TEST set…")
rows = []
for label, yhat, ylo, yhi in [
    ("LightGBM", lgb_yhat_test, lgb_lo, lgb_hi),
    ("XGBoost",  xgb_preds_test[0.5], xgb_preds_test[0.1], xgb_preds_test[0.9]),
    ("CatBoost", cat_preds_test[0.5], cat_preds_test[0.1], cat_preds_test[0.9]),
    ("MLP",      mlp_yhat_test, None, None),
    ("Stacking", stack_yhat_test, None, None),
]:
    r = metrics(y_te, yhat, label)
    if ylo is not None:
        r["cobertura_IC80"] = float(np.mean((y_te >= ylo) & (y_te <= yhi)) * 100)
    else:
        r["cobertura_IC80"] = np.nan
    rows.append(r)

# Estructural v3 también para benchmark
holdv3 = pd.read_parquet(DATA / "holdout_v3.parquet")
# Convertir RMSE log a algo comparable a nivel agregado
struct_rmse_med = holdv3.rmse_out.median()
struct_mape_med = holdv3.mape_out.median()
rows.insert(0, {
    "modelo": "Estructural v3 (Bayes jerárquico + IV)",
    "rmse_log": struct_rmse_med,
    "mape_qty_med": struct_mape_med,
    "pinball_q50": np.nan,
    "cobertura_IC80": np.nan,
})

cmp = pd.DataFrame(rows)
cmp.to_parquet(DATA / "model_comparison.parquet", index=False)

print("\n" + "="*90)
print("COMPARACIÓN DE MODELOS EN TEST SET (últimos 60 días)")
print("="*90)
print(cmp.to_string(index=False))

# Por SKU para análisis fino
print(f"\n— Por SKU: ¿qué modelo gana más frecuentemente? —")
per_sku_rows = []
for sku, g_t in test.groupby("cve_art"):
    idx = g_t.index
    if len(idx) < 5:
        continue
    pos = test.index.get_indexer(idx)
    y = y_te[pos]
    row = {"cve_art": sku, "n_test": len(idx)}
    for label, yhat in [("LightGBM", lgb_yhat_test),
                        ("XGBoost",  xgb_preds_test[0.5]),
                        ("CatBoost", cat_preds_test[0.5]),
                        ("MLP",      mlp_yhat_test),
                        ("Stacking", stack_yhat_test)]:
        qty_t = np.exp(y); qty_p = np.exp(yhat[pos])
        row[f"mape_{label}"] = float(np.median(np.abs(qty_p - qty_t) / np.maximum(qty_t,1) * 100))
        row[f"rmse_{label}"] = float(np.sqrt(np.mean((yhat[pos] - y)**2)))
    per_sku_rows.append(row)
per_sku = pd.DataFrame(per_sku_rows)
per_sku.to_parquet(DATA / "model_comparison_by_sku.parquet", index=False)

# Wins por modelo
models_compared = ["LightGBM","XGBoost","CatBoost","MLP","Stacking"]
print(f"\n— Frecuencia de wins por SKU (MAPE más bajo, n={len(per_sku):,} SKUs) —")
wins = per_sku[[f"mape_{m}" for m in models_compared]].idxmin(axis=1).str.replace("mape_","")
for m in models_compared:
    print(f"   {m:10s}: {(wins==m).sum():5d} SKUs ({(wins==m).mean()*100:5.1f}%)")

print(f"\n— Por familia: qué modelo es mejor en cada una (top 10 fam por SKUs) —")
fam_map = train.drop_duplicates("cve_art")[["cve_art","fam"]]
per_sku_f = per_sku.merge(fam_map, on="cve_art")
fam_results = per_sku_f.groupby("fam").agg(
    n_skus=("cve_art","count"),
    mape_lgbm=("mape_LightGBM","median"),
    mape_xgb=("mape_XGBoost","median"),
    mape_cat=("mape_CatBoost","median"),
    mape_mlp=("mape_MLP","median"),
    mape_stack=("mape_Stacking","median"),
).query("n_skus >= 30").sort_values("n_skus", ascending=False).head(10)
print(fam_results.to_string())
