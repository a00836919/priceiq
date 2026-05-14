"""Fine-tuning per-SKU: corrección residual sobre el modelo global LightGBM.

Idea:
  yhat_final(sku, t) = yhat_global(t) + yhat_local_sku(residual_t)

Para cada SKU top:
  1. Predecimos con LGBM global → residuales en train y val
  2. Entrenamos un LGBM CHICO sobre esos residuales con sus propias features
  3. Validamos: si MAPE_local en val < MAPE_global en val → usamos local
                 si no → descartamos (fallback al global)
  4. Final = global + local (cuando aplica)

Esto evita overfitting porque el local sólo aprende lo que el global no capturó.

Salidas:
  models/local_lgbm/{sku}.txt        — modelos locales (uno por SKU top que mejora)
  data/local_sku_metadata.parquet    — tabla: cve_art, n_obs, mape_val_global, mape_val_local, usa_local
  data/predictions_hybrid.parquet    — predicciones híbridas (global + local) en TEST
"""
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

DATA = Path("data")
MODELS = Path("models")
LOCAL_DIR = MODELS / "local_lgbm"; LOCAL_DIR.mkdir(parents=True, exist_ok=True)

MIN_OBS_TRAIN = 400      # mínimo de obs en train para entrenar local
MIN_OBS_VAL   = 15       # mínimo de obs en val para validar
TOP_N_BY_VOL  = 500      # candidatos: top-N SKUs por volumen

# ---------------------------------------------------------------------------
# 1. Cargar y replicar features (igual que lgbm_demand.py)
# ---------------------------------------------------------------------------
print("[1/5] Reconstruyendo features…")
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
train = panel_m[panel_m.fecha < fecha_val].copy()
val   = panel_m[(panel_m.fecha >= fecha_val) & (panel_m.fecha < fecha_test)].copy()
test  = panel_m[panel_m.fecha >= fecha_test].copy()

# ---------------------------------------------------------------------------
# 2. Predicciones del modelo global (LightGBM q=0.5)
# ---------------------------------------------------------------------------
print("\n[2/5] Predicciones del LightGBM global en train/val/test…")
lgb_global = lgb.Booster(model_file=str(MODELS / "lgbm_q50.txt"))
train["yhat_global"] = lgb_global.predict(train[features].values)
val["yhat_global"]   = lgb_global.predict(val[features].values)
test["yhat_global"]  = lgb_global.predict(test[features].values)

# Residuales
train["residual"] = train.log_qty - train.yhat_global
val["residual"]   = val.log_qty - val.yhat_global

# ---------------------------------------------------------------------------
# 3. Seleccionar SKUs candidatos: top-N por volumen en train + min obs
# ---------------------------------------------------------------------------
print("\n[3/5] Seleccionando SKUs candidatos para fine-tuning…")
sku_stats = train.groupby("cve_art").agg(
    n_train=("log_qty","size"),
    qty_total=("qty_no_promo","sum"),
).reset_index()

# Cruce con val para asegurar evaluación
val_n = val.groupby("cve_art").size().rename("n_val").reset_index()
sku_stats = sku_stats.merge(val_n, on="cve_art", how="left").fillna({"n_val":0})

candidatos = sku_stats[(sku_stats.n_train >= MIN_OBS_TRAIN)
                      & (sku_stats.n_val >= MIN_OBS_VAL)].nlargest(TOP_N_BY_VOL, "qty_total")
print(f"   candidatos: {len(candidatos):,} SKUs (filtros: n_train≥{MIN_OBS_TRAIN}, "
      f"n_val≥{MIN_OBS_VAL}, top-{TOP_N_BY_VOL} por volumen)")

# ---------------------------------------------------------------------------
# 4. Entrenar modelo local por cada candidato (LGBM chico sobre residual)
# ---------------------------------------------------------------------------
print(f"\n[4/5] Entrenando LGBM local por SKU (sobre residuales)…")
local_params = dict(
    objective="regression",
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=15,
    max_depth=5,
    min_child_samples=10,
    feature_fraction=0.8,
    reg_lambda=0.5,
    verbose=-1,
    seed=42,
)

results = []
n_used = 0
for k, (_, row) in enumerate(candidatos.iterrows(), 1):
    sku = row.cve_art
    g_tr = train[train.cve_art == sku]
    g_va = val[val.cve_art == sku]
    if len(g_tr) < MIN_OBS_TRAIN or len(g_va) < MIN_OBS_VAL:
        continue

    Xtr = g_tr[features].values
    ytr = g_tr.residual.values
    Xva = g_va[features].values
    yva_resid = g_va.residual.values

    m = lgb.LGBMRegressor(**local_params)
    m.fit(Xtr, ytr,
          eval_set=[(Xva, yva_resid)],
          callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])

    # Evaluar: ¿el local mejora MAPE en val?
    pred_local_val = m.predict(Xva)
    yhat_global_val = g_va.yhat_global.values
    yhat_hybrid_val = yhat_global_val + pred_local_val
    y_val = g_va.log_qty.values
    qty_val = np.exp(y_val)
    mape_global = float(np.median(np.abs(np.exp(yhat_global_val) - qty_val) / np.maximum(qty_val,1)*100))
    mape_hybrid = float(np.median(np.abs(np.exp(yhat_hybrid_val) - qty_val) / np.maximum(qty_val,1)*100))
    usa_local = mape_hybrid < mape_global  # criterio: solo si mejora

    if usa_local:
        n_used += 1
        m.booster_.save_model(str(LOCAL_DIR / f"{sku}.txt"))

    results.append({
        "cve_art": sku,
        "n_train": int(len(g_tr)),
        "n_val": int(len(g_va)),
        "qty_total_train": float(g_tr.qty_no_promo.sum()),
        "mape_val_global": mape_global,
        "mape_val_hybrid": mape_hybrid,
        "delta_mape": mape_hybrid - mape_global,
        "usa_local": bool(usa_local),
    })

    if k % 50 == 0:
        print(f"   [{k}/{len(candidatos)}]  con local activo: {n_used}")

meta = pd.DataFrame(results)
meta.to_parquet(DATA / "local_sku_metadata.parquet", index=False)
print(f"\n   Total candidatos evaluados: {len(meta):,}")
print(f"   Con local activo (mejora val): {n_used:,} ({n_used/len(meta)*100:.1f}%)")
print(f"   Mediana de Δ MAPE en val (negativo = mejora): {meta.delta_mape.median():+.2f}pp")
print(f"   Mediana de Δ MAPE en SKUs que usan local:     "
      f"{meta[meta.usa_local].delta_mape.median():+.2f}pp")

# ---------------------------------------------------------------------------
# 5. Predicciones híbridas en TEST
# ---------------------------------------------------------------------------
print(f"\n[5/5] Predicciones híbridas en TEST set…")
test["yhat_hybrid"] = test.yhat_global.copy()  # default = global
locals_activos = set(meta[meta.usa_local].cve_art)
for sku in locals_activos:
    g_te = test[test.cve_art == sku]
    if len(g_te) == 0:
        continue
    m = lgb.Booster(model_file=str(LOCAL_DIR / f"{sku}.txt"))
    delta = m.predict(g_te[features].values)
    test.loc[g_te.index, "yhat_hybrid"] = g_te.yhat_global.values + delta

# Métricas
test["qty_no_promo_exp"] = np.exp(test.log_qty)
test["qty_hat_global"] = np.exp(test.yhat_global)
test["qty_hat_hybrid"] = np.exp(test.yhat_hybrid)
test["mape_global"] = np.abs(test.qty_hat_global - test.qty_no_promo_exp) / np.maximum(test.qty_no_promo_exp,1)*100
test["mape_hybrid"] = np.abs(test.qty_hat_hybrid - test.qty_no_promo_exp) / np.maximum(test.qty_no_promo_exp,1)*100

test[["fecha","cve_art","descripcion","fam","log_qty","qty_no_promo",
      "yhat_global","yhat_hybrid","qty_hat_global","qty_hat_hybrid",
      "mape_global","mape_hybrid"]].to_parquet(DATA / "predictions_hybrid.parquet", index=False)

# ---------------------------------------------------------------------------
# DIAGNÓSTICO
# ---------------------------------------------------------------------------
print("\n" + "="*80, "\nDIAGNÓSTICO PER-SKU FINE-TUNING\n", "="*80)

print(f"\n— Métricas globales en TEST (mismas filas, comparación pareada) —")
all_skus = test.cve_art.unique()
locals_in_test = test[test.cve_art.isin(locals_activos)]
no_locals     = test[~test.cve_art.isin(locals_activos)]

def med_mape(df, col):
    if len(df) == 0: return np.nan
    return float(df[col].median())

print(f"  GLOBAL  (todos los SKUs):                       MAPE mediano = {med_mape(test,'mape_global'):.1f}%")
print(f"  HÍBRIDO (todos los SKUs):                       MAPE mediano = {med_mape(test,'mape_hybrid'):.1f}%")
print(f"\n  Sólo SKUs CON local activo ({len(locals_activos)} SKUs, {len(locals_in_test):,} obs):")
print(f"    Global :   {med_mape(locals_in_test,'mape_global'):.1f}%")
print(f"    Híbrido:   {med_mape(locals_in_test,'mape_hybrid'):.1f}%")
print(f"    Mejora:    {med_mape(locals_in_test,'mape_global') - med_mape(locals_in_test,'mape_hybrid'):+.1f} pp")
print(f"\n  SKUs SIN local (resto, {test.cve_art.nunique()-len(locals_activos)} SKUs):")
print(f"    Global = Híbrido = {med_mape(no_locals,'mape_global'):.1f}%")

# Top 15 SKUs donde más ayuda el local
print(f"\n— TOP 15 SKUs con MAYOR mejora del fine-tuning (en TEST) —")
sku_mape_test = test.groupby("cve_art").agg(
    n=("log_qty","size"),
    mape_g=("mape_global","median"),
    mape_h=("mape_hybrid","median"),
    desc=("descripcion","first"),
    fam=("fam","first"),
    vol_test=("qty_no_promo","sum"),
).reset_index()
sku_mape_test["delta"] = sku_mape_test.mape_h - sku_mape_test.mape_g
sku_mape_test["usa_local"] = sku_mape_test.cve_art.isin(locals_activos)
ayudados = sku_mape_test[sku_mape_test.usa_local].nsmallest(15, "delta")
print(ayudados[["cve_art","desc","fam","n","vol_test","mape_g","mape_h","delta"]].to_string(index=False))

# Top SKUs donde EMPEORA (sanity check)
print(f"\n— SKUs donde el local EMPEORA en test (sanity check) —")
empeoran = sku_mape_test[sku_mape_test.usa_local & (sku_mape_test.delta > 5)]
print(f"   {len(empeoran)} SKUs donde delta>5pp (esperable algo por ruido en val→test)")

# Volumen cubierto
total_qty = test.qty_no_promo.sum()
qty_with_local = locals_in_test.qty_no_promo.sum()
print(f"\n— Cobertura del fine-tuning por volumen —")
print(f"   Volumen TOTAL en test: {total_qty:,.0f} unidades")
print(f"   Volumen cubierto por modelos locales: {qty_with_local:,.0f} ({qty_with_local/total_qty*100:.1f}%)")
