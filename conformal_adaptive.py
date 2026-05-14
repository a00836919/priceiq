"""Conformal Prediction Adaptativo — calibra los IC del LightGBM por SKU/familia.

Método: Conformalized Quantile Regression (CQR) Mondrian.

Paso a paso:
  1. En VAL: para cada observación, calcular score = max(qhi - y, y - qlo, 0)
              (qhi=yhat_q90, qlo=yhat_q10).
              El score mide cuánto "se pasó" el IC original.
  2. Agrupar por SKU (si tiene ≥30 obs en val) o por FAMILIA (fallback):
       q_g = percentil-α de los scores del grupo
       (α=0.8 para IC80%; en realidad usamos (1-α)·(1+1/n_cal) para ajuste finito)
  3. IC calibrado en TEST:
       lo' = qlo - q_g
       hi' = qhi + q_g
  4. Validar: cobertura empírica en test ≥ 80% por SKU (objetivo).

Garantía teórica (intercambiabilidad): cobertura marginal ≥ α en cada grupo.

Salidas:
  data/conformal_quantiles.parquet    — q_g por SKU y por familia
  data/predictions_conformal.parquet  — IC original e IC calibrado en test
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
TARGET_LEVEL = 0.80    # IC80%: cubrir 80% de obs
MIN_OBS_SKU = 30        # Mínimo en val para calibrar por SKU
MIN_OBS_FAM = 100       # Mínimo en val para calibrar por familia

# ---------------------------------------------------------------------------
# 1. Reconstruir features (igual que antes)
# ---------------------------------------------------------------------------
print("[1/4] Reconstruyendo features y splits…")
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
val   = panel_m[(panel_m.fecha >= fecha_val) & (panel_m.fecha < fecha_test)].copy()
test  = panel_m[panel_m.fecha >= fecha_test].copy()

# ---------------------------------------------------------------------------
# 2. Predicciones de cuantiles q10/q90 (modelos LGBM)
# ---------------------------------------------------------------------------
print("\n[2/4] Cargando LGBM q10/q90 y prediciendo en val/test…")
lgb_q10 = lgb.Booster(model_file=str(MODELS / "lgbm_q10.txt"))
lgb_q90 = lgb.Booster(model_file=str(MODELS / "lgbm_q90.txt"))
val["q10"] = lgb_q10.predict(val[features].values)
val["q90"] = lgb_q90.predict(val[features].values)
test["q10"] = lgb_q10.predict(test[features].values)
test["q90"] = lgb_q90.predict(test[features].values)

# ---------------------------------------------------------------------------
# 3. Calcular conformity scores y cuantiles por grupo
# ---------------------------------------------------------------------------
print("\n[3/4] Calculando scores de conformidad y cuantiles por SKU/familia…")
val["score"] = np.maximum(val.q10 - val.log_qty, val.log_qty - val.q90).clip(lower=0)
# Nota: si score=0 significa que la obs YA estaba dentro del IC original.
# El cuantil α del score nos dice cuánto AGRANDAR el IC.

# Por SKU (con ≥30 obs)
sku_n = val.groupby("cve_art").size()
skus_calibr = sku_n[sku_n >= MIN_OBS_SKU].index.tolist()
quant_sku = {}
for sku in skus_calibr:
    s = val[val.cve_art == sku].score
    # Ajuste de muestra finita Vovk: nivel = ceil((n+1)·α) / n
    n = len(s)
    level = np.minimum(np.ceil((n+1)*TARGET_LEVEL)/n, 1.0)
    quant_sku[sku] = float(s.quantile(level))

# Por familia (todos los SKUs juntos)
quant_fam = {}
for fam, sub in val.groupby("fam"):
    if len(sub) < MIN_OBS_FAM:
        continue
    n = len(sub)
    level = np.minimum(np.ceil((n+1)*TARGET_LEVEL)/n, 1.0)
    quant_fam[fam] = float(sub.score.quantile(level))

# Fallback global
n_glob = len(val)
level_glob = np.minimum(np.ceil((n_glob+1)*TARGET_LEVEL)/n_glob, 1.0)
quant_global = float(val.score.quantile(level_glob))

print(f"   cuantiles calibrados por SKU:     {len(quant_sku):,}")
print(f"   cuantiles calibrados por familia: {len(quant_fam):,}")
print(f"   global fallback:                  {quant_global:.4f}")

# Tabla de cuantiles
qtab = pd.DataFrame([{"nivel":"sku","clave":k,"q":v} for k,v in quant_sku.items()]
                 + [{"nivel":"fam","clave":k,"q":v} for k,v in quant_fam.items()]
                 + [{"nivel":"global","clave":"_global","q":quant_global}])
qtab.to_parquet(DATA / "conformal_quantiles.parquet", index=False)

# ---------------------------------------------------------------------------
# 4. Aplicar cuantiles calibrados en test
# ---------------------------------------------------------------------------
print("\n[4/4] Aplicando IC calibrados en TEST…")
def get_quantile(row):
    if row.cve_art in quant_sku: return quant_sku[row.cve_art]
    if row.fam in quant_fam: return quant_fam[row.fam]
    return quant_global

test["q_adj"] = test.apply(get_quantile, axis=1)
test["q10_calib"] = test.q10 - test.q_adj
test["q90_calib"] = test.q90 + test.q_adj

# Cobertura original vs calibrada
test["in_ci_orig"]   = (test.log_qty >= test.q10)        & (test.log_qty <= test.q90)
test["in_ci_calib"]  = (test.log_qty >= test.q10_calib)  & (test.log_qty <= test.q90_calib)
test["ancho_orig"]   = test.q90 - test.q10
test["ancho_calib"]  = test.q90_calib - test.q10_calib

# Exportar
out = test[["fecha","cve_art","descripcion","fam","log_qty","qty_no_promo",
            "q10","q90","q10_calib","q90_calib","q_adj",
            "in_ci_orig","in_ci_calib","ancho_orig","ancho_calib"]]
out.to_parquet(DATA / "predictions_conformal.parquet", index=False)

# ---------------------------------------------------------------------------
# DIAGNÓSTICO
# ---------------------------------------------------------------------------
print("\n" + "="*80, "\nDIAGNÓSTICO CONFORMAL ADAPTATIVO\n", "="*80)

print(f"\n— Cobertura del IC en TEST (objetivo: {TARGET_LEVEL*100:.0f}%) —")
print(f"  Cobertura GLOBAL:")
print(f"    Original (LGBM q10/q90):   {test.in_ci_orig.mean()*100:.1f}%")
print(f"    Calibrado (CQR Mondrian):  {test.in_ci_calib.mean()*100:.1f}%")
print(f"\n  Ancho mediano del IC (en log):")
print(f"    Original:                  {test.ancho_orig.median():.3f}")
print(f"    Calibrado:                 {test.ancho_calib.median():.3f}")

# Cobertura por SKU: distribución
sku_cov = test.groupby("cve_art").agg(
    n=("log_qty","size"),
    cov_orig=("in_ci_orig","mean"),
    cov_calib=("in_ci_calib","mean"),
).query("n >= 10")

print(f"\n— Cobertura por SKU (n≥10 obs en test) —")
print(f"  SKUs analizados: {len(sku_cov):,}")
print(f"  Cobertura ORIGINAL:")
print(f"    mediana: {sku_cov.cov_orig.median()*100:.1f}%   "
      f"% con cov<70%: {(sku_cov.cov_orig<0.70).mean()*100:.1f}%   "
      f"% con cov>90%: {(sku_cov.cov_orig>0.90).mean()*100:.1f}%")
print(f"  Cobertura CALIBRADA:")
print(f"    mediana: {sku_cov.cov_calib.median()*100:.1f}%   "
      f"% con cov<70%: {(sku_cov.cov_calib<0.70).mean()*100:.1f}%   "
      f"% con cov>90%: {(sku_cov.cov_calib>0.90).mean()*100:.1f}%")

# Histograma simple en stdout
def hist_text(s, n_bins=12, lo=0.0, hi=1.0):
    bins = np.linspace(lo, hi, n_bins+1)
    counts, _ = np.histogram(s.clip(lo,hi), bins=bins)
    width = max(counts)
    for i in range(n_bins):
        bar = "█" * int(40 * counts[i] / width)
        print(f"   {bins[i]*100:5.1f}-{bins[i+1]*100:5.1f}% │ {bar} {counts[i]}")

print(f"\n— Distribución de cobertura por SKU (original) —")
hist_text(sku_cov.cov_orig)
print(f"\n— Distribución de cobertura por SKU (calibrada) —")
hist_text(sku_cov.cov_calib)

# Ejemplos
print(f"\n— TOP 10 SKUs con peor cobertura ORIGINAL, comparada vs calibrada —")
worst = sku_cov.nsmallest(10, "cov_orig").reset_index()
meta = test.drop_duplicates("cve_art")[["cve_art","descripcion","fam"]]
worst = worst.merge(meta, on="cve_art")
print(worst[["cve_art","descripcion","fam","n","cov_orig","cov_calib"]].to_string(index=False))
