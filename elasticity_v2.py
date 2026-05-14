"""Elasticidad propia v2 — mejoras sobre baseline:

  1. Filtro de stockouts: trim asimétrico de días con demanda anómalamente baja
  2. Splines temporales finos: semana-del-año (52 buckets) + dow
  3. IV-2SLS con log(costo) como instrumento para log(precio)
  4. Validación holdout: últimos 60 días reservados para test
  5. Comparación OLS vs IV con diagnóstico de instrumento débil

Salidas en data/:
  - elasticidades_sku_v2.parquet : OLS y IV lado a lado por SKU
  - elasticidades_fam_v2.parquet : pooled por familia (OLS + IV)
  - holdout_metrics.parquet      : RMSE/MAE/MAPE in-sample y out-of-sample
"""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.iv import IV2SLS
import warnings
warnings.filterwarnings("ignore")

DATA = Path("data")
HOLDOUT_DAYS = 60

# ---------------------------------------------------------------------------
# Cargar y preparar
# ---------------------------------------------------------------------------
panel = pd.read_parquet(DATA / "panel_diario.parquet")
panel["qty_no_promo"] = panel.qty_total - panel.qty_promo

mask = (
    panel.precio_normal_sinpromo.notna() & (panel.precio_normal_sinpromo > 0)
    & (panel.qty_no_promo > 0)
    & panel.costo_vigente.notna() & (panel.costo_vigente > 0)
)
panel = panel[mask].copy()
panel["log_qty"] = np.log(panel.qty_no_promo)
panel["log_p"] = np.log(panel.precio_normal_sinpromo)
panel["log_c"] = np.log(panel.costo_vigente)
panel["woy"] = panel.fecha.dt.isocalendar().week.astype(int)
panel["dow"] = panel.dow.astype(int)

print(f"Obs antes de stockout-trim: {len(panel):,}")

# ---------------------------------------------------------------------------
# 1. Filtro de stockouts: trim asimétrico bajo (z-score < -2 en log_qty por SKU)
# ---------------------------------------------------------------------------
def trim_low(g: pd.DataFrame) -> pd.DataFrame:
    if len(g) < 30:
        return g
    z = (g.log_qty - g.log_qty.mean()) / g.log_qty.std()
    return g[z > -2.0]  # quita cola baja sospechosa de stockout

panel = panel.groupby("cve_art", group_keys=False).apply(trim_low)
print(f"Obs después de stockout-trim: {len(panel):,}")

# ---------------------------------------------------------------------------
# Holdout temporal
# ---------------------------------------------------------------------------
fecha_max = panel.fecha.max()
fecha_corte = fecha_max - pd.Timedelta(days=HOLDOUT_DAYS)
train = panel[panel.fecha < fecha_corte].copy()
test  = panel[panel.fecha >= fecha_corte].copy()
print(f"\nHoldout: train hasta {fecha_corte.date()} ({len(train):,} obs), "
      f"test {len(test):,} obs")


# ---------------------------------------------------------------------------
# Helper: matriz de controles (dow + woy + log_c)
# ---------------------------------------------------------------------------
def build_controls(df: pd.DataFrame, with_cost: bool = True) -> pd.DataFrame:
    parts = [
        pd.get_dummies(df.dow, prefix="dow", drop_first=True).astype(float),
        pd.get_dummies(df.woy, prefix="woy", drop_first=True).astype(float),
    ]
    if with_cost:
        parts.append(df[["log_c"]].reset_index(drop=True))
    out = pd.concat([p.reset_index(drop=True) for p in parts], axis=1)
    return out


# ---------------------------------------------------------------------------
# Fit OLS y IV para un grupo
# ---------------------------------------------------------------------------
def fit_pair(g: pd.DataFrame):
    """Devuelve dict con OLS y IV para el mismo grupo."""
    if g.log_p.std() < 0.01 or g.log_p.nunique() < 3 or len(g) < 80:
        return None
    if g.log_c.std() < 0.005:
        return None  # sin variación de costo → IV no identifica

    # OLS con costo como control
    Xc = build_controls(g, with_cost=True)
    Xols = pd.concat([g[["log_p"]].reset_index(drop=True), Xc], axis=1)
    Xols = sm.add_constant(Xols)
    try:
        ols = sm.OLS(g.log_qty.values, Xols.values).fit(cov_type="HC3")
    except Exception:
        return None
    j = list(Xols.columns).index("log_p")
    ols_beta, ols_se = ols.params[j], ols.bse[j]

    # IV: log_c es instrumento, controles SIN log_c
    Xex = build_controls(g, with_cost=False)
    Xex = sm.add_constant(Xex.reset_index(drop=True))
    endog = g[["log_p"]].reset_index(drop=True)
    instr = g[["log_c"]].reset_index(drop=True)
    y = pd.Series(g.log_qty.values).reset_index(drop=True)
    try:
        iv = IV2SLS(y, Xex, endog, instr).fit(cov_type="robust")
    except Exception:
        return None
    iv_beta = float(iv.params["log_p"])
    iv_se = float(iv.std_errors["log_p"])

    # Primera etapa para F del instrumento (relevancia)
    try:
        first = sm.OLS(g.log_p.values, sm.add_constant(
            pd.concat([Xex.drop(columns="const"),
                       g[["log_c"]].reset_index(drop=True)], axis=1).values
        )).fit(cov_type="HC3")
        # F del coeficiente de log_c
        log_c_idx = first.model.exog_names.index("x" + str(first.model.exog.shape[1]-1)) \
            if "log_c" not in first.model.exog_names else first.model.exog_names.index("log_c")
        f_first = (first.params[-1] / first.bse[-1])**2
    except Exception:
        f_first = np.nan

    return {
        "n": len(g),
        "log_p_nunique": int(g.log_p.nunique()),
        "log_c_nunique": int(g.log_c.nunique()),
        "ols_beta": ols_beta, "ols_se": ols_se,
        "ols_ci_lo": ols_beta - 1.96*ols_se, "ols_ci_hi": ols_beta + 1.96*ols_se,
        "ols_pvalue": ols.pvalues[j],
        "ols_r2_adj": ols.rsquared_adj,
        "iv_beta": iv_beta, "iv_se": iv_se,
        "iv_ci_lo": iv_beta - 1.96*iv_se, "iv_ci_hi": iv_beta + 1.96*iv_se,
        "iv_pvalue": float(iv.pvalues["log_p"]),
        "iv_r2": float(iv.rsquared),
        "first_stage_F": f_first,
    }


# ---------------------------------------------------------------------------
# Fit por SKU (en TRAIN solamente para honest holdout)
# ---------------------------------------------------------------------------
print("\nEstimando OLS y IV por SKU (train only)…")
results = []
for sku, g in train.groupby("cve_art"):
    out = fit_pair(g)
    if out is None:
        continue
    out["cve_art"] = sku
    out["lin"] = g.lin.iloc[0]
    out["fam"] = g.fam.iloc[0]
    out["descripcion"] = g.descripcion.iloc[0]
    out["qty_total"] = float(g.qty_no_promo.sum())
    results.append(out)

elas = pd.DataFrame(results)
cols_first = ["cve_art","descripcion","lin","fam","n","qty_total",
              "ols_beta","ols_se","ols_ci_lo","ols_ci_hi","ols_pvalue","ols_r2_adj",
              "iv_beta","iv_se","iv_ci_lo","iv_ci_hi","iv_pvalue","iv_r2",
              "first_stage_F","log_p_nunique","log_c_nunique"]
elas = elas[cols_first].sort_values("qty_total", ascending=False).reset_index(drop=True)
elas.to_parquet(DATA / "elasticidades_sku_v2.parquet", index=False)
print(f"   → elasticidades_sku_v2.parquet: {len(elas):,} SKUs")


# ---------------------------------------------------------------------------
# Pooled por familia con FE de SKU + WoY
# ---------------------------------------------------------------------------
print("\nEstimando por familia (pooled)…")
fam_res = []
for fam_, g in train.groupby("fam"):
    if g.cve_art.nunique() < 3 or len(g) < 500 or g.log_p.std() < 0.02:
        continue
    if g.log_c.std() < 0.005:
        continue
    sku_d = pd.get_dummies(g.cve_art, prefix="sku", drop_first=True).astype(float).reset_index(drop=True)
    Xex = pd.concat([build_controls(g, with_cost=False), sku_d], axis=1)
    Xex = sm.add_constant(Xex.reset_index(drop=True))
    if Xex.shape[1] >= len(g):
        continue
    endog = g[["log_p"]].reset_index(drop=True)
    instr = g[["log_c"]].reset_index(drop=True)
    y = pd.Series(g.log_qty.values).reset_index(drop=True)
    try:
        iv = IV2SLS(y, Xex, endog, instr).fit(
            cov_type="clustered",
            clusters=g.cve_art.reset_index(drop=True).values)
    except Exception:
        continue
    # OLS comparable
    Xols = pd.concat([endog, Xex.drop(columns="const"),
                      g[["log_c"]].reset_index(drop=True)], axis=1)
    Xols = sm.add_constant(Xols)
    try:
        ols = sm.OLS(y.values, Xols.values).fit(
            cov_type="cluster", cov_kwds={"groups": g.cve_art.values})
    except Exception:
        continue
    j = list(Xols.columns).index("log_p")
    fam_res.append({
        "fam": fam_,
        "n_skus": int(g.cve_art.nunique()),
        "n_obs": int(len(g)),
        "ols_beta": ols.params[j], "ols_se": ols.bse[j], "ols_pvalue": ols.pvalues[j],
        "iv_beta": float(iv.params["log_p"]),
        "iv_se": float(iv.std_errors["log_p"]),
        "iv_pvalue": float(iv.pvalues["log_p"]),
        "iv_r2": float(iv.rsquared),
        "qty_total": float(g.qty_no_promo.sum()),
    })

fam_df = pd.DataFrame(fam_res).sort_values("qty_total", ascending=False)
fam_df["iv_ci_lo"] = fam_df.iv_beta - 1.96*fam_df.iv_se
fam_df["iv_ci_hi"] = fam_df.iv_beta + 1.96*fam_df.iv_se
fam_df["ols_ci_lo"] = fam_df.ols_beta - 1.96*fam_df.ols_se
fam_df["ols_ci_hi"] = fam_df.ols_beta + 1.96*fam_df.ols_se
fam_df.to_parquet(DATA / "elasticidades_fam_v2.parquet", index=False)
print(f"   → elasticidades_fam_v2.parquet: {len(fam_df):,} familias")


# ---------------------------------------------------------------------------
# Holdout temporal: predecir test con OLS por SKU
# ---------------------------------------------------------------------------
print("\nValidación holdout (predicción últimos 60 días)…")
metrics = []
for sku, g_train in train.groupby("cve_art"):
    g_test = test[test.cve_art == sku]
    if len(g_train) < 80 or len(g_test) < 5:
        continue
    if g_train.log_p.std() < 0.01 or g_train.log_c.std() < 0.005:
        continue
    Xtr_c = build_controls(g_train, with_cost=True)
    Xtr = pd.concat([g_train[["log_p"]].reset_index(drop=True), Xtr_c], axis=1)
    Xtr = sm.add_constant(Xtr).reset_index(drop=True)
    try:
        m = sm.OLS(g_train.log_qty.values, Xtr.values).fit()
    except Exception:
        continue
    # Construir X de test alineado a columnas de train
    Xte_c = build_controls(g_test, with_cost=True)
    Xte = pd.concat([g_test[["log_p"]].reset_index(drop=True), Xte_c], axis=1)
    Xte = sm.add_constant(Xte).reset_index(drop=True)
    Xte = Xte.reindex(columns=Xtr.columns, fill_value=0.0)
    try:
        yhat = m.predict(Xte.values)
    except Exception:
        continue
    y_true = g_test.log_qty.values
    rmse = float(np.sqrt(np.mean((yhat - y_true)**2)))
    mape_q = float(np.mean(np.abs(np.exp(yhat) - np.exp(y_true)) /
                            np.maximum(np.exp(y_true), 1)) * 100)
    # in-sample
    yhat_in = m.predict(Xtr.values)
    rmse_in = float(np.sqrt(np.mean((yhat_in - g_train.log_qty.values)**2)))
    metrics.append({
        "cve_art": sku, "n_train": len(g_train), "n_test": len(g_test),
        "rmse_log_in": rmse_in, "rmse_log_out": rmse,
        "mape_qty_out": mape_q,
    })

mt = pd.DataFrame(metrics)
mt.to_parquet(DATA / "holdout_metrics.parquet", index=False)
print(f"   → holdout_metrics.parquet: {len(mt):,} SKUs validados")


# ---------------------------------------------------------------------------
# DIAGNÓSTICO
# ---------------------------------------------------------------------------
print("\n" + "="*80)
print("DIAGNÓSTICO v2")
print("="*80)

print(f"\n— OLS vs IV por SKU (n={len(elas):,}) —")
for tag, b, p, ci_lo, ci_hi in [
    ("OLS", "ols_beta", "ols_pvalue", "ols_ci_lo", "ols_ci_hi"),
    ("IV ", "iv_beta",  "iv_pvalue",  "iv_ci_lo",  "iv_ci_hi"),
]:
    neg = (elas[b] < 0).sum()
    sig = ((elas[b] < 0) & (elas[p] < 0.05)).sum()
    sig_strict = ((elas[b] < 0) & (elas[p] < 0.05) & (elas[b] > -5) & (elas[ci_hi] < 0)).sum()
    print(f"  {tag} | β<0: {neg:5d} ({neg/len(elas)*100:5.1f}%) | "
          f"β<0 y p<0.05: {sig:5d} ({sig/len(elas)*100:5.1f}%) | "
          f"+ rango y CI excluye 0: {sig_strict:5d}")

# Instrumento débil
print(f"\n  First-stage F mediano: {elas.first_stage_F.median():.1f}")
print(f"  SKUs con F>10 (instrumento fuerte): {(elas.first_stage_F>10).sum():,} ({(elas.first_stage_F>10).mean()*100:.1f}%)")

# Diferencia OLS vs IV en SKUs bien identificados
strong = elas[(elas.first_stage_F > 10) &
              (elas.iv_beta < 0) & (elas.iv_pvalue < 0.05) &
              (elas.iv_beta > -5)]
print(f"\n  SKUs bien identificados con IV (F>10, β<0, p<0.05, β>-5): {len(strong):,}")
if len(strong) > 0:
    diff = (strong.iv_beta - strong.ols_beta)
    print(f"  Diferencia mediana IV − OLS: {diff.median():+.3f}")
    print(f"  Si IV es más negativa → OLS sub-estima magnitud de elasticidad")

print(f"\n— Por familia (top 15 por volumen, OLS vs IV) —")
print(fam_df.head(15)[["fam","n_skus","n_obs",
                       "ols_beta","ols_pvalue",
                       "iv_beta","iv_se","iv_pvalue","iv_r2"]].to_string(index=False))

print(f"\n— Holdout temporal —")
print(f"  SKUs validados: {len(mt):,}")
print(f"  RMSE in-sample (log):       mediana={mt.rmse_log_in.median():.3f}")
print(f"  RMSE out-of-sample (log):   mediana={mt.rmse_log_out.median():.3f}")
print(f"  MAPE de cantidad out-of-sample: mediana={mt.mape_qty_out.median():.1f}%")
print(f"  SKUs con RMSE_out < RMSE_in*1.5 (no overfitting severo): "
      f"{(mt.rmse_log_out < mt.rmse_log_in*1.5).sum():,} ({(mt.rmse_log_out < mt.rmse_log_in*1.5).mean()*100:.1f}%)")

print("\n— Top 10 SKUs con elasticidad IV bien identificada y holdout bueno —")
merged = strong.merge(mt, on="cve_art", how="inner")
merged = merged[merged.rmse_log_out < merged.rmse_log_in * 1.5]
top = merged.nlargest(10, "qty_total")[
    ["cve_art","descripcion","fam","iv_beta","iv_ci_lo","iv_ci_hi",
     "first_stage_F","rmse_log_out","mape_qty_out"]]
print(top.to_string(index=False))
