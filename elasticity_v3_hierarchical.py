"""Elasticidad propia v3 — modelo jerárquico (Frequentist Bayes via Mixed Effects).

Para cada FAMILIA fitteamos un modelo de efectos mixtos:
    log(qty_it) = α_fam + β_fam · log(p_it) + δ · log(c_it) + γ_dow + γ_mes
                  + (a_i + b_i · log(p_it))             ← efectos aleatorios por SKU
                  + ε

Donde a_i, b_i ~ Normal(0, Σ).  El β del SKU i es:  β_i = β_fam + b_i  (BLUP).

Esto implementa "partial pooling": SKUs con poca data heredan β_fam (con poca b_i propia),
SKUs con mucha data se acercan a su elasticidad individual. Resuelve el agujero de los
~6,000 SKUs que individualmente no se identificaban.

Salidas en data/:
  - elasticidades_jerarquico_sku.parquet   — β_i, IC95, n por SKU (con shrinkage)
  - elasticidades_jerarquico_fam.parquet   — β_fam, σ_β, n por familia
  - holdout_v3.parquet                     — RMSE/MAPE en últimos 60 días
"""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
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
panel["dow"] = panel.dow.astype(int)
panel["mes"] = panel.mes.astype(int)

# Stockout trim asimétrico (igual que v2)
def trim_low(g):
    if len(g) < 30: return g
    z = (g.log_qty - g.log_qty.mean()) / g.log_qty.std()
    return g[z > -2.0]
panel = panel.groupby("cve_art", group_keys=False).apply(trim_low)

# Holdout temporal
fecha_max = panel.fecha.max()
corte = fecha_max - pd.Timedelta(days=HOLDOUT_DAYS)
train = panel[panel.fecha < corte].copy()
test  = panel[panel.fecha >= corte].copy()
print(f"Obs train: {len(train):,}, test: {len(test):,}")

# ---------------------------------------------------------------------------
# Centrado de log_p por SKU para ayudar al optimizador
# ---------------------------------------------------------------------------
sku_p_mean = train.groupby("cve_art").log_p.mean().rename("log_p_mu")
train = train.join(sku_p_mean, on="cve_art")
test  = test.join(sku_p_mean, on="cve_art")
train["log_p_c"] = train.log_p - train.log_p_mu
test["log_p_c"]  = test.log_p  - test.log_p_mu

# ---------------------------------------------------------------------------
# Fit por familia
# ---------------------------------------------------------------------------
print("\nFitteando MixedLM por familia (random intercept + slope sobre log_p)…")

fam_results = []     # filas por familia
sku_results = []     # filas por SKU con β shrunken

# filtramos familias con suficiente data
fam_sizes = train.groupby("fam").agg(n=("log_qty","count"), nsku=("cve_art","nunique"))
familias = fam_sizes[(fam_sizes.n >= 500) & (fam_sizes.nsku >= 3)].index.tolist()
print(f"  Familias a procesar: {len(familias)}")

for k, fam in enumerate(familias, 1):
    g = train[train.fam == fam].copy()
    if g.log_p.std() < 0.02 or g.log_c.std() < 0.005:
        continue

    formula = "log_qty ~ log_p_c + log_c + C(dow) + C(mes)"
    try:
        # random intercept + random slope sobre log_p_c por SKU
        md = smf.mixedlm(formula, g, groups=g.cve_art,
                         re_formula="~log_p_c")
        mdf = md.fit(method="lbfgs", reml=True, maxiter=200)
    except Exception as e:
        # backoff: sólo random intercept
        try:
            md = smf.mixedlm(formula, g, groups=g.cve_art)
            mdf = md.fit(method="lbfgs", reml=True, maxiter=200)
            random_slope = False
        except Exception:
            continue
    else:
        random_slope = True

    if not mdf.converged:
        continue

    # β fijo de familia
    beta_fam = float(mdf.fe_params.get("log_p_c", np.nan))
    se_fam = float(mdf.bse_fe.get("log_p_c", np.nan))
    sigma_b = np.nan
    if random_slope:
        try:
            cov = mdf.cov_re
            # cov es 2x2: [[σ²_a, σ_ab],[σ_ab, σ²_b]] con orden (Intercept, log_p_c)
            sigma_b = float(np.sqrt(cov.loc["log_p_c","log_p_c"]))
        except Exception:
            sigma_b = np.nan

    fam_results.append({
        "fam": fam,
        "n_obs": int(len(g)),
        "n_skus": int(g.cve_art.nunique()),
        "beta_fam": beta_fam,
        "se_fam": se_fam,
        "ci_lo_fam": beta_fam - 1.96*se_fam,
        "ci_hi_fam": beta_fam + 1.96*se_fam,
        "sigma_between_sku": sigma_b,   # heterogeneidad de β entre SKUs
        "random_slope": random_slope,
    })

    # β por SKU = β_fam + BLUP del random slope
    if random_slope:
        re = mdf.random_effects  # dict: sku -> Series con [Intercept, log_p_c]
        for sku, s in re.items():
            b_i = float(s.get("log_p_c", 0.0))
            beta_i = beta_fam + b_i
            n_i = int((g.cve_art == sku).sum())
            sku_results.append({
                "cve_art": sku, "fam": fam,
                "beta_i": beta_i,
                "blup_slope": b_i,
                "beta_fam": beta_fam,
                "n_train": n_i,
                "shrinkage": abs(b_i)/(abs(b_i)+1e-9),  # placeholder, mejor dabajo
            })
    else:
        for sku in g.cve_art.unique():
            sku_results.append({
                "cve_art": sku, "fam": fam,
                "beta_i": beta_fam,
                "blup_slope": 0.0,
                "beta_fam": beta_fam,
                "n_train": int((g.cve_art == sku).sum()),
                "shrinkage": 1.0,
            })

    if k % 20 == 0:
        print(f"  [{k}/{len(familias)}] {fam} | β_fam={beta_fam:+.3f} | "
              f"σ_b={sigma_b:.3f} | n_skus={g.cve_art.nunique()}")

fam_df = pd.DataFrame(fam_results).sort_values("n_obs", ascending=False)
sku_df = pd.DataFrame(sku_results)

# Adjuntar metadata SKU
meta = panel.groupby("cve_art").agg(
    descripcion=("descripcion","first"),
    lin=("lin","first"),
    qty_total=("qty_no_promo","sum"),
).reset_index()
sku_df = sku_df.merge(meta, on="cve_art", how="left")

# Re-estimar IC del β_i con un proxy: combinar varianza fija + var aleatoria
sku_df = sku_df.merge(fam_df[["fam","se_fam","sigma_between_sku","beta_fam"]].rename(
    columns={"beta_fam":"_b_fam"}), on="fam", how="left", suffixes=("","_y"))
# Aproximación: var(β_i) ≈ se_fam² + (σ_between² · weight) — donde weight refleja shrinkage
# Para reportar, asumimos var(β_i) ≈ se_fam² + cov_blup, pero como cov_blup no es trivial,
# usamos un IC efectivo: si BLUP ~0 (shrunken), IC ≈ IC del β_fam.
# Si BLUP grande, β_i es más cerca del individual y el IC se ensancha por σ_between.
sku_df["beta_se_eff"] = np.sqrt(sku_df.se_fam**2
                                + (sku_df.sigma_between_sku.fillna(0)**2)
                                  * (sku_df.blup_slope.abs() /
                                     (sku_df.sigma_between_sku.fillna(1e-9) + 1e-9)).clip(0,1)**2)
sku_df["ci_lo"] = sku_df.beta_i - 1.96 * sku_df.beta_se_eff
sku_df["ci_hi"] = sku_df.beta_i + 1.96 * sku_df.beta_se_eff

sku_df = sku_df[["cve_art","descripcion","fam","lin","n_train","qty_total",
                 "beta_i","beta_fam","blup_slope","beta_se_eff","ci_lo","ci_hi"]]
sku_df = sku_df.sort_values("qty_total", ascending=False).reset_index(drop=True)

fam_df.to_parquet(DATA / "elasticidades_jerarquico_fam.parquet", index=False)
sku_df.to_parquet(DATA / "elasticidades_jerarquico_sku.parquet", index=False)

print(f"\n→ familia: {len(fam_df):,} | sku: {len(sku_df):,}")


# ---------------------------------------------------------------------------
# Validación holdout: predecir test con β_i jerárquico
# ---------------------------------------------------------------------------
print("\nValidación holdout (predicción últimos 60 días con jerárquico)…")

# Para predecir necesitamos también α_i, controles temporales y log_c.
# Reusamos los dataframes y un OLS por SKU PERO ANCLADO al β_i jerárquico:
# fijamos β_i y estimamos el resto (α + intercepto temporal) por SKU con OLS.
sku_beta_map = sku_df.set_index("cve_art").beta_i.to_dict()

metrics = []
for sku, g_tr in train.groupby("cve_art"):
    g_te = test[test.cve_art == sku]
    if len(g_te) < 5 or len(g_tr) < 30:
        continue
    if sku not in sku_beta_map:
        continue
    bi = sku_beta_map[sku]
    # y* = log_qty - β_i * log_p   →  ajustamos el resto con OLS
    y_star = g_tr.log_qty.values - bi * g_tr.log_p_c.values
    X_tr = pd.concat([
        g_tr[["log_c"]].reset_index(drop=True),
        pd.get_dummies(g_tr.dow, prefix="dow", drop_first=True).astype(float).reset_index(drop=True),
        pd.get_dummies(g_tr.mes, prefix="mes", drop_first=True).astype(float).reset_index(drop=True),
    ], axis=1)
    X_tr = sm.add_constant(X_tr)
    try:
        m = sm.OLS(y_star, X_tr.values).fit()
    except Exception:
        continue

    X_te = pd.concat([
        g_te[["log_c"]].reset_index(drop=True),
        pd.get_dummies(g_te.dow, prefix="dow", drop_first=True).astype(float).reset_index(drop=True),
        pd.get_dummies(g_te.mes, prefix="mes", drop_first=True).astype(float).reset_index(drop=True),
    ], axis=1)
    X_te = sm.add_constant(X_te)
    X_te = X_te.reindex(columns=X_tr.columns, fill_value=0.0)

    yhat_star = m.predict(X_te.values)
    yhat = yhat_star + bi * g_te.log_p_c.values
    y_true = g_te.log_qty.values

    rmse = float(np.sqrt(np.mean((yhat - y_true)**2)))
    mape = float(np.mean(np.abs(np.exp(yhat) - np.exp(y_true)) /
                         np.maximum(np.exp(y_true), 1)) * 100)
    yhat_in = m.predict(X_tr.values) + bi * g_tr.log_p_c.values
    rmse_in = float(np.sqrt(np.mean((yhat_in - g_tr.log_qty.values)**2)))
    metrics.append({
        "cve_art": sku, "n_test": len(g_te),
        "rmse_in": rmse_in, "rmse_out": rmse, "mape_out": mape,
        "beta_i": bi,
    })

mt = pd.DataFrame(metrics)
mt.to_parquet(DATA / "holdout_v3.parquet", index=False)
print(f"→ holdout_v3.parquet: {len(mt):,} SKUs validados")


# ---------------------------------------------------------------------------
# DIAGNÓSTICO Y COMPARACIÓN
# ---------------------------------------------------------------------------
print("\n" + "="*80)
print("DIAGNÓSTICO v3 (jerárquico)")
print("="*80)

print(f"\n— Cobertura por SKU —")
print(f"  Total SKUs en jerárquico:                 {len(sku_df):,}")
print(f"  β_i < 0 (signo correcto):                  {(sku_df.beta_i < 0).sum():,} "
      f"({(sku_df.beta_i < 0).mean()*100:.1f}%)")
print(f"  β_i < 0 y β_i ∈ (-5, 0):                   {((sku_df.beta_i<0)&(sku_df.beta_i>-5)).sum():,}")
print(f"  β_i < 0 y CI95 excluye 0:                  "
      f"{((sku_df.beta_i<0)&(sku_df.ci_hi<0)).sum():,}")
print(f"  β_i < 0, CI95 excluye 0, β_i ∈ (-5, 0):    "
      f"{((sku_df.beta_i<0)&(sku_df.ci_hi<0)&(sku_df.beta_i>-5)).sum():,}")

print(f"\n— Comparación de cobertura entre versiones —")
v1 = pd.read_parquet(DATA/"elasticidades_sku.parquet")
v2 = pd.read_parquet(DATA/"elasticidades_sku_v2.parquet")
def conteo(df, b, ci):
    if ci is None:
        return ((df[b]<0) & (df[b]>-5)).sum()
    return ((df[b]<0) & (df[b]>-5) & (df[ci]<0)).sum()
print(f"  v1 OLS naive:        {conteo(v1, 'beta', 'ci_hi')} con β<0 y CI<0")
print(f"  v2 OLS (con costo):  {conteo(v2, 'ols_beta', 'ols_ci_hi')} con β<0 y CI<0")
print(f"  v2 IV:               {conteo(v2, 'iv_beta', 'iv_ci_hi')} con β<0 y CI<0")
print(f"  v3 Jerárquico:       {((sku_df.beta_i<0)&(sku_df.ci_hi<0)&(sku_df.beta_i>-5)).sum()} con β<0 y CI<0")

print(f"\n— Holdout v3 —")
print(f"  RMSE in-sample (mediana):   {mt.rmse_in.median():.3f}")
print(f"  RMSE out-of-sample:         {mt.rmse_out.median():.3f}")
print(f"  MAPE out-of-sample (med):   {mt.mape_out.median():.1f}%")
print(f"  SKUs con RMSE_out<RMSE_in*1.5: "
      f"{(mt.rmse_out < mt.rmse_in*1.5).sum():,} ({(mt.rmse_out < mt.rmse_in*1.5).mean()*100:.1f}%)")

# Comparar holdout v2 vs v3
mt2 = pd.read_parquet(DATA/"holdout_metrics.parquet")
both = mt.merge(mt2, on="cve_art", suffixes=("_v3","_v2"))
print(f"\n  Comparación holdout (mismos {len(both):,} SKUs):")
print(f"    Mediana MAPE  v2: {both.mape_qty_out.median():.1f}%   v3: {both.mape_out.median():.1f}%")
print(f"    Mediana RMSE  v2: {both.rmse_log_out.median():.3f}   v3: {both.rmse_out.median():.3f}")
print(f"    SKUs donde v3 < v2 RMSE: {(both.rmse_out < both.rmse_log_out).sum()} "
      f"({(both.rmse_out < both.rmse_log_out).mean()*100:.1f}%)")

print(f"\n— Top 15 familias por volumen, β_fam jerárquico —")
fam_df_disp = fam_df.merge(
    train.groupby("fam").qty_no_promo.sum().rename("qty_total").reset_index(),
    on="fam"
).sort_values("qty_total", ascending=False)
print(fam_df_disp.head(15)[["fam","n_skus","n_obs","beta_fam","ci_lo_fam","ci_hi_fam",
                             "sigma_between_sku"]].to_string(index=False))

print(f"\n— Top 10 SKUs (alto volumen) con β_i shrunken —")
print(sku_df.head(10)[["cve_art","descripcion","fam","beta_i","ci_lo","ci_hi","n_train"]].to_string(index=False))
