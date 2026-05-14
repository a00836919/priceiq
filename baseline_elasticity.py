"""Baseline de elasticidad precio propia.

Modelo: log(qty_no_promo) = α + β·log(precio) + γ_dow + γ_mes + δ·log(costo) + ε
β es la elasticidad precio propia. Esperamos β < 0 y significativo.

Salidas:
  data/elasticidades_sku.parquet  — una fila por SKU con β, SE, IC, n, R², diagnósticos
  data/elasticidades_fam.parquet  — pooled por familia con efectos fijos de SKU
"""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
import warnings
warnings.filterwarnings("ignore")

DATA = Path("data")

# ---------------------------------------------------------------------------
# Cargar y preparar
# ---------------------------------------------------------------------------
panel = pd.read_parquet(DATA / "panel_diario.parquet")
panel["qty_no_promo"] = panel.qty_total - panel.qty_promo

# Filtros mínimos para tener observación válida
mask = (
    panel.precio_normal_sinpromo.notna()
    & (panel.precio_normal_sinpromo > 0)
    & (panel.qty_no_promo > 0)
    & panel.costo_vigente.notna()
    & (panel.costo_vigente > 0)
)
panel = panel[mask].copy()

panel["log_qty"] = np.log(panel.qty_no_promo)
panel["log_p"] = np.log(panel.precio_normal_sinpromo)
panel["log_c"] = np.log(panel.costo_vigente)
panel["dow"] = panel.dow.astype("category")
panel["mes"] = panel.mes.astype("category")

print(f"Observaciones válidas: {len(panel):,} ({panel.cve_art.nunique():,} SKUs)")


# ---------------------------------------------------------------------------
# Helper: estimar elasticidad para un grupo
# ---------------------------------------------------------------------------
def fit_loglog(df: pd.DataFrame, with_cost: bool = True):
    """Devuelve dict con β, SE, IC95, R², n para log(qty)~log(p)+dow+mes(+log_c)."""
    if df.log_p.std() < 0.01 or df.log_p.nunique() < 3 or len(df) < 30:
        return None
    cols = ["log_p"]
    if with_cost and df.log_c.std() > 0.01:
        cols.append("log_c")
    X = pd.concat([df[cols],
                   pd.get_dummies(df.dow, prefix="dow", drop_first=True).astype(float),
                   pd.get_dummies(df.mes, prefix="mes", drop_first=True).astype(float)],
                  axis=1)
    X = sm.add_constant(X)
    y = df.log_qty.values
    try:
        res = OLS(y, X.values, hasconst=True).fit(cov_type="HC3")
    except Exception:
        return None
    idx = list(X.columns).index("log_p")
    beta = res.params[idx]
    se = res.bse[idx]
    ci_lo, ci_hi = beta - 1.96 * se, beta + 1.96 * se
    pval = res.pvalues[idx]
    return {
        "beta": beta, "se": se, "ci_lo": ci_lo, "ci_hi": ci_hi,
        "pvalue": pval, "r2_adj": res.rsquared_adj, "n": int(res.nobs),
        "log_p_std": df.log_p.std(), "log_p_nunique": df.log_p.nunique(),
    }


# ---------------------------------------------------------------------------
# Estimación por SKU
# ---------------------------------------------------------------------------
print("\nEstimando elasticidad por SKU…")
results = []
for sku, g in panel.groupby("cve_art"):
    if len(g) < 100:  # mínimo de observaciones
        continue
    out = fit_loglog(g)
    if out is None:
        continue
    out["cve_art"] = sku
    out["lin"] = g.lin.iloc[0]
    out["fam"] = g.fam.iloc[0]
    out["marca"] = g.marca.iloc[0]
    out["descripcion"] = g.descripcion.iloc[0]
    out["qty_total"] = g.qty_no_promo.sum()
    results.append(out)

elasti_sku = pd.DataFrame(results)
elasti_sku = elasti_sku[["cve_art", "descripcion", "lin", "fam", "marca",
                         "n", "log_p_nunique", "log_p_std", "qty_total",
                         "beta", "se", "ci_lo", "ci_hi", "pvalue", "r2_adj"]]
elasti_sku = elasti_sku.sort_values("qty_total", ascending=False).reset_index(drop=True)
elasti_sku.to_parquet(DATA / "elasticidades_sku.parquet", index=False)

print(f"   → elasticidades_sku.parquet: {len(elasti_sku):,} SKUs estimados")


# ---------------------------------------------------------------------------
# Estimación por familia (pooled con SKU como FE)
# ---------------------------------------------------------------------------
print("\nEstimando elasticidad por familia (pooled, FE de SKU)…")
fam_results = []
for fam, g in panel.groupby("fam"):
    if g.cve_art.nunique() < 3 or len(g) < 500:
        continue
    if g.log_p.std() < 0.02:
        continue
    # FE de SKU vía dummies; demean sería más eficiente pero ok para baseline
    X = pd.concat([
        g[["log_p", "log_c"]],
        pd.get_dummies(g.cve_art, prefix="sku", drop_first=True).astype(float),
        pd.get_dummies(g.dow, prefix="dow", drop_first=True).astype(float),
        pd.get_dummies(g.mes, prefix="mes", drop_first=True).astype(float),
    ], axis=1)
    X = sm.add_constant(X)
    if X.shape[1] >= len(g):  # demasiados dummies
        continue
    try:
        res = OLS(g.log_qty.values, X.values, hasconst=True).fit(
            cov_type="cluster", cov_kwds={"groups": g.cve_art.values})
    except Exception:
        continue
    idx = list(X.columns).index("log_p")
    beta, se = res.params[idx], res.bse[idx]
    fam_results.append({
        "fam": fam,
        "n_skus": int(g.cve_art.nunique()),
        "n_obs": int(len(g)),
        "beta": beta, "se": se,
        "ci_lo": beta - 1.96 * se, "ci_hi": beta + 1.96 * se,
        "pvalue": res.pvalues[idx],
        "r2_adj": res.rsquared_adj,
        "qty_total": float(g.qty_no_promo.sum()),
    })

elasti_fam = pd.DataFrame(fam_results).sort_values("qty_total", ascending=False)
elasti_fam.to_parquet(DATA / "elasticidades_fam.parquet", index=False)
print(f"   → elasticidades_fam.parquet: {len(elasti_fam):,} familias estimadas")


# ---------------------------------------------------------------------------
# DIAGNÓSTICO
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("DIAGNÓSTICO DE VIABILIDAD")
print("=" * 80)

print(f"\n— Por SKU ({len(elasti_sku):,} estimados) —")
print(f"  β < 0 (signo correcto):                 "
      f"{(elasti_sku.beta < 0).sum():,} ({(elasti_sku.beta < 0).mean()*100:.1f}%)")
print(f"  β < 0 y significativo (p<0.05):         "
      f"{((elasti_sku.beta<0)&(elasti_sku.pvalue<0.05)).sum():,} "
      f"({((elasti_sku.beta<0)&(elasti_sku.pvalue<0.05)).mean()*100:.1f}%)")
print(f"  β en rango razonable [-5, 0]:           "
      f"{((elasti_sku.beta>=-5)&(elasti_sku.beta<0)).sum():,}")
print(f"  R² adj > 0.3:                            "
      f"{(elasti_sku.r2_adj>0.3).sum():,}")
print(f"\n  Distribución de β (todas):")
print(elasti_sku.beta.describe(percentiles=[.1,.25,.5,.75,.9]).to_string())

# Por línea: qué tan bien funciona en abarrotes vs frescos
print(f"\n— Por línea (signo correcto y significativo) —")
res_lin = elasti_sku.assign(ok=(elasti_sku.beta<0)&(elasti_sku.pvalue<0.05)).groupby("lin").agg(
    n=("cve_art","count"),
    pct_ok=("ok", lambda s: s.mean()*100),
    beta_med=("beta", "median"),
).sort_values("n", ascending=False).head(12)
print(res_lin.to_string())

print(f"\n— Por familia (top 15 por volumen) —")
print(elasti_fam.head(15)[["fam","n_skus","n_obs","beta","se","ci_lo","ci_hi","pvalue","r2_adj"]].to_string(index=False))

print(f"\n— Top 15 SKUs con elasticidad bien identificada (alto volumen, signif, β<0) —")
ok = elasti_sku[(elasti_sku.beta<0) & (elasti_sku.pvalue<0.01) & (elasti_sku.beta>=-5)]
print(ok.head(15)[["cve_art","descripcion","lin","fam","beta","se","ci_lo","ci_hi","r2_adj","n"]].to_string(index=False))
