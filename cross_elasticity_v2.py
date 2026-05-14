"""Elasticidad cruzada v2: top-10 vecinos + pooling jerárquico empírico-Bayesiano.

Dos mejoras:
  1. Subir top vecinos de 5 a 10 (4 sustitutos + 6 complementos por focal)
  2. Empirical Bayes shrinkage por grupo (fam_focal, fam_vecino, tipo):
       β_ij ~ N(μ_g, τ²)        — heterogeneidad entre pares del grupo
       β̂_ij ~ N(β_ij, σ²_ij)   — error de muestreo
     Posterior: combina β̂_ij con μ_g pesando por (1/σ²_ij + 1/τ²).
     Pares con poca data heredan más del grupo; pares bien medidos mantienen su valor.
  3. Diagnóstico de colinealidad por focal (condition number); skip si > 50

Salidas:
  - cross_elasticities_v2.parquet      : por par (focal, vecino) con β raw y β shrunk
  - cross_elasticities_groups.parquet  : por grupo (fam_a, fam_b, tipo) con μ_g y τ
"""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

DATA = Path("data")
MAX_SUBS = 4
MAX_COMP = 6
MIN_OVERLAP = 80
MAX_COND = 1e6   # threshold relajado: que la SE robusta refleje la colinealidad

# ---------------------------------------------------------------------------
# Cargar y preparar (igual que v1)
# ---------------------------------------------------------------------------
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
def trim_low(g):
    if len(g) < 30: return g
    z = (g.log_qty - g.log_qty.mean()) / g.log_qty.std()
    return g[z > -2.0]
panel = panel.groupby("cve_art", group_keys=False).apply(trim_low)

prices = panel[["fecha","cve_art","log_p"]].pivot_table(
    index="fecha", columns="cve_art", values="log_p", aggfunc="mean"
).sort_index().ffill(limit=14)

vec = pd.read_parquet(DATA / "sku_neighbors.parquet")
elas = pd.read_parquet(DATA / "elasticidades_jerarquico_sku.parquet")
focales_validos = elas[(elas.beta_i < 0) & (elas.beta_i > -5)].cve_art.unique()
focales_objetivo = sorted(set(focales_validos) & set(vec.foco.unique()))

def top_neighbors(g):
    sus = g[g.tipo=="sustituto"].nlargest(MAX_SUBS, "score_sustituto")
    com = g[g.tipo=="complemento"].nlargest(MAX_COMP, "lift")
    return pd.concat([sus, com])

vec_top = (vec[vec.foco.isin(focales_objetivo)]
           .groupby("foco", group_keys=False).apply(top_neighbors))
print(f"Focales: {len(focales_objetivo):,} | Pares: {len(vec_top):,}")


# ---------------------------------------------------------------------------
# 1. Estimación per-focal con OLS (con check de colinealidad)
# ---------------------------------------------------------------------------
print("\nEstimando OLS por focal con hasta 10 vecinos…")
results = []
n_skipped_cond = 0
for k, foc in enumerate(focales_objetivo, 1):
    if k % 200 == 0:
        print(f"  [{k}/{len(focales_objetivo)}] colinealidad-skip={n_skipped_cond}")

    g = panel[panel.cve_art == foc].copy()
    if len(g) < MIN_OVERLAP:
        continue
    nbrs_foc = vec_top[vec_top.foco == foc]
    if len(nbrs_foc) == 0:
        continue

    df = g[["fecha","log_qty","log_p","log_c"]].copy().rename(columns={"log_p":"log_p_self"})
    df["dow"] = df.fecha.dt.dayofweek
    df["mes"] = df.fecha.dt.month

    nbrs_used = []
    for _, r in nbrs_foc.iterrows():
        j = r.vecino
        if j not in prices.columns:
            continue
        df = df.merge(prices[[j]].rename(columns={j: f"lp_{j}"}),
                      left_on="fecha", right_index=True, how="left")
        nbrs_used.append((j, r.tipo, r.lift, r.fam_vec))
    if not nbrs_used:
        continue

    df = df.dropna()
    if len(df) < MIN_OVERLAP or df.log_p_self.std() < 0.01:
        continue

    X_cols = ["log_p_self", "log_c"] + [f"lp_{j}" for j,_,_,_ in nbrs_used]
    X = df[X_cols].copy()
    keep_cols = [c for c in X.columns if X[c].std() > 0.005]
    X = X[keep_cols]
    X = pd.concat([X.reset_index(drop=True),
                   pd.get_dummies(df.dow, prefix="dow", drop_first=True).astype(float).reset_index(drop=True),
                   pd.get_dummies(df.mes, prefix="mes", drop_first=True).astype(float).reset_index(drop=True)],
                  axis=1)
    X = sm.add_constant(X)
    if X.shape[1] >= len(df) - 5:
        continue

    # Diagnóstico de colinealidad: condition number
    Xv = X.values.astype(float)
    Xn = Xv / (np.linalg.norm(Xv, axis=0, keepdims=True) + 1e-12)
    cond = float(np.linalg.cond(Xn))
    if cond > MAX_COND:
        n_skipped_cond += 1
        continue

    try:
        m = sm.OLS(df.log_qty.values, X.values).fit(cov_type="HC3")
    except Exception:
        continue

    if "log_p_self" not in X.columns:
        continue
    idx_self = list(X.columns).index("log_p_self")
    beta_self = float(m.params[idx_self])
    se_self = float(m.bse[idx_self])

    for (j, tipo, lift_pair, fam_vec) in nbrs_used:
        col = f"lp_{j}"
        if col not in X.columns:
            continue
        idx = list(X.columns).index(col)
        b = float(m.params[idx]); se = float(m.bse[idx])
        results.append({
            "focal": foc, "vecino": j, "tipo_vecino": tipo,
            "lift": lift_pair, "fam_vec": fam_vec,
            "beta_self": beta_self, "se_self": se_self,
            "beta_raw": b, "se_raw": se,
            "ci_lo_raw": b - 1.96*se, "ci_hi_raw": b + 1.96*se,
            "pvalue_raw": float(m.pvalues[idx]),
            "n_obs": int(len(df)), "r2_adj": float(m.rsquared_adj),
            "cond_number": cond,
        })

cross = pd.DataFrame(results)
print(f"\n  Pares estimados: {len(cross):,} | "
      f"focales con colinealidad alta saltadas: {n_skipped_cond}")

# Adjuntar metadata
meta = panel.groupby("cve_art").agg(
    descripcion=("descripcion","first"),
    fam=("fam","first"),
).reset_index()
cross = cross.merge(meta.rename(columns={"cve_art":"focal","descripcion":"desc_focal","fam":"fam_focal"}),
                    on="focal", how="left")
cross = cross.merge(meta.rename(columns={"cve_art":"vecino","descripcion":"desc_vecino"}),
                    on="vecino", how="left").drop(columns=["fam"], errors="ignore")


# ---------------------------------------------------------------------------
# 2. Empirical Bayes shrinkage por grupo (fam_focal, fam_vec, tipo_vecino)
# ---------------------------------------------------------------------------
print("\nAplicando empirical Bayes por grupo (fam_focal, fam_vec, tipo)…")

def eb_shrink(group: pd.DataFrame) -> pd.DataFrame:
    """Pooling parcial con piso de heterogeneidad para evitar shrinkage degenerado.

    Cuando la varianza observada entre β_raw del grupo ≤ varianza muestral promedio,
    el estimador clásico colapsa τ²→0 y da SE_shrunk=0 (sobreconfianza).
    Solución: piso τ² ≥ 0.25·median(σ²) — prior débil de que sí hay heterogeneidad.
    Además, SE_shrunk incluye la incertidumbre de μ_g (no sólo del par individual).
    """
    n = len(group)
    if n < 3:
        out = group.copy()
        out["beta_shrunk"] = out.beta_raw
        out["se_shrunk"] = out.se_raw
        out["mu_group"] = out.beta_raw.mean()
        out["tau_group"] = np.nan
        out["weight_group"] = 0.0
        return out

    sigma2 = group.se_raw.values**2
    # τ²: estimador de momentos. Sin piso: si τ²_hat=0 hacemos full pooling
    # y la varianza posterior queda absorbida por la incertidumbre de μ_g.
    var_obs = float(group.beta_raw.var(ddof=1))
    tau2 = max(0.0, var_obs - sigma2.mean())
    tau = float(np.sqrt(tau2))

    # μ_g: media ponderada inversa-varianza
    w = 1.0 / (sigma2 + tau2)
    mu_g = float(np.sum(w * group.beta_raw.values) / np.sum(w))
    se_mu = float(np.sqrt(1.0 / np.sum(w)))   # SE de μ_g

    # Shrinkage por par. Si τ²=0, w_ind=0, beta_shrunk=μ_g y SE = SE_μ
    if tau2 > 0:
        w_ind = tau2 / (tau2 + sigma2)
    else:
        w_ind = np.zeros_like(sigma2)
    beta_shrunk = w_ind * group.beta_raw.values + (1 - w_ind) * mu_g
    # varianza posterior: combina incertidumbre del par y de μ_g
    var_shrunk = w_ind * sigma2 + (1 - w_ind)**2 * se_mu**2
    se_shrunk = np.sqrt(var_shrunk)

    out = group.copy()
    out["beta_shrunk"] = beta_shrunk
    out["se_shrunk"]   = se_shrunk
    out["mu_group"]    = mu_g
    out["se_mu_group"] = se_mu
    out["tau_group"]   = tau
    out["weight_group"] = 1 - w_ind  # cuánto se shrinkea hacia el grupo (0 = nada, 1 = todo)
    return out

cross["group"] = cross.fam_focal.astype(str) + "|" + cross.fam_vec.astype(str) + "|" + cross.tipo_vecino.astype(str)
cross = cross.groupby("group", group_keys=False).apply(eb_shrink)

cross["ci_lo_shrunk"] = cross.beta_shrunk - 1.96*cross.se_shrunk
cross["ci_hi_shrunk"] = cross.beta_shrunk + 1.96*cross.se_shrunk

# Z-test sobre shrunk
cross["z_shrunk"] = cross.beta_shrunk / cross.se_shrunk
cross["pvalue_shrunk"] = 2*(1 - pd.Series(np.abs(cross.z_shrunk)).apply(
    lambda z: 0.5*(1+np.math.erf(z/np.sqrt(2)))))

cross = cross[["focal","desc_focal","fam_focal","tipo_vecino",
               "vecino","desc_vecino","fam_vec","lift",
               "beta_raw","se_raw","ci_lo_raw","ci_hi_raw","pvalue_raw",
               "beta_shrunk","se_shrunk","ci_lo_shrunk","ci_hi_shrunk","pvalue_shrunk",
               "mu_group","se_mu_group","tau_group","weight_group",
               "beta_self","se_self","n_obs","r2_adj","cond_number"]]
cross.to_parquet(DATA / "cross_elasticities_v2.parquet", index=False)


# Tabla a nivel grupo
groups = (cross.groupby(["fam_focal","fam_vec","tipo_vecino"]).agg(
    n_pairs=("focal","count"),
    mu_group=("mu_group","first"),
    se_mu_group=("se_mu_group","first"),
    tau_group=("tau_group","first"),
    median_n_obs=("n_obs","median"),
    weight_group_mean=("weight_group","mean"),
).reset_index())
groups["ci_lo_group"] = groups.mu_group - 1.96*groups.se_mu_group
groups["ci_hi_group"] = groups.mu_group + 1.96*groups.se_mu_group
groups.to_parquet(DATA / "cross_elasticities_groups.parquet", index=False)


# ---------------------------------------------------------------------------
# DIAGNÓSTICO
# ---------------------------------------------------------------------------
print("\n" + "="*80, "\nDIAGNÓSTICO v2 (top-10 + jerárquico)\n", "="*80)

print(f"\nPares estimados:           {len(cross):,}")
print(f"Grupos (fam_a, fam_b, tipo): {len(groups):,}")

sub = cross[cross.tipo_vecino == "sustituto"]
com = cross[cross.tipo_vecino == "complemento"]

print(f"\n— SUSTITUTOS (β esperado >0) —")
for tag, b, ci_lo, p in [("RAW   ","beta_raw","ci_lo_raw","pvalue_raw"),
                         ("SHRUNK","beta_shrunk","ci_lo_shrunk","pvalue_shrunk")]:
    n_pos = (sub[b] > 0).sum()
    n_sig = ((sub[b] > 0) & (sub[ci_lo] > 0)).sum()
    print(f"  {tag} | β>0: {n_pos:5d} ({n_pos/len(sub)*100:5.1f}%) | "
          f"β>0 + CI excluye 0: {n_sig:5d} ({n_sig/len(sub)*100:5.1f}%)")

print(f"\n— COMPLEMENTOS (β esperado <0) —")
for tag, b, ci_hi, p in [("RAW   ","beta_raw","ci_hi_raw","pvalue_raw"),
                         ("SHRUNK","beta_shrunk","ci_hi_shrunk","pvalue_shrunk")]:
    n_neg = (com[b] < 0).sum()
    n_sig = ((com[b] < 0) & (com[ci_hi] < 0)).sum()
    print(f"  {tag} | β<0: {n_neg:5d} ({n_neg/len(com)*100:5.1f}%) | "
          f"β<0 + CI excluye 0: {n_sig:5d} ({n_sig/len(com)*100:5.1f}%)")

# Comparación con v1
print("\n— Comparación cobertura cruzadas v1 vs v2 —")
v1 = pd.read_parquet(DATA / "cross_elasticities.parquet")
v1_sub_ok = ((v1.tipo_vecino=="sustituto")&(v1.beta_cross>0)&(v1.ci_lo>0)).sum()
v1_com_ok = ((v1.tipo_vecino=="complemento")&(v1.beta_cross<0)&(v1.ci_hi<0)).sum()
v2_sub_ok = ((cross.tipo_vecino=="sustituto")&(cross.beta_shrunk>0)&(cross.ci_lo_shrunk>0)).sum()
v2_com_ok = ((cross.tipo_vecino=="complemento")&(cross.beta_shrunk<0)&(cross.ci_hi_shrunk<0)).sum()
print(f"  v1 (top-5,  raw):    sustitutos={v1_sub_ok:4d} | complementos={v1_com_ok:4d}")
print(f"  v2 (top-10, shrunk): sustitutos={v2_sub_ok:4d} | complementos={v2_com_ok:4d}")

# Top grupos con canibalización media identificada
print("\n— TOP 10 grupos (fam_focal, fam_vec, tipo) con μ_g más fuerte y bien identificada —")
sub_g = groups[(groups.tipo_vecino=="sustituto") & (groups.mu_group>0) & (groups.n_pairs>=5)]
sub_g = sub_g.sort_values("mu_group", ascending=False)
print("\n  Sustitutos (canibalización promedio):")
print(sub_g.head(10)[["fam_focal","fam_vec","n_pairs","mu_group","tau_group",
                      "ci_lo_group","ci_hi_group"]].to_string(index=False))

com_g = groups[(groups.tipo_vecino=="complemento") & (groups.mu_group<0) & (groups.n_pairs>=5)]
com_g = com_g.sort_values("mu_group")
print("\n  Complementos (efecto compra-conjunta promedio):")
print(com_g.head(10)[["fam_focal","fam_vec","n_pairs","mu_group","tau_group",
                      "ci_lo_group","ci_hi_group"]].to_string(index=False))

# Ejemplo: ahora con shrinkage la Coca 600ml debería tener IC más estrecho
print("\n— Ejemplo Coca 600ml (006913) con cruzadas SHRUNK —")
ej = cross[cross.focal=="006913"].sort_values("beta_shrunk", ascending=False)
print(ej[["tipo_vecino","desc_vecino","fam_vec","beta_raw","beta_shrunk","se_shrunk","ci_lo_shrunk","ci_hi_shrunk","weight_group"]].to_string(index=False))
