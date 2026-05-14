"""Elasticidad cruzada DIRIGIDA: sólo entre SKUs identificados como vecinos.

Modelo por SKU focal i:
   log(qty_i,t) = α + β_ii · log(p_i,t)
                    + Σ_{j ∈ vecinos(i)} β_ij · log(p_j,t)
                    + δ · log(c_i,t) + dow + mes + ε

Interpretación:
   β_ii < 0 (propia)
   β_ij > 0 → SUSTITUTO (j más caro → la gente compra más i)
   β_ij < 0 → COMPLEMENTO (j más caro → menos canastas → menos i)

Estrategia para multicolinearidad:
   - Máximo 5 vecinos por focal (3 sustitutos + 2 complementos por fuerza)
   - Filtrar focal con β_ii previo identificable (de v3 jerárquico)
   - Forward-fill precio del vecino en días sin venta del vecino
   - Días con precio del vecino missing tras FFill → drop

Salidas en data/:
   - cross_elasticities.parquet  — fila por par (focal, vecino) con β_ij, SE, IC95, signo, tipo
"""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

DATA = Path("data")
MAX_SUBS = 3
MAX_COMP = 2
MIN_OVERLAP = 80   # mínimo días con todos los precios disponibles

# ---------------------------------------------------------------------------
# Cargar
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

# trim asimétrico de stockout (igual que v3)
def trim_low(g):
    if len(g) < 30: return g
    z = (g.log_qty - g.log_qty.mean()) / g.log_qty.std()
    return g[z > -2.0]
panel = panel.groupby("cve_art", group_keys=False).apply(trim_low)

# Tabla pivote de precios diarios por SKU (para hacer joins eficientes)
print("Construyendo tabla de precios por SKU-día (para vecinos)…")
prices = panel[["fecha","cve_art","log_p"]].pivot_table(
    index="fecha", columns="cve_art", values="log_p", aggfunc="mean")
# Forward-fill por SKU sólo dentro de [primera_venta, última_venta]
prices = prices.sort_index().ffill(limit=14)  # limita FFill a 14 días
print(f"   matriz de log-precios: {prices.shape[0]} días × {prices.shape[1]:,} SKUs")

# Vecinos
vec = pd.read_parquet(DATA / "sku_neighbors.parquet")
print(f"Vecinos disponibles: {len(vec):,} relaciones, "
      f"{vec.foco.nunique():,} SKUs focales")

# Para focal: filtrar a los con β_ii previo razonable
elas = pd.read_parquet(DATA / "elasticidades_jerarquico_sku.parquet")
focales_validos = elas[(elas.beta_i < 0) & (elas.beta_i > -5)].cve_art.unique()
focales_objetivo = set(focales_validos) & set(vec.foco.unique())
print(f"Focales con β_ii válido y vecinos: {len(focales_objetivo):,}")

# Para cada focal, tomar top sustitutos y top complementos
def top_neighbors(g):
    sus = g[g.tipo=="sustituto"].nlargest(MAX_SUBS, "score_sustituto")
    com = g[g.tipo=="complemento"].nlargest(MAX_COMP, "lift")
    return pd.concat([sus, com])

vec_top = (vec[vec.foco.isin(focales_objetivo)]
           .groupby("foco", group_keys=False).apply(top_neighbors))
print(f"Pares focal-vecino a estimar: {len(vec_top):,}")


# ---------------------------------------------------------------------------
# Estimar por focal
# ---------------------------------------------------------------------------
print("\nEstimando elasticidad cruzada por focal…")
results = []
focales_lista = list(focales_objetivo)
n_total = len(focales_lista)

for k, foc in enumerate(focales_lista, 1):
    if k % 200 == 0:
        print(f"  [{k}/{n_total}]")

    g = panel[panel.cve_art == foc].copy()
    if len(g) < MIN_OVERLAP:
        continue

    vecinos_foc = vec_top[vec_top.foco == foc]
    if len(vecinos_foc) == 0:
        continue

    # Adjuntar log_p de cada vecino
    df = g[["fecha","log_qty","log_p","log_c"]].copy()
    df = df.rename(columns={"log_p":"log_p_self"})
    df["dow"] = df.fecha.dt.dayofweek
    df["mes"] = df.fecha.dt.month

    nbrs_used = []
    for _, r in vecinos_foc.iterrows():
        j = r.vecino
        if j not in prices.columns:
            continue
        df = df.merge(prices[[j]].rename(columns={j: f"lp_{j}"}),
                      left_on="fecha", right_index=True, how="left")
        nbrs_used.append((j, r.tipo, r.lift, r.fam_vec))

    if not nbrs_used:
        continue

    # Drop NA en cualquier precio de vecino (días sin precio FFilleable)
    df = df.dropna()
    if len(df) < MIN_OVERLAP:
        continue

    # Variación en log_p_self requerida
    if df.log_p_self.std() < 0.01:
        continue

    # Construir X
    X_cols = ["log_p_self", "log_c"] + [f"lp_{j}" for j,_,_,_ in nbrs_used]
    X = df[X_cols].copy()
    # Si algún vecino tiene varianza ~0, eliminarlo
    keep_cols = [c for c in X.columns if X[c].std() > 0.005]
    X = X[keep_cols]
    X = pd.concat([X.reset_index(drop=True),
                   pd.get_dummies(df.dow, prefix="dow", drop_first=True).astype(float).reset_index(drop=True),
                   pd.get_dummies(df.mes, prefix="mes", drop_first=True).astype(float).reset_index(drop=True)],
                  axis=1)
    X = sm.add_constant(X)
    if X.shape[1] >= len(df) - 5:
        continue

    y = df.log_qty.values
    try:
        m = sm.OLS(y, X.values).fit(cov_type="HC3")
    except Exception:
        continue

    # β_ii (propia)
    if "log_p_self" not in X.columns:
        continue
    idx_self = list(X.columns).index("log_p_self")
    beta_self = m.params[idx_self]
    se_self = m.bse[idx_self]

    # β_ij por vecino
    for (j, tipo, lift_pair, fam_vec) in nbrs_used:
        col = f"lp_{j}"
        if col not in X.columns:
            continue
        idx = list(X.columns).index(col)
        b = m.params[idx]
        se = m.bse[idx]
        results.append({
            "focal": foc,
            "vecino": j,
            "tipo_vecino": tipo,
            "lift": lift_pair,
            "fam_vec": fam_vec,
            "beta_self": beta_self,
            "se_self": se_self,
            "beta_cross": b,
            "se_cross": se,
            "ci_lo": b - 1.96*se,
            "ci_hi": b + 1.96*se,
            "pvalue": m.pvalues[idx],
            "n_obs": int(len(df)),
            "r2_adj": m.rsquared_adj,
        })

cross = pd.DataFrame(results)

# Adjuntar metadata
meta = panel.groupby("cve_art").agg(
    descripcion=("descripcion","first"),
    fam=("fam","first"),
).reset_index()
cross = cross.merge(meta.rename(columns={"cve_art":"focal","descripcion":"desc_focal","fam":"fam_focal"}),
                    on="focal", how="left")
cross = cross.merge(meta.rename(columns={"cve_art":"vecino","descripcion":"desc_vecino"}),
                    on="vecino", how="left")
cross = cross.drop(columns=["fam"], errors="ignore")

cross = cross[["focal","desc_focal","fam_focal","tipo_vecino",
               "vecino","desc_vecino","fam_vec","lift",
               "beta_cross","se_cross","ci_lo","ci_hi","pvalue",
               "beta_self","se_self","n_obs","r2_adj"]]
cross.to_parquet(DATA / "cross_elasticities.parquet", index=False)
print(f"\n→ cross_elasticities.parquet: {len(cross):,} pares estimados")


# ---------------------------------------------------------------------------
# DIAGNÓSTICO
# ---------------------------------------------------------------------------
print("\n" + "="*80, "\nDIAGNÓSTICO\n", "="*80)

print(f"\nTotal pares estimados: {len(cross):,}")
print(f"  Sustitutos: {(cross.tipo_vecino=='sustituto').sum():,}")
print(f"  Complementos: {(cross.tipo_vecino=='complemento').sum():,}")

# Signos esperados:
# Sustitutos → β_ij > 0 (precio del sustituto sube → mi demanda sube)
# Complementos → β_ij < 0
sub = cross[cross.tipo_vecino == "sustituto"]
com = cross[cross.tipo_vecino == "complemento"]

print(f"\nSUSTITUTOS — β_ij esperado > 0:")
print(f"  β > 0 (signo correcto):           {(sub.beta_cross > 0).sum():,} ({(sub.beta_cross > 0).mean()*100:.1f}%)")
print(f"  β > 0 y p<0.05 (significativo):   {((sub.beta_cross > 0) & (sub.pvalue < 0.05)).sum():,}")
print(f"  β > 0, p<0.05, IC excluye 0:      {((sub.beta_cross > 0) & (sub.ci_lo > 0)).sum():,}")
print(f"  β median: {sub.beta_cross.median():+.3f}")

print(f"\nCOMPLEMENTOS — β_ij esperado < 0:")
print(f"  β < 0 (signo correcto):           {(com.beta_cross < 0).sum():,} ({(com.beta_cross < 0).mean()*100:.1f}%)")
print(f"  β < 0 y p<0.05 (significativo):   {((com.beta_cross < 0) & (com.pvalue < 0.05)).sum():,}")
print(f"  β < 0, p<0.05, IC excluye 0:      {((com.beta_cross < 0) & (com.ci_hi < 0)).sum():,}")
print(f"  β median: {com.beta_cross.median():+.3f}")

# Top sustitutos bien identificados (canibalización clara)
print(f"\n— TOP 15 sustitutos con CANIBALIZACIÓN identificada (β_ij>0, p<0.05, signif) —")
sub_ok = sub[(sub.beta_cross > 0) & (sub.pvalue < 0.05) & (sub.ci_lo > 0)].copy()
sub_ok["fuerza"] = sub_ok.beta_cross / sub_ok.se_cross
print(sub_ok.nlargest(15, "fuerza")[
    ["desc_focal","desc_vecino","fam_focal","beta_cross","ci_lo","ci_hi","beta_self","n_obs"]
].to_string(index=False))

# Top complementos bien identificados
print(f"\n— TOP 15 complementos identificados (β_ij<0, p<0.05) —")
com_ok = com[(com.beta_cross < 0) & (com.pvalue < 0.05) & (com.ci_hi < 0)].copy()
com_ok["fuerza"] = -com_ok.beta_cross / com_ok.se_cross
print(com_ok.nlargest(15, "fuerza")[
    ["desc_focal","desc_vecino","fam_focal","fam_vec","beta_cross","ci_lo","ci_hi","n_obs"]
].to_string(index=False))

# Caso de uso real: para los sustitutos, ¿cuánto canibaliza una bajada de precio?
print(f"\n— Ejemplo de simulación: si bajas el precio de Coca 600ml 10% —")
coca = "006913"  # COCA COLA 600ML
suss = cross[(cross.focal == coca) & (cross.tipo_vecino == "sustituto")]
if len(suss) > 0:
    own = elas[elas.cve_art == coca]
    if len(own) > 0:
        b_own = own.beta_i.iloc[0]
        print(f"   β_ii (propia): {b_own:.3f}")
        print(f"   Demanda propia subiría ~{-b_own*0.10*100:.1f}%")
        print(f"   Pero canibaliza a:")
        for _, r in suss.iterrows():
            sig = "✓" if (r.beta_cross > 0 and r.pvalue < 0.05) else "?"
            # si bajamos p_self 10%, p_self / p_vec baja ⇒ ¿efecto en qty_vec?
            # qty_vec depende de β del modelo de qty_vec. Aquí estimamos efecto de p_vec en qty_self.
            # Para canibalización inversa necesitamos el reverso (qty_vec ~ p_self) que NO está en este modelo
            # — sólo podemos reportar la dirección detectada.
            print(f"   {sig} {r.desc_vecino[:50]:50s} β_ij={r.beta_cross:+.3f} (IC [{r.ci_lo:+.2f},{r.ci_hi:+.2f}])")
