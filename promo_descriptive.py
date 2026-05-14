"""Descriptivo de promociones históricas — observación directa de canibalización.

Lógica:
  1. Identificar EVENTO = período contiguo de promo para (cve_art, NumProm)
  2. Para cada evento medir:
       uplift propio  = qty_diario(durante) / qty_diario(baseline)  − 1
       descuento real = 1 − precio_efectivo / precio_normal
  3. Para cada sustituto del SKU promovido:
       canibalización = qty_diario(sustituto durante) / qty_diario(sustituto baseline) − 1
       (sólo si el sustituto NO está él mismo en promo)
  4. Comparar canibalización observada vs β_ij del modelo
       (signo coincide? magnitud razonable?)

Salidas:
  - promo_events.parquet           : un row por evento (focal, NumProm) con uplift propio
  - promo_canibalization.parquet   : un row por (evento, sustituto) con efecto observado
  - promo_model_validation.parquet : comparación observado vs β_ij predicho
"""
from pathlib import Path
import numpy as np
import pandas as pd

SRC = Path("tablas_merksyst_parquet")
DATA = Path("data")

PRE_DAYS = 14    # ventana baseline antes de la promo
POST_DAYS = 14   # opcional — no la usamos para baseline, sí para post-mortem
MIN_PROMO_DAYS = 2
MIN_LINES = 5

# ---------------------------------------------------------------------------
# 1. Cargar transacciones y construir tabla diaria SKU × día
# ---------------------------------------------------------------------------
print("[1/5] Cargando transacciones y armando matriz diaria…")
det = pd.read_parquet(SRC / "pdvdet.parquet",
    columns=["FecOpe","cve_art","cantidad","precio","precio_normal",
             "tipo_promocion","NumProm","status"])
det = det[(det.status == "01") & (det.cantidad > 0)].copy()
det["cve_art"] = det.cve_art.str.strip()
det["NumProm"] = det.NumProm.str.strip()
det["tp"] = det.tipo_promocion.str.strip()
det["fecha"] = pd.to_datetime(det.FecOpe, format="%Y%m%d")
det["es_promo"] = det.tp != ""

# Diario por SKU: qty con y sin promo
diario = det.groupby(["cve_art","fecha"]).agg(
    qty=("cantidad","sum"),
    qty_promo=("cantidad", lambda s: s[det.loc[s.index,"es_promo"]].sum()),
    precio_med=("precio","median"),
    precio_normal_med=("precio_normal","median"),
).reset_index()
diario["qty_no_promo"] = diario.qty - diario.qty_promo

# Para baseline: tabla de días sin promo del SKU
no_promo_days = diario[diario.qty_promo == 0].copy()
print(f"   diario: {len(diario):,} filas | días-SKU sin promo: {len(no_promo_days):,}")

# ---------------------------------------------------------------------------
# 2. Detectar EVENTOS: un evento = período contiguo de promo de (cve_art, NumProm)
# ---------------------------------------------------------------------------
print("\n[2/5] Detectando eventos de promoción…")
promo_lines = det[det.es_promo & (det.NumProm != "")].copy()
print(f"   líneas con NumProm válido: {len(promo_lines):,}")
print(f"   NumProm distintos: {promo_lines.NumProm.nunique()}")
print(f"   SKUs distintos en promo: {promo_lines.cve_art.nunique()}")

# Agrupar por (cve_art, NumProm) — cada combinación es un evento
ev = promo_lines.groupby(["cve_art","NumProm"]).agg(
    fec_min=("fecha","min"),
    fec_max=("fecha","max"),
    n_lines=("cantidad","size"),
    qty_during=("cantidad","sum"),
    p_med=("precio","median"),
    pn_med=("precio_normal","median"),
).reset_index()
ev["dias_promo"] = (ev.fec_max - ev.fec_min).dt.days + 1
ev = ev[(ev.n_lines >= MIN_LINES) & (ev.dias_promo >= MIN_PROMO_DAYS)].copy()
ev["qty_during_per_day"] = ev.qty_during / ev.dias_promo
ev["descuento_pct"] = 1 - ev.p_med / ev.pn_med
print(f"   eventos válidos (>={MIN_LINES} líneas, >={MIN_PROMO_DAYS} días): {len(ev):,}")
print(f"   eventos por SKU — describe:\n{ev.groupby('cve_art').size().describe().to_string()}")


# ---------------------------------------------------------------------------
# 3. Baseline pre-promo y uplift propio
# ---------------------------------------------------------------------------
print("\n[3/5] Calculando baseline pre-promo y uplift propio…")
# Para cada evento: media diaria del SKU en los PRE_DAYS antes (sólo días sin promo)
no_promo_idx = no_promo_days.set_index(["cve_art","fecha"])

def baseline_for(sku, start, days_back=PRE_DAYS):
    rng_start = start - pd.Timedelta(days=days_back)
    rng_end   = start - pd.Timedelta(days=1)
    if (sku, rng_start) > no_promo_idx.index.max():
        return np.nan, 0
    if sku not in no_promo_days.cve_art.unique():
        return np.nan, 0
    sub = no_promo_days[(no_promo_days.cve_art == sku) &
                        (no_promo_days.fecha >= rng_start) &
                        (no_promo_days.fecha <= rng_end)]
    if len(sub) == 0:
        return np.nan, 0
    return float(sub.qty_no_promo.mean()), len(sub)

baselines, n_baseline_days = [], []
for _, r in ev.iterrows():
    b, n = baseline_for(r.cve_art, r.fec_min)
    baselines.append(b); n_baseline_days.append(n)
ev["qty_baseline_per_day"] = baselines
ev["n_baseline_days"] = n_baseline_days
ev["uplift_pct"] = (ev.qty_during_per_day - ev.qty_baseline_per_day) / ev.qty_baseline_per_day * 100
ev["uplift_x"] = ev.qty_during_per_day / ev.qty_baseline_per_day  # multiplicador

# Adjuntar metadata
iar = pd.read_parquet(SRC / "inviar.parquet", columns=["art","fam","s_fam","lin","marca","des1"])
iar["art"] = iar.art.str.strip()
iar = iar.rename(columns={"art":"cve_art","des1":"descripcion"})
for c in ["fam","s_fam","lin","marca","descripcion"]:
    iar[c] = iar[c].str.strip()
ev = ev.merge(iar, on="cve_art", how="left")

ev_clean = ev[ev.qty_baseline_per_day.notna() & (ev.n_baseline_days >= 3)].copy()
print(f"   eventos con baseline válido (>=3 días sin promo previos): {len(ev_clean):,}")
print(f"   uplift_pct describe:\n{ev_clean.uplift_pct.describe(percentiles=[.1,.25,.5,.75,.9]).to_string()}")

ev_clean.to_parquet(DATA / "promo_events.parquet", index=False)


# ---------------------------------------------------------------------------
# 4. CANIBALIZACIÓN: efecto en sustitutos durante cada evento
# ---------------------------------------------------------------------------
print("\n[4/5] Midiendo canibalización en sustitutos…")
vec = pd.read_parquet(DATA / "sku_neighbors.parquet")
sus = vec[vec.tipo == "sustituto"][["foco","vecino"]].rename(
    columns={"foco":"focal","vecino":"sustituto"})
print(f"   relaciones de sustituto disponibles: {len(sus):,}")

# Conjunto de días de promo por SKU (para excluir cuando el sustituto está él mismo en promo)
promo_days_set = (promo_lines.groupby("cve_art").fecha.apply(set)).to_dict()

canib_rows = []
for _, e in ev_clean.iterrows():
    susts = sus[sus.focal == e.cve_art].sustituto.tolist()
    if not susts:
        continue
    rng_dur = pd.date_range(e.fec_min, e.fec_max)
    rng_pre_lo = e.fec_min - pd.Timedelta(days=PRE_DAYS)
    rng_pre_hi = e.fec_min - pd.Timedelta(days=1)
    rng_pre = pd.date_range(rng_pre_lo, rng_pre_hi)

    for sb in susts:
        sb_promo = promo_days_set.get(sb, set())
        # Días disponibles del sustituto sin promo propia
        # durante el evento focal: días en rng_dur que no están en sb_promo
        dur_days_clean = [d for d in rng_dur if d not in sb_promo]
        pre_days_clean = [d for d in rng_pre if d not in sb_promo]
        if len(dur_days_clean) < 1 or len(pre_days_clean) < 3:
            continue
        # qty del sustituto en esos días
        sb_panel = diario[diario.cve_art == sb]
        if len(sb_panel) == 0:
            continue
        qty_dur = sb_panel[sb_panel.fecha.isin(dur_days_clean)].qty_no_promo.sum() / max(len(dur_days_clean),1)
        qty_pre = sb_panel[sb_panel.fecha.isin(pre_days_clean)].qty_no_promo.sum() / max(len(pre_days_clean),1)
        if qty_pre <= 0:
            continue
        canib_rows.append({
            "focal": e.cve_art, "desc_focal": e.descripcion,
            "fam_focal": e.fam, "NumProm": e.NumProm,
            "fec_min": e.fec_min, "fec_max": e.fec_max,
            "dias_promo": e.dias_promo,
            "descuento_pct": e.descuento_pct,
            "uplift_focal_pct": e.uplift_pct,
            "sustituto": sb,
            "qty_sust_baseline": qty_pre,
            "qty_sust_durante": qty_dur,
            "canib_pct": (qty_dur - qty_pre)/qty_pre*100,
        })

canib = pd.DataFrame(canib_rows)
canib = canib.merge(iar.rename(columns={"cve_art":"sustituto","descripcion":"desc_sustituto",
                                          "fam":"fam_sust"})[["sustituto","desc_sustituto","fam_sust"]],
                    on="sustituto", how="left")
canib.to_parquet(DATA / "promo_canibalization.parquet", index=False)
print(f"   pares (evento × sustituto) medidos: {len(canib):,}")


# ---------------------------------------------------------------------------
# 5. VALIDACIÓN: ¿el modelo de β_ij predice lo observado?
# ---------------------------------------------------------------------------
print("\n[5/5] Validando contra β_ij del modelo cruzado…")
cross = pd.read_parquet(DATA / "cross_elasticities_v2.parquet")
cross_sus = cross[cross.tipo_vecino == "sustituto"][
    ["focal","vecino","beta_raw","beta_shrunk","ci_lo_shrunk","ci_hi_shrunk","pvalue_raw"]
].rename(columns={"vecino":"sustituto","beta_raw":"beta_modelo_raw","beta_shrunk":"beta_modelo_shrunk"})

# Promediar canibalización observada por par (focal, sustituto) sobre todos sus eventos
obs_pair = canib.groupby(["focal","sustituto"]).agg(
    n_eventos=("canib_pct","size"),
    canib_pct_med=("canib_pct","median"),
    canib_pct_mean=("canib_pct","mean"),
    descuento_med=("descuento_pct","median"),
    uplift_focal_med=("uplift_focal_pct","median"),
).reset_index()

val = obs_pair.merge(cross_sus, on=["focal","sustituto"], how="left")
# La intuición: si focal baja precio (descuento>0), demanda focal sube, sustituto baja (canib<0)
# β_modelo > 0 implica que SUBIR precio focal hace SUBIR demanda sustituto.
# Por tanto BAJAR precio focal hace BAJAR demanda sustituto → canib_observada < 0 si β>0
# Predicción modelo: canib_predicha_pct ≈ -β_modelo · descuento_med · 100
val["canib_predicho_pct"] = -val.beta_modelo_shrunk * val.descuento_med * 100
val["match_signo"] = (np.sign(val.canib_pct_med) == np.sign(val.canib_predicho_pct)) & val.beta_modelo_shrunk.notna()
val.to_parquet(DATA / "promo_model_validation.parquet", index=False)


# ---------------------------------------------------------------------------
# DIAGNÓSTICO
# ---------------------------------------------------------------------------
print("\n" + "="*80, "\nDIAGNÓSTICO PROMOCIONES HISTÓRICAS\n", "="*80)

print(f"\nEventos de promoción detectados: {len(ev_clean):,}")
print(f"  SKUs con al menos 1 evento medible: {ev_clean.cve_art.nunique():,}")
print(f"  Total promociones distintas (NumProm): {ev_clean.NumProm.nunique()}")
print(f"  Mediana descuento: {ev_clean.descuento_pct.median()*100:.1f}%")
print(f"  Mediana uplift propio: {ev_clean.uplift_pct.median():.1f}%")

# Top promociones por uplift
print("\n— TOP 10 eventos por uplift propio (≥10 días promo, ≥10% descuento) —")
top = ev_clean[(ev_clean.dias_promo>=10) & (ev_clean.descuento_pct>=0.10)].nlargest(10, "uplift_pct")
print(top[["cve_art","descripcion","fam","NumProm","dias_promo",
           "descuento_pct","uplift_pct","qty_baseline_per_day","qty_during_per_day"]].to_string(index=False))

# Top canibalización observada
print(f"\n— TOP 10 sustitutos con MAYOR canibalización observada —")
top_canib = canib[(canib.canib_pct < -10) & (canib.descuento_pct > 0.05)].nsmallest(10, "canib_pct")
print(top_canib[["desc_focal","desc_sustituto","fam_focal","descuento_pct","uplift_focal_pct","canib_pct"]].to_string(index=False))

# Distribución de canibalización
print(f"\n— Distribución de canibalización observada (canib_pct) —")
print(canib.canib_pct.describe(percentiles=[.1,.25,.5,.75,.9]).to_string())
print(f"  Pares con canibalización < -10%:  {(canib.canib_pct<-10).sum():,} ({(canib.canib_pct<-10).mean()*100:.1f}%)")
print(f"  Pares con efecto positivo > +10%: {(canib.canib_pct>10).sum():,} (puede ser efecto-bundle)")

# Validación modelo
val_ok = val[val.beta_modelo_shrunk.notna()]
print(f"\n— Validación contra modelo cruzado —")
print(f"  Pares con observado Y β_modelo: {len(val_ok):,}")
if len(val_ok) > 0:
    print(f"  Coincidencia de signo (modelo predice dirección correcta): "
          f"{val_ok.match_signo.sum():,} ({val_ok.match_signo.mean()*100:.1f}%)")
    corr = val_ok[["canib_pct_med","canib_predicho_pct"]].corr().iloc[0,1]
    print(f"  Correlación obs vs predicho:    {corr:+.3f}")

# Familias con uplift más alto (más sensibles a promo)
print(f"\n— Top 10 familias por uplift mediano (n_eventos≥10) —")
fam_up = ev_clean.groupby("fam").agg(
    n_ev=("uplift_pct","size"),
    uplift_med=("uplift_pct","median"),
    desc_med=("descuento_pct","median"),
).query("n_ev>=10").sort_values("uplift_med", ascending=False).head(10)
print(fam_up.to_string())

# Promociones rentables vs no — métrica heurística
ev_clean["uplift_x_neto"] = ev_clean.uplift_x * (1 - ev_clean.descuento_pct)
print(f"\n— Promociones que valieron la pena vs no (heurística: ingreso relativo) —")
print(f"  Eventos con ingreso > baseline (uplift_x · (1-desc) > 1):  "
      f"{(ev_clean.uplift_x_neto > 1).sum():,} de {len(ev_clean):,} "
      f"({(ev_clean.uplift_x_neto > 1).mean()*100:.1f}%)")
print(f"  Mediana de uplift_x_neto:  {ev_clean.uplift_x_neto.median():.2f}")
