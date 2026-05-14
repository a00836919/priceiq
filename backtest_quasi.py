"""Backtest quasi-experimental: validar el modelo en CAMBIOS REALES de precio.

Esto responde a "¿el modelo predice bien cuando el precio cambia?" — es la pregunta
que importa para la herramienta porque su caso de uso es justamente cambios de precio.

Lógica:
  1. Identificar eventos de cambio de precio en el TEST SET (no contaminado por training).
     Un evento = día donde |Δp| ≥ 5% vs precio mediano de los 7 días anteriores.
  2. Para cada evento, comparar:
       qty_real_post     — observada en los días POST-cambio
       qty_predicha_post — predicción del modelo al precio NUEVO
       qty_real_pre      — observada antes (baseline)
       qty_predicha_pre  — predicción al precio viejo
  3. Métricas:
       MAPE_post     — error del modelo en el caso de uso real
       bias_post     — sesgo del modelo en cambios
       cobertura IC  — % de obs post-cambio dentro del IC calibrado
       direction_acc — % de veces que el modelo acertó el SIGNO del cambio en qty

Esto valida que "cuando el precio cambia, el modelo sí captura el efecto".

Salidas:
  data/backtest_quasi_events.parquet  — un row por evento
  data/backtest_quasi_summary.parquet — agregado global y por familia
"""
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

DATA = Path("data")

# Tunables
MIN_PCT_CHANGE = 0.05    # |Δp| ≥ 5% para considerar evento
WINDOW_PRE  = 7          # días antes del cambio (para baseline)
WINDOW_POST = 7          # días después del cambio (para validar predicción)
MIN_OBS_PRE  = 4         # mínimo de obs en ventana pre
MIN_OBS_POST = 4         # mínimo de obs en ventana post

# ---------------------------------------------------------------------------
# 1. Cargar predicciones del LGBM con IC calibrado (CONFORMAL ya aplicado)
# ---------------------------------------------------------------------------
print("[1/4] Cargando predicciones del LGBM con IC calibrado…")
preds = pd.read_parquet(DATA / "predictions_conformal.parquet")
# columnas: fecha, cve_art, descripcion, fam, log_qty, qty_no_promo,
# q10, q90, q10_calib, q90_calib, q_adj, in_ci_orig, in_ci_calib

# Necesitamos también q50 (predicción central) — la sacamos de lgbm_predictions
lgb_pred = pd.read_parquet(DATA / "lgbm_predictions.parquet")[
    ["fecha","cve_art","yhat_q50","qty_hat"]]
preds = preds.merge(lgb_pred, on=["fecha","cve_art"], how="left")

# Y el precio (estaba en el panel)
panel = pd.read_parquet(DATA / "panel_diario.parquet")
panel["qty_no_promo"] = panel.qty_total - panel.qty_promo
preds = preds.merge(panel[["fecha","cve_art","precio_normal_sinpromo","costo_vigente"]],
                    on=["fecha","cve_art"], how="left")
preds = preds.sort_values(["cve_art","fecha"]).reset_index(drop=True)
print(f"   Predicciones disponibles: {len(preds):,}")

# ---------------------------------------------------------------------------
# 2. Detectar eventos de cambio de precio
# ---------------------------------------------------------------------------
print("\n[2/4] Detectando eventos de cambio de precio en test…")

events = []
for sku, g in preds.groupby("cve_art"):
    if len(g) < WINDOW_PRE + WINDOW_POST + 1:
        continue
    g = g.sort_values("fecha").reset_index(drop=True)
    # mediana móvil de 7 días previos (excluyendo el día t)
    g["p_pre_med"] = g.precio_normal_sinpromo.shift(1).rolling(WINDOW_PRE, min_periods=3).median()
    g["pct_chg"] = (g.precio_normal_sinpromo - g.p_pre_med) / g.p_pre_med
    cambios = g[g.pct_chg.abs() >= MIN_PCT_CHANGE].copy()

    for _, ev in cambios.iterrows():
        t = ev.fecha
        pre = g[(g.fecha < t) & (g.fecha >= t - pd.Timedelta(days=WINDOW_PRE))]
        post = g[(g.fecha >= t) & (g.fecha < t + pd.Timedelta(days=WINDOW_POST))]
        if len(pre) < MIN_OBS_PRE or len(post) < MIN_OBS_POST:
            continue
        # Verificar que precio post sea estable (no múltiples cambios en la ventana)
        if post.precio_normal_sinpromo.nunique() > 3:
            continue

        ev_row = {
            "cve_art": sku,
            "descripcion": ev.descripcion,
            "fam": ev.fam,
            "fec_cambio": t,
            "p_pre": float(pre.precio_normal_sinpromo.median()),
            "p_post": float(post.precio_normal_sinpromo.median()),
            "pct_chg_real": float(ev.pct_chg),
            # Demanda real
            "qty_real_pre":  float(pre.qty_no_promo.median()),
            "qty_real_post": float(post.qty_no_promo.median()),
            "qty_chg_real_pct": float((post.qty_no_promo.median() - pre.qty_no_promo.median()) /
                                       max(pre.qty_no_promo.median(), 0.1)),
            # Predicción del modelo
            "qty_pred_pre":  float(np.exp(pre.yhat_q50).median()),
            "qty_pred_post": float(np.exp(post.yhat_q50).median()),
            "qty_chg_pred_pct": float((np.exp(post.yhat_q50).median() - np.exp(pre.yhat_q50).median()) /
                                       max(np.exp(pre.yhat_q50).median(), 0.1)),
            # IC calibrado en post
            "q10_post": float(post.q10_calib.median()),
            "q90_post": float(post.q90_calib.median()),
            "in_ci_calib_post_rate": float(post.in_ci_calib.mean()),
            # Tamaños de ventana
            "n_pre": int(len(pre)),
            "n_post": int(len(post)),
        }
        events.append(ev_row)

ev_df = pd.DataFrame(events)

# Errores y métricas por evento
ev_df["mape_post"] = np.abs(ev_df.qty_pred_post - ev_df.qty_real_post) / np.maximum(ev_df.qty_real_post, 0.1) * 100
ev_df["bias_post"] = (ev_df.qty_pred_post - ev_df.qty_real_post) / np.maximum(ev_df.qty_real_post, 0.1) * 100
ev_df["direccion_correcta"] = np.sign(ev_df.qty_chg_real_pct) == np.sign(ev_df.qty_chg_pred_pct)
ev_df["es_suba"] = ev_df.pct_chg_real > 0

ev_df.to_parquet(DATA / "backtest_quasi_events.parquet", index=False)
print(f"   Eventos detectados: {len(ev_df):,}")
print(f"   SKUs con eventos:   {ev_df.cve_art.nunique():,}")

# ---------------------------------------------------------------------------
# 3. Métricas globales
# ---------------------------------------------------------------------------
print("\n[3/4] Calculando métricas globales y por familia…")

def resumen(df, label):
    return {
        "scope": label,
        "n_eventos": len(df),
        "n_skus": df.cve_art.nunique(),
        "mape_med": float(df.mape_post.median()),
        "mape_p25": float(df.mape_post.quantile(0.25)),
        "mape_p75": float(df.mape_post.quantile(0.75)),
        "bias_med": float(df.bias_post.median()),
        "direccion_acc_pct": float(df.direccion_correcta.mean() * 100),
        "cobertura_IC_post_rate": float(df.in_ci_calib_post_rate.mean() * 100),
    }

global_summary = resumen(ev_df, "GLOBAL")

# Por dirección del cambio
subas = ev_df[ev_df.es_suba]
bajadas = ev_df[~ev_df.es_suba]
print(f"\n— Resumen GLOBAL —")
for k, v in global_summary.items():
    if isinstance(v, float):
        print(f"  {k:25s}: {v:8.2f}")
    else:
        print(f"  {k:25s}: {v}")

print(f"\n— Subidas de precio (n={len(subas):,}) —")
print(f"  MAPE mediano:          {subas.mape_post.median():.1f}%")
print(f"  Bias mediano:          {subas.bias_post.median():+.1f}%")
print(f"  Dirección acertada:    {subas.direccion_correcta.mean()*100:.1f}%")

print(f"\n— Bajadas de precio (n={len(bajadas):,}) —")
print(f"  MAPE mediano:          {bajadas.mape_post.median():.1f}%")
print(f"  Bias mediano:          {bajadas.bias_post.median():+.1f}%")
print(f"  Dirección acertada:    {bajadas.direccion_correcta.mean()*100:.1f}%")

# Por familia
print(f"\n— Por familia (top 10 por #eventos) —")
fam_rows = []
for fam, gg in ev_df.groupby("fam"):
    if len(gg) < 10:
        continue
    fam_rows.append({**resumen(gg, fam), "fam": fam})
fam_df = pd.DataFrame(fam_rows).sort_values("n_eventos", ascending=False).head(15)
print(fam_df[["fam","n_eventos","n_skus","mape_med","bias_med","direccion_acc_pct","cobertura_IC_post_rate"]].to_string(index=False))

# Save summary
all_summary = pd.DataFrame([global_summary] + fam_rows)
all_summary.to_parquet(DATA / "backtest_quasi_summary.parquet", index=False)

# ---------------------------------------------------------------------------
# 4. Tabla de "credibilidad" por familia — para tarjeta de confianza
# ---------------------------------------------------------------------------
print("\n[4/4] Generando tabla de credibilidad por familia (para tarjeta de confianza)…")
# Para cada familia: MAPE en cambios + cobertura IC
fam_credibility = ev_df.groupby("fam").agg(
    n_eventos=("cve_art","count"),
    mape_med=("mape_post","median"),
    direccion_acc=("direccion_correcta","mean"),
    cobertura_IC=("in_ci_calib_post_rate","mean"),
).reset_index()
fam_credibility["direccion_acc"] *= 100
fam_credibility["cobertura_IC"] *= 100
fam_credibility.to_parquet(DATA / "credibilidad_por_familia.parquet", index=False)
print(f"   {len(fam_credibility):,} familias con métricas de credibilidad.")

print(f"\n— Mejores familias por dirección_acertada —")
print(fam_credibility.sort_values("direccion_acc", ascending=False).head(8).to_string(index=False))
print(f"\n— Peores familias por dirección_acertada —")
print(fam_credibility[fam_credibility.n_eventos>=10].sort_values("direccion_acc").head(5).to_string(index=False))

# Ejemplos: casos textbook
print(f"\n— 10 EVENTOS donde el modelo más acertó (MAPE más bajo) —")
ejex = ev_df.nsmallest(10, "mape_post")[
    ["cve_art","descripcion","fam","fec_cambio","pct_chg_real",
     "qty_real_post","qty_pred_post","mape_post"]]
print(ejex.to_string(index=False))
