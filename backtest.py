"""Backtesting: ¿qué hubiera pasado si la herramienta se hubiera usado en los últimos 60 días?

Lógica:
  Para cada (SKU, día) en el test set:
    1. costo_real_dia = costo vigente del proveedor ese día
    2. p_real, qty_real, margen_real = observados
    3. p_recomendado = recommend_price(sku, costo_real_dia, conservatism)
    4. qty_predicha_real = predicción del modelo al precio REAL (test del modelo)
    5. qty_predicha_rec = predicción del modelo al precio RECOMENDADO (counterfactual)
    6. margen_rec_modelo  = (p_rec - costo) · qty_predicha_rec
    7. margen_real_modelo = (p_real - costo) · qty_predicha_real

Comparaciones honestas:
  A. ¿El modelo cree que su recomendación mejora?
       Δ_modelo = margen_rec_modelo − margen_real_modelo  (sí siempre, por construcción)
  B. ¿Cuánto difieren predicción y realidad? (sanity check)
       error_modelo = margen_real_modelo − margen_real_real
  C. Métrica honesta: si confiamos en el modelo, ¿cuánto ganaríamos?
       ganancia_si_modelo_correcto = margen_rec_modelo − margen_real_real

Limitaciones: NO conocemos la demanda real al precio recomendado (no se observó).
La ganancia es del modelo, no observada. Acotada por la incertidumbre del modelo
(MAPE de ~29% en cuantil mediano).

Salidas:
  data/backtest_results.parquet   — un row por (sku, día) en test con métricas
  data/backtest_summary.parquet   — agregado por SKU y por familia
"""
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

import recommender as R

DATA = Path("data")
CONSERVATISM = 0.5     # nivel para el backtest
N_SKUS_BACKTEST = 1500 # sólo los top SKUs por volumen para ahorrar tiempo

# ---------------------------------------------------------------------------
# 1. Identificar test set y cargar
# ---------------------------------------------------------------------------
print("Cargando datos…")
R._load_all()
panel = R._cache["panel"]
features = R._cache["features"]
fac_map  = R._cache["factor_uds"]
q50      = R._cache["models"]["q50"]
local_bs = R._cache["local_bs"]
qsku     = R._cache["qsku"]
qfam     = R._cache["qfam"]
qglb     = R._cache["qglb"]

# Construir test set como en lgbm_demand
panel_m = panel.dropna(subset=["log_qty_lag1","log_qty_lag7"]).copy()
fecha_max = panel_m.fecha.max()
fecha_test = fecha_max - pd.Timedelta(days=60)
test = panel_m[panel_m.fecha >= fecha_test].copy()
print(f"Test set: {len(test):,} filas, {test.cve_art.nunique():,} SKUs")

# Filtrar a SKUs candidatos (con elasticidad estructural válida)
elas = R._cache["elas"]
valid_skus = set(elas[(elas.beta_i < 0) & (elas.beta_i > -5)].cve_art)
test = test[test.cve_art.isin(valid_skus)]

# Top N por volumen en test
vol_per_sku = test.groupby("cve_art").qty_no_promo.sum().nlargest(N_SKUS_BACKTEST).index
test = test[test.cve_art.isin(vol_per_sku)]
print(f"Backtesting sobre {test.cve_art.nunique():,} SKUs (top por volumen en test), "
      f"{len(test):,} (SKU,día) observaciones")


# ---------------------------------------------------------------------------
# 2. Para cada SKU: encontrar precio recomendado a costo de cada día y comparar
# ---------------------------------------------------------------------------
def recommend_at_cost(sku, costo_caja, base_row):
    """Recomendación rápida sin recargar artefactos."""
    fac = fac_map.get(sku, 1.0)
    new_cost_unit = costo_caja / fac

    # Reusar misma lógica de predict_demand_curve pero in-line para velocidad
    g_hist = panel[panel.cve_art == sku].sort_values("fecha")
    p_actual = float(g_hist.precio_normal_sinpromo.iloc[-1])
    p_hist_min = float(g_hist.precio_normal_sinpromo.quantile(0.05))
    p_hist_max = float(g_hist.precio_normal_sinpromo.quantile(0.95))
    lo = max(new_cost_unit * 1.05, p_hist_min * 0.80)
    hi = max(p_hist_max * 1.25, p_actual * 1.30, new_cost_unit * 1.50)
    if hi <= lo: hi = lo * 1.5
    price_grid = np.linspace(lo, hi, 30)  # más chico para velocidad

    # Build feature matrix
    rows = []
    next_date = base_row.fecha + pd.Timedelta(days=0)  # mismo día del test
    for p in price_grid:
        r = base_row.copy()
        r["log_p"] = np.log(p)
        r["log_c"] = np.log(costo_caja)
        r["margen_unit"] = p - new_cost_unit
        r["margen_pct"] = (p - new_cost_unit) / max(p, 0.01)
        r["dias_desde_cambio_costo"] = 0
        r["frac_promo"] = 0.0
        rows.append(r)
    Xrec = pd.DataFrame(rows)[features].values
    yhat = q50.predict(Xrec)
    if sku in local_bs:
        yhat = yhat + local_bs[sku].predict(Xrec)
    qty = np.exp(yhat)

    # Penalización extrapolación
    p_range = max(p_hist_max - p_hist_min, p_actual * 0.10)
    sigma_eff = p_range * (1.0 - 0.9 * CONSERVATISM) + 1e-6
    sigma_actual = max(p_actual * 0.40 * (1.0 - 0.7 * CONSERVATISM), 0.01)
    dist_inner = np.maximum(np.maximum(price_grid - p_hist_max, p_hist_min - price_grid), 0.0)
    dist_actual = np.abs(price_grid - p_actual)
    weight = np.exp(-0.5 * (dist_inner / sigma_eff)**2) \
           * np.exp(-0.5 * (dist_actual / max(sigma_actual, p_range))**2 * CONSERVATISM)
    margen_unit = price_grid - new_cost_unit
    margen_pen  = margen_unit * qty * weight
    # piso: no recomendar bajo costo
    valid = price_grid >= new_cost_unit * 1.05
    if not valid.any():
        return p_actual, float(qty[len(qty)//2])
    idx_best = np.where(valid, margen_pen, -np.inf).argmax()
    return float(price_grid[idx_best]), float(qty[idx_best])


# ---------------------------------------------------------------------------
# 3. Recorrer test set
# ---------------------------------------------------------------------------
print("\nCorriendo backtest…")
results = []
for sku in tqdm(sorted(test.cve_art.unique()), desc="SKUs"):
    g_te = test[test.cve_art == sku]
    if len(g_te) == 0:
        continue
    fac = fac_map.get(sku, 1.0)
    # Predicción del modelo al precio REAL (para comparar)
    X_real = g_te[features].values
    yhat_real = q50.predict(X_real)
    if sku in local_bs:
        yhat_real = yhat_real + local_bs[sku].predict(X_real)
    qty_pred_real = np.exp(yhat_real)

    # Para cada día, recomendar al costo de ese día
    for i, (_, row) in enumerate(g_te.iterrows()):
        p_real = float(row.precio_normal_sinpromo)
        qty_real = float(row.qty_no_promo)
        costo_caja = float(row.costo_vigente)
        costo_unit = costo_caja / fac
        margen_real_real = (p_real - costo_unit) * qty_real

        p_rec, qty_rec = recommend_at_cost(sku, costo_caja, row)
        margen_real_modelo = (p_real - costo_unit) * qty_pred_real[i]
        margen_rec_modelo  = (p_rec - costo_unit) * qty_rec

        results.append({
            "fecha": row.fecha, "cve_art": sku, "fam": row.fam,
            "descripcion": row.descripcion,
            "costo_unit": costo_unit,
            "p_real": p_real, "qty_real": qty_real,
            "p_rec": p_rec,
            "qty_pred_real": float(qty_pred_real[i]),
            "qty_pred_rec":  qty_rec,
            "margen_real_obs":    margen_real_real,
            "margen_real_modelo": margen_real_modelo,
            "margen_rec_modelo":  margen_rec_modelo,
        })

bt = pd.DataFrame(results)
bt["delta_modelo"] = bt.margen_rec_modelo - bt.margen_real_modelo   # ganancia interna
bt["error_modelo"] = bt.margen_real_modelo - bt.margen_real_obs     # bias predicción
bt["ganancia_si_modelo_correcto"] = bt.margen_rec_modelo - bt.margen_real_obs
bt.to_parquet(DATA / "backtest_results.parquet", index=False)


# ---------------------------------------------------------------------------
# 4. Resumen
# ---------------------------------------------------------------------------
print("\n" + "="*80, "\nRESULTADOS DEL BACKTEST\n", "="*80)
print(f"\nObservaciones: {len(bt):,}  SKUs: {bt.cve_art.nunique():,}")

# Totales
tot_obs = bt.margen_real_obs.sum()
tot_modelo = bt.margen_real_modelo.sum()
tot_rec = bt.margen_rec_modelo.sum()
print(f"\nMargen total en los 60 días (suma sobre SKU·día):")
print(f"  REAL observado:                       ${tot_obs:>14,.0f}")
print(f"  Modelo al precio REAL (predicción):   ${tot_modelo:>14,.0f}   "
      f"(error modelo: {(tot_modelo/tot_obs-1)*100:+.1f}%)")
print(f"  Modelo al precio RECOMENDADO:         ${tot_rec:>14,.0f}")
print(f"\nGanancia teórica del recomendador (si el modelo fuera correcto):")
print(f"  Δ vs precio real observado:           ${tot_rec - tot_obs:>+14,.0f}  "
      f"({(tot_rec/tot_obs-1)*100:+.1f}%)")
print(f"  Δ vs precio real predicho (intra-modelo):  ${tot_rec - tot_modelo:>+14,.0f}  "
      f"({(tot_rec/tot_modelo-1)*100:+.1f}%)")

# Porcentaje de casos donde recomendador propone subir vs bajar vs igual
bt["accion"] = np.where(bt.p_rec > bt.p_real * 1.005, "subir",
                np.where(bt.p_rec < bt.p_real * 0.995, "bajar", "igual"))
print(f"\nDistribución de acciones recomendadas:")
print(bt.accion.value_counts().to_string())

print(f"\nCambio % de precio recomendado vs real (mediana):")
bt["pct_change"] = (bt.p_rec / bt.p_real - 1) * 100
print(f"  Mediana: {bt.pct_change.median():+.1f}%")
print(f"  P25-P75: [{bt.pct_change.quantile(0.25):+.1f}, {bt.pct_change.quantile(0.75):+.1f}]")

# Por SKU
print(f"\n— TOP 15 SKUs donde el recomendador agregaría MÁS margen —")
sku_agg = bt.groupby(["cve_art","descripcion","fam"]).agg(
    n=("fecha","size"),
    margen_real=("margen_real_obs","sum"),
    margen_rec=("margen_rec_modelo","sum"),
).reset_index()
sku_agg["ganancia"] = sku_agg.margen_rec - sku_agg.margen_real
sku_agg["pct_gain"] = sku_agg.ganancia / sku_agg.margen_real.replace(0, np.nan) * 100
top = sku_agg.nlargest(15, "ganancia")
print(top[["cve_art","descripcion","fam","n","margen_real","margen_rec","ganancia","pct_gain"]].to_string(index=False))

# Por familia
print(f"\n— Resumen por FAMILIA (top 10 por volumen de margen) —")
fam_agg = bt.groupby("fam").agg(
    n_obs=("fecha","size"),
    n_skus=("cve_art","nunique"),
    margen_real=("margen_real_obs","sum"),
    margen_rec=("margen_rec_modelo","sum"),
).reset_index()
fam_agg["ganancia"] = fam_agg.margen_rec - fam_agg.margen_real
fam_agg["pct_gain"] = fam_agg.ganancia / fam_agg.margen_real.replace(0, np.nan) * 100
print(fam_agg.nlargest(10, "margen_real").to_string(index=False))

fam_agg.to_parquet(DATA / "backtest_summary.parquet", index=False)
print(f"\n✓ Resultados guardados en data/backtest_results.parquet y backtest_summary.parquet")
