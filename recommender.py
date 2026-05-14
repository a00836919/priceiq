"""Módulo recomendador: integra todos los modelos para responder
'dado un SKU y un nuevo costo del proveedor, ¿qué precio recomendar?'

Componentes:
  - predict_demand_curve(sku, new_cost, price_grid) → DataFrame con qty + IC por precio
  - recommend_price(sku, new_cost) → dict con precio óptimo y desglose
  - get_substitutes(sku, k=5) → top vecinos por embedding + canibalización esperada
  - get_promo_benchmarks(sku) → eventos históricos de promo de la familia

Usado tanto por Streamlit como por scripts.
"""
from __future__ import annotations
from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from gensim.models import KeyedVectors
import warnings
warnings.filterwarnings("ignore")

DATA = Path("data")
MODELS = Path("models")
LOCAL_DIR = MODELS / "local_lgbm"

# ---------------------------------------------------------------------------
# Carga única de artefactos (cached)
# ---------------------------------------------------------------------------
_cache = {}

def _load_all():
    if _cache:
        return _cache
    panel = pd.read_parquet(DATA / "panel_diario.parquet")
    panel["qty_no_promo"] = panel.qty_total - panel.qty_promo
    panel = panel[(panel.precio_normal_sinpromo.notna()) & (panel.precio_normal_sinpromo > 0)
                  & (panel.qty_no_promo > 0)
                  & panel.costo_vigente.notna() & (panel.costo_vigente > 0)].copy()
    panel["log_qty"] = np.log(panel.qty_no_promo)
    panel["log_p"] = np.log(panel.precio_normal_sinpromo)
    panel["log_c"] = np.log(panel.costo_vigente)
    panel = panel.sort_values(["cve_art","fecha"]).reset_index(drop=True)

    emb = pd.read_parquet(DATA / "sku_embeddings.parquet")
    emb_cols = [c for c in emb.columns if c.startswith("emb_")]

    # Lags y features
    g = panel.groupby("cve_art", group_keys=False)
    panel["log_qty_lag1"] = g.log_qty.shift(1)
    panel["log_qty_lag7"] = g.log_qty.shift(7)
    panel["log_qty_ma7"]  = g.log_qty.shift(1).rolling(7, min_periods=3).mean().reset_index(0, drop=True)
    panel["log_p_lag1"]   = g.log_p.shift(1)
    panel["margen_unit_lag1"] = g.margen_unit.shift(1)
    panel = panel.merge(emb[["cve_art"] + emb_cols], on="cve_art", how="left")
    panel[emb_cols] = panel[emb_cols].fillna(0.0)

    encoders = {}
    for c in ["lin","fam","marca","tipo_articulo","cve_pro","proveedor_ult"]:
        panel[c] = panel[c].fillna("__NA__").astype(str)
        le = LabelEncoder()
        panel[c + "_le"] = le.fit_transform(panel[c])
        encoders[c] = le
    panel["woy"] = panel.fecha.dt.isocalendar().week.astype(int)

    features_num = ["log_p","log_c","margen_unit","margen_pct",
                    "dias_desde_cambio_costo","frac_promo",
                    "log_qty_lag1","log_qty_lag7","log_qty_ma7","log_p_lag1","margen_unit_lag1",
                    "dow","mes","año","woy"]
    features_cat = ["lin_le","fam_le","marca_le","tipo_articulo_le","cve_pro_le","proveedor_ult_le"]
    features = features_num + features_cat + emb_cols

    # Modelos
    models = {
        "q10": lgb.Booster(model_file=str(MODELS / "lgbm_q10.txt")),
        "q50": lgb.Booster(model_file=str(MODELS / "lgbm_q50.txt")),
        "q90": lgb.Booster(model_file=str(MODELS / "lgbm_q90.txt")),
    }
    # Locales: cve_art -> Booster
    local_bs = {}
    for f in LOCAL_DIR.glob("*.txt"):
        local_bs[f.stem] = lgb.Booster(model_file=str(f))

    # Conformal quantiles
    qtab = pd.read_parquet(DATA / "conformal_quantiles.parquet")
    qsku = {r.clave: r.q for _, r in qtab[qtab.nivel=="sku"].iterrows()}
    qfam = {r.clave: r.q for _, r in qtab[qtab.nivel=="fam"].iterrows()}
    qglb = float(qtab[qtab.nivel=="global"].q.iloc[0])

    # Item2Vec
    try:
        kv = KeyedVectors.load(str(DATA / "item2vec_model.kv"))
    except Exception:
        kv = None

    # Cross-elasticities
    cross = pd.read_parquet(DATA / "cross_elasticities_v2.parquet")
    # Promo events
    promos = pd.read_parquet(DATA / "promo_events.parquet")
    # Elasticidad estructural
    elas = pd.read_parquet(DATA / "elasticidades_jerarquico_sku.parquet")
    # Factor_uds (caja → unidad) — el panel tiene costo POR CAJA; el cliente piensa en UNIDADES
    try:
        fac = pd.read_parquet(DATA / "sku_factor_uds.parquet")
        factor_uds = dict(zip(fac.cve_art, fac.factor_uds))
    except Exception:
        factor_uds = {}
    # Credibilidad por familia (del backtest quasi-experimental)
    try:
        cred = pd.read_parquet(DATA / "credibilidad_por_familia.parquet")
        cred_fam = {r.fam: r for _, r in cred.iterrows()}
    except Exception:
        cred_fam = {}

    _cache.update(dict(
        panel=panel, emb_cols=emb_cols, features=features, encoders=encoders,
        models=models, local_bs=local_bs,
        qsku=qsku, qfam=qfam, qglb=qglb,
        kv=kv, cross=cross, promos=promos, elas=elas,
        factor_uds=factor_uds, cred_fam=cred_fam,
    ))
    return _cache


def list_skus() -> pd.DataFrame:
    """Lista SKUs disponibles con metadata."""
    c = _load_all()
    df = c["panel"].drop_duplicates("cve_art")[["cve_art","descripcion","fam","lin","marca"]]
    return df.sort_values("descripcion").reset_index(drop=True)


def get_sku_state(sku: str) -> dict:
    """Estado actual del SKU. Costos convertidos a UNIDAD (dividiendo por factor_uds)."""
    c = _load_all()
    g = c["panel"][c["panel"].cve_art == sku].sort_values("fecha")
    if len(g) == 0:
        return None
    last = g.iloc[-1]
    fac = c["factor_uds"].get(sku, 1.0)
    elas_row = c["elas"][c["elas"].cve_art == sku]
    beta_est = float(elas_row.beta_i.iloc[0]) if len(elas_row) else None
    return {
        "cve_art": sku,
        "descripcion": last.descripcion,
        "fam": last.fam, "lin": last.lin, "marca": last.marca,
        "precio_actual": float(last.precio_normal_sinpromo),         # ya unitario
        "costo_actual_unitario": float(last.costo_vigente) / fac,    # convertir a unitario
        "costo_actual_caja": float(last.costo_vigente),              # de la fuente
        "factor_uds": fac,
        "qty_diaria_med": float(g.qty_no_promo.median()),
        "qty_diaria_p75": float(g.qty_no_promo.quantile(0.75)),
        "n_dias": int(len(g)),
        "fecha_ultima": last.fecha,
        "elasticidad_estructural": beta_est,
    }


def predict_demand_curve(sku: str, new_cost_unit: float,
                          price_grid: np.ndarray | None = None,
                          conservatism: float = 0.5) -> pd.DataFrame:
    """Curva precio→demanda con IC calibrado y penalización por extrapolación.

    new_cost_unit: POR UNIDAD (precio que paga la tienda al proveedor).
    conservatism ∈ [0, 1]:
        0   = recomendar puramente por modelo (riesgo de extrapolación)
        0.5 = balance (default)
        1   = obligar a quedar muy cerca del precio actual
    """
    c = _load_all()
    panel = c["panel"]
    g = panel[panel.cve_art == sku].sort_values("fecha")
    if len(g) == 0:
        raise ValueError(f"SKU {sku} no encontrado")
    base = g.iloc[-1].copy()
    fac = c["factor_uds"].get(sku, 1.0)
    new_cost_caja = new_cost_unit * fac

    p_actual = float(base.precio_normal_sinpromo)
    p_hist_min = float(g.precio_normal_sinpromo.quantile(0.05))
    p_hist_max = float(g.precio_normal_sinpromo.quantile(0.95))

    if price_grid is None:
        # Grid amplio para visualizar; la penalización guía el óptimo.
        lo = max(new_cost_unit * 1.05, p_hist_min * 0.80)
        hi = max(p_hist_max * 1.25, p_actual * 1.30, new_cost_unit * 1.50)
        if hi <= lo:
            hi = lo * 1.5
        price_grid = np.linspace(lo, hi, 50)

    rows = []
    for p in price_grid:
        r = base.copy()
        r["log_p"] = np.log(p)
        r["log_c"] = np.log(new_cost_caja)
        r["margen_unit"] = p - new_cost_unit
        r["margen_pct"] = (p - new_cost_unit) / max(p, 0.01)
        r["dias_desde_cambio_costo"] = 0
        r["frac_promo"] = 0.0
        next_date = base.fecha + pd.Timedelta(days=1)
        r["dow"] = next_date.dayofweek
        r["mes"] = next_date.month
        r["año"] = next_date.year
        r["woy"] = int(next_date.isocalendar().week)
        rows.append(r)
    df = pd.DataFrame(rows).reset_index(drop=True)

    # Predicciones q10, q50, q90
    X = df[c["features"]].values
    yhat_q10 = c["models"]["q10"].predict(X)
    yhat_q50 = c["models"]["q50"].predict(X)
    yhat_q90 = c["models"]["q90"].predict(X)

    # Local correction si existe
    if sku in c["local_bs"]:
        delta = c["local_bs"][sku].predict(X)
        yhat_q50 = yhat_q50 + delta
        # nota: q10/q90 no se ajustan localmente; el conformal abajo lo arreglará

    # Conformal: cuantil de calibración
    fam = base.fam
    q_adj = c["qsku"].get(sku, c["qfam"].get(fam, c["qglb"]))
    yhat_q10_calib = yhat_q10 - q_adj
    yhat_q90_calib = yhat_q90 + q_adj

    qty_med = np.exp(yhat_q50)
    qty_lo  = np.exp(yhat_q10_calib)
    qty_hi  = np.exp(yhat_q90_calib)
    margen_unit = price_grid - new_cost_unit

    # ── Penalización por extrapolación ────────────────────────────────
    # Peso de credibilidad = 1 dentro de [p_q05, p_q95] del histórico.
    # Cae cuadráticamente fuera. conservatism controla qué tan rápido cae.
    # σ_efectivo = (rango_histórico) · (2 - 1.8·conservatism) → para
    # conservatism=1, σ chico (caída rápida); para 0, σ grande (caída lenta).
    p_med = (p_hist_min + p_hist_max) / 2
    p_range = max(p_hist_max - p_hist_min, p_actual * 0.10)
    sigma_eff = p_range * (1.0 - 0.9 * conservatism) + 1e-6  # nunca cero
    dist_inner = np.maximum(price_grid - p_hist_max, p_hist_min - price_grid)
    dist_inner = np.maximum(dist_inner, 0.0)  # 0 dentro del rango
    # Penalización adicional por desviarse del precio actual cuando conservatism alto
    dist_actual = np.abs(price_grid - p_actual)
    sigma_actual = max(p_actual * 0.40 * (1.0 - 0.7 * conservatism), 0.01)
    weight = np.exp(-0.5 * (dist_inner / sigma_eff)**2) \
           * np.exp(-0.5 * (dist_actual / max(sigma_actual, p_range))**2 * conservatism)
    weight = np.clip(weight, 1e-6, 1.0)

    margen_med_raw = margen_unit * qty_med
    return pd.DataFrame({
        "precio": price_grid,
        "qty_med": qty_med,
        "qty_lo": qty_lo,
        "qty_hi": qty_hi,
        "ingreso_med": price_grid * qty_med,
        "ingreso_lo": price_grid * qty_lo,
        "ingreso_hi": price_grid * qty_hi,
        "margen_unit": margen_unit,
        "margen_med": margen_med_raw,           # SIN penalización (para mostrar)
        "margen_lo":  margen_unit * qty_lo,
        "margen_hi":  margen_unit * qty_hi,
        "credibilidad": weight,                  # 0-1: confianza en la predicción a ese precio
        "margen_med_penalizado": margen_med_raw * weight,  # usado para optimizar
        "ingreso_med_penalizado": price_grid * qty_med * weight,
    })


def recommend_price(sku: str, new_cost_unit: float, objetivo: str = "margen",
                     conservatism: float = 0.5) -> dict:
    """Recomienda precio óptimo (POR UNIDAD) usando margen PENALIZADO.

    conservatism ∈ [0,1] controla cuánto castigar la extrapolación.
    """
    curve = predict_demand_curve(sku, new_cost_unit, conservatism=conservatism)
    state = get_sku_state(sku)

    valid = curve[curve.precio >= new_cost_unit * 1.05]
    if objetivo == "margen":
        col = "margen_med_penalizado"
    elif objetivo == "ingreso":
        col = "ingreso_med_penalizado"
    else:
        col = "qty_med"

    best = valid.loc[valid[col].idxmax()]
    # Precio actual del SKU para comparar
    p_actual = state["precio_actual"]

    # Margen al precio actual (con el NUEVO costo)
    actual_row = curve.iloc[(curve.precio - p_actual).abs().argsort().iloc[0]]

    return {
        "precio_recomendado": float(best.precio),
        "qty_esperada": float(best.qty_med),
        "qty_lo": float(best.qty_lo),
        "qty_hi": float(best.qty_hi),
        "margen_unit": float(best.margen_unit),
        "margen_esperado_diario": float(best.margen_med),
        "ingreso_esperado_diario": float(best.ingreso_med),
        "credibilidad": float(best.credibilidad),
        "precio_actual": p_actual,
        "qty_al_precio_actual": float(actual_row.qty_med),
        "margen_al_precio_actual": float(actual_row.margen_med),
        "delta_margen_diario": float(best.margen_med - actual_row.margen_med),
        "objetivo": objetivo,
        "conservatism": conservatism,
        "curva": curve,
        "state": state,
    }


def recommended_range(rec: dict, threshold: float = 0.95,
                       min_credibilidad: float = 0.6) -> dict:
    """A partir de una recomendación, encuentra el RANGO de precios donde:
       - margen esperado ≥ threshold · margen_óptimo
       - credibilidad ≥ min_credibilidad

    Esto evita el sesgo de presentar un único punto como precisión espuria.
    """
    curve = rec["curva"].copy()
    best_margen = float(curve.margen_med_penalizado.max())
    if best_margen <= 0:
        return {"lo": rec["precio_recomendado"], "hi": rec["precio_recomendado"],
                "n_puntos": 1, "best_margen": best_margen}
    cond = ((curve.margen_med_penalizado >= threshold * best_margen)
            & (curve.credibilidad >= min_credibilidad))
    zona = curve[cond]
    if len(zona) == 0:
        return {"lo": rec["precio_recomendado"], "hi": rec["precio_recomendado"],
                "n_puntos": 1, "best_margen": best_margen}
    return {
        "lo": float(zona.precio.min()),
        "hi": float(zona.precio.max()),
        "n_puntos": int(len(zona)),
        "best_margen": best_margen,
        "margen_lo": float(zona.iloc[0].margen_med),
        "margen_hi": float(zona.iloc[-1].margen_med),
    }


def confidence_card(sku: str, rec: dict) -> dict:
    """Tarjeta de confianza para una recomendación.

    Cuatro componentes (0-1 cada uno):
      - credibilidad_modelo: penalización por extrapolación (ya calculada)
      - precision_familia:   1 - MAPE_familia (de backtest quasi-experimental)
      - cobertura_familia:   cobertura empírica del IC (de backtest)
      - direccion_familia:   tasa de acierto del signo de cambio
      - cobertura_sku:       n_obs del SKU en train (suficiencia de datos)
    Score final = promedio ponderado, mapeado a etiqueta.
    """
    c = _load_all()
    state = get_sku_state(sku)
    n_dias = state["n_dias"]
    fam = state["fam"]

    credib_model = float(rec["credibilidad"])
    fam_cred = c["cred_fam"].get(fam)
    if fam_cred is None:
        precision_fam = 0.5
        cobertura_fam = 0.5
        direccion_fam = 0.5
        n_eventos_fam = 0
    else:
        precision_fam = max(0.0, 1.0 - float(fam_cred.mape_med) / 100.0)
        cobertura_fam = float(fam_cred.cobertura_IC) / 100.0
        direccion_fam = float(fam_cred.direccion_acc) / 100.0
        n_eventos_fam = int(fam_cred.n_eventos)

    sufic_sku = min(1.0, n_dias / 500.0)

    # Score: ponderaciones que dan peso al modelo y a evidencia empírica de familia
    weights = {"credib": 0.30, "precision": 0.25, "cobertura": 0.20,
               "direccion": 0.15, "sufic": 0.10}
    score = (
        weights["credib"]   * credib_model +
        weights["precision"]* precision_fam +
        weights["cobertura"]* cobertura_fam +
        weights["direccion"]* direccion_fam +
        weights["sufic"]    * sufic_sku
    )

    if score >= 0.75:
        nivel, emoji = "ALTA", "🟢"
    elif score >= 0.55:
        nivel, emoji = "MEDIA", "🟡"
    else:
        nivel, emoji = "BAJA", "🔴"

    # Texto auto-generado para el cliente
    texto = (
        f"Confianza {nivel} ({score*100:.0f}/100). "
        f"Modelo entrenado con {n_dias} días de historia del producto. "
    )
    if n_eventos_fam >= 10:
        texto += (
            f"En la familia {fam}, evaluamos {n_eventos_fam} cambios de precio reales: "
            f"el modelo predijo cantidad con error mediano de {(1-precision_fam)*100:.1f}%, "
            f"acertó la dirección del cambio en {direccion_fam*100:.0f}% de los casos, "
            f"y su intervalo de confianza cubrió la realidad en {cobertura_fam*100:.0f}%. "
        )
    else:
        texto += f"Pocos eventos de cambio histórico en familia {fam} para validar — usar con criterio. "
    if credib_model < 0.5:
        texto += "⚠️ Precio recomendado está fuera del rango histórico observado del producto."

    return {
        "score": float(score),
        "nivel": nivel,
        "emoji": emoji,
        "texto": texto,
        "componentes": {
            "credibilidad_modelo": credib_model,
            "precision_familia": precision_fam,
            "cobertura_familia": cobertura_fam,
            "direccion_familia": direccion_fam,
            "suficiencia_sku": sufic_sku,
        },
        "n_eventos_fam": n_eventos_fam,
        "n_dias_sku": n_dias,
        "mape_familia_pct": (1 - precision_fam) * 100,
    }


def get_substitutes(sku: str, k: int = 5) -> pd.DataFrame:
    """Top-k sustitutos por similitud embedding + cruzada estimada si existe."""
    c = _load_all()
    if c["kv"] is None or sku not in c["kv"].key_to_index:
        return pd.DataFrame()
    sim = c["kv"].most_similar(sku, topn=k)
    rows = []
    panel = c["panel"]
    meta = panel.drop_duplicates("cve_art")[["cve_art","descripcion","fam","marca"]]
    for v, s in sim:
        m = meta[meta.cve_art == v]
        if len(m) == 0:
            continue
        # ¿hay β_ij cuantificado?
        cr = c["cross"][(c["cross"].focal == sku) & (c["cross"].vecino == v)]
        beta_ij = float(cr.beta_shrunk.iloc[0]) if len(cr) else np.nan
        ci_lo = float(cr.ci_lo_shrunk.iloc[0]) if len(cr) else np.nan
        ci_hi = float(cr.ci_hi_shrunk.iloc[0]) if len(cr) else np.nan
        tipo = str(cr.tipo_vecino.iloc[0]) if len(cr) else "sin_clasificar"
        rows.append({
            "vecino": v,
            "descripcion": m.descripcion.iloc[0],
            "fam": m.fam.iloc[0],
            "marca": m.marca.iloc[0],
            "similitud_embedding": s,
            "beta_cross_shrunk": beta_ij,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "tipo_vecino_canasta": tipo,
        })
    return pd.DataFrame(rows)


def estimate_canibalization(sku: str, new_price: float, p_actual: float | None = None) -> pd.DataFrame:
    """Para cada sustituto, estima cambio % en su demanda si pasamos del precio actual al nuevo.

    Usa β_ij_cross_shrunk si está disponible:
       Δlog(qty_vecino) = β_ij · Δlog(precio_focal)
       % cambio ≈ exp(Δlog(qty)) - 1
    """
    c = _load_all()
    state = get_sku_state(sku)
    if p_actual is None:
        p_actual = state["precio_actual"]
    if p_actual <= 0:
        return pd.DataFrame()
    delta_logp = np.log(new_price) - np.log(p_actual)

    subs = get_substitutes(sku, k=5)
    if len(subs) == 0:
        return subs
    subs["delta_qty_pct"] = np.where(
        subs.beta_cross_shrunk.notna(),
        (np.exp(subs.beta_cross_shrunk * delta_logp) - 1) * 100,
        np.nan
    )
    subs["delta_qty_pct_lo"] = (np.exp(subs.ci_lo * delta_logp) - 1) * 100
    subs["delta_qty_pct_hi"] = (np.exp(subs.ci_hi * delta_logp) - 1) * 100
    return subs


def get_promo_benchmarks(sku: str, k: int = 8) -> pd.DataFrame:
    """Promociones históricas del SKU + promedios de la familia."""
    c = _load_all()
    promos = c["promos"]
    state = get_sku_state(sku)
    if state is None:
        return pd.DataFrame()

    # Promos del SKU
    own = promos[promos.cve_art == sku].copy()
    own["scope"] = "este_sku"
    # Promos de la familia (para benchmark)
    fam_p = promos[(promos.fam == state["fam"]) & (promos.cve_art != sku)].copy()
    fam_p["scope"] = "misma_familia"

    res = pd.concat([own.nlargest(min(k, len(own)), "uplift_pct"),
                     fam_p.nlargest(min(k, len(fam_p)), "uplift_pct")], ignore_index=True)
    return res[["scope","cve_art","descripcion","NumProm","fec_min","fec_max",
                "dias_promo","descuento_pct","uplift_pct","qty_baseline_per_day",
                "qty_during_per_day"]]


if __name__ == "__main__":
    print("Cargando artefactos…")
    _load_all()
    skus = list_skus()
    print(f"  SKUs disponibles: {len(skus):,}")

    for test_sku in ["006913", "006989", "002439"]:
        print(f"\n{'='*70}\nSKU {test_sku}\n{'='*70}")
        s = get_sku_state(test_sku)
        print(f"  {s['descripcion']} ({s['fam']}, factor_uds={s['factor_uds']})")
        print(f"  precio_actual unit: ${s['precio_actual']:.2f}")
        print(f"  costo_actual unit:  ${s['costo_actual_unitario']:.2f}")
        print(f"  costo caja:         ${s['costo_actual_caja']:.2f}")
        print(f"  Elasticidad estructural: {s['elasticidad_estructural']:+.3f}")
        new_cost = s['costo_actual_unitario'] * 1.10
        rec = recommend_price(test_sku, new_cost, objetivo="margen")
        print(f"\n  Nuevo costo unit: ${new_cost:.2f} (+10%)")
        print(f"  Precio recomendado: ${rec['precio_recomendado']:.2f}")
        print(f"  Qty/día esperada:   {rec['qty_esperada']:.1f} (IC: [{rec['qty_lo']:.1f}, {rec['qty_hi']:.1f}])")
        print(f"  Margen unit:        ${rec['margen_unit']:.2f}")
        print(f"  Margen/día:         ${rec['margen_esperado_diario']:.2f}")
        print(f"  Δ vs precio actual: ${rec['delta_margen_diario']:+.2f}/día")
