"""FastAPI backend que expone el recommender como API REST.

Endpoints:
  GET  /api/health                       — healthcheck
  GET  /api/skus                         — lista catálogo (paginado, búsqueda)
  GET  /api/skus/{sku}                   — estado actual del SKU
  POST /api/recommend                    — recomendación de precio
  POST /api/validate-cost                — validación del nuevo costo
  GET  /api/skus/{sku}/substitutes       — top sustitutos + canibalización
  GET  /api/skus/{sku}/promos            — promociones históricas
  GET  /api/backtest-summary             — métricas globales del backtest

Correr: uvicorn backend.main:app --reload --port 8000
"""
import os
import sys
import math
import json as _json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

def _clean_json(obj):
    """Reemplaza NaN/inf por None recursivamente para JSON compliant."""
    if isinstance(obj, dict):
        return {k: _clean_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_json(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

import recommender as R

app = FastAPI(
    title="PriceIQ API",
    description="API REST para recomendación de precio basada en costo del proveedor.",
    version="1.0.0",
)

# Application Insights (opcional — solo si APPLICATIONINSIGHTS_CONNECTION_STRING está set)
_ai_conn = os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING")
if _ai_conn:
    try:
        from azure.monitor.opentelemetry import configure_azure_monitor
        configure_azure_monitor(connection_string=_ai_conn)
        print("[telemetry] Application Insights habilitado")
    except ImportError:
        print("[telemetry] azure-monitor-opentelemetry no instalado — telemetría desactivada")
    except Exception as e:
        print(f"[telemetry] error al inicializar: {e}")
else:
    print("[telemetry] APPLICATIONINSIGHTS_CONNECTION_STRING no set — telemetría desactivada")

# CORS: configurable vía env var CORS_ORIGINS (separado por comas).
# Default: localhost para desarrollo. En producción: setear con la URL del frontend.
_default_origins = "http://localhost:3000,http://localhost:3001,http://127.0.0.1:3000"
_origins_env = os.environ.get("CORS_ORIGINS", _default_origins)
_origins = [o.strip() for o in _origins_env.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
print(f"[cors] orígenes permitidos: {_origins}")


# ---------------------------------------------------------------------------
# Carga inicial (cachea los artefactos al primer request)
# ---------------------------------------------------------------------------
@app.on_event("startup")
def startup():
    print("[startup] Pre-cargando artefactos del recommender…")
    R._load_all()
    print("[startup] OK — listo para requests.")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class RecommendRequest(BaseModel):
    sku: str
    new_cost: float = Field(..., gt=0, description="Nuevo costo por UNIDAD")
    conservatism: float = Field(0.5, ge=0, le=1)
    objetivo: str = Field("margen", pattern="^(margen|ingreso|cantidad)$")

class ValidateCostRequest(BaseModel):
    sku: str
    new_cost: float = Field(..., gt=0)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    """Healthcheck para Container Apps probes. Verifica que el cache esté listo."""
    cache = R._cache
    is_ready = bool(cache.get("models"))
    return {
        "status": "ok" if is_ready else "warming",
        "version": app.version,
        "ready": is_ready,
        "data_dir": str(R.DATA),
        "models_dir": str(R.MODELS),
        "n_skus": len(cache.get("panel", pd.DataFrame()).cve_art.unique()) if is_ready else 0,
    }


@app.get("/api/skus")
def list_skus(
    q: Optional[str] = Query(None, description="Filtro de texto"),
    fam: Optional[str] = Query(None),
    limit: int = Query(50, le=500),
    offset: int = 0,
):
    """Lista de SKUs con filtros, paginada."""
    df = R.list_skus()
    if q:
        ql = q.lower()
        df = df[df.descripcion.str.lower().str.contains(ql, na=False) |
                df.cve_art.str.contains(ql, na=False)]
    if fam:
        df = df[df.fam == fam]
    total = len(df)
    df = df.iloc[offset:offset + limit]
    return {
        "total": int(total),
        "items": df.to_dict(orient="records"),
        "limit": limit,
        "offset": offset,
    }


@app.get("/api/skus/{sku}")
def get_sku(sku: str):
    """Estado actual del SKU."""
    state = R.get_sku_state(sku)
    if state is None:
        raise HTTPException(404, f"SKU {sku} no encontrado")
    # Convertir Timestamps a string para JSON
    state["fecha_ultima"] = state["fecha_ultima"].isoformat()
    return state


@app.post("/api/recommend")
def recommend(req: RecommendRequest):
    """Recomendación completa: precio óptimo + rango + tarjeta de confianza."""
    try:
        rec = R.recommend_price(
            req.sku, req.new_cost,
            objetivo=req.objetivo,
            conservatism=req.conservatism,
        )
    except ValueError as e:
        raise HTTPException(404, str(e))

    rng  = R.recommended_range(rec, threshold=0.95, min_credibilidad=0.6)
    card = R.confidence_card(req.sku, rec)
    cost_check = R.validate_cost_input(req.sku, req.new_cost)

    curve = rec["curva"].copy()
    # Reducir tamaño del JSON: solo columnas que el frontend necesita
    curve_out = curve[["precio","qty_med","qty_lo","qty_hi",
                       "margen_med","margen_lo","margen_hi",
                       "margen_med_penalizado","credibilidad"]].to_dict(orient="records")

    state = rec["state"]
    # Limpiar timestamps
    state["fecha_ultima"] = state["fecha_ultima"].isoformat() if state.get("fecha_ultima") else None

    return {
        "precio_recomendado": rec["precio_recomendado"],
        "qty_esperada":       rec["qty_esperada"],
        "qty_lo":             rec["qty_lo"],
        "qty_hi":             rec["qty_hi"],
        "margen_unit":        rec["margen_unit"],
        "margen_esperado_diario": rec["margen_esperado_diario"],
        "ingreso_esperado_diario": rec["ingreso_esperado_diario"],
        "credibilidad":       rec["credibilidad"],
        "precio_actual":      rec["precio_actual"],
        "margen_al_precio_actual": rec["margen_al_precio_actual"],
        "qty_al_precio_actual": rec["qty_al_precio_actual"],
        "delta_margen_diario": rec["delta_margen_diario"],
        "modelo_usado":       rec["modelo_usado"],
        "objetivo":           rec["objetivo"],
        "conservatism":       rec["conservatism"],
        "rango": {
            "lo": rng["lo"], "hi": rng["hi"],
            "n_puntos": rng["n_puntos"],
        },
        "confianza": {
            "score":       card["score"],
            "nivel":       card["nivel"],
            "emoji":       card["emoji"],
            "texto":       card["texto"],
            "componentes": card["componentes"],
            "n_eventos_fam": card["n_eventos_fam"],
            "n_dias_sku":  card["n_dias_sku"],
        },
        "cost_check": {
            "severity": cost_check["severity"],
            "is_outlier": cost_check["is_outlier"],
            "message": cost_check["message"],
            "delta_pct": cost_check["delta_pct"],
            "costo_actual": cost_check["costo_actual"],
            "costo_hist_min": cost_check["costo_hist_min"],
            "costo_hist_max": cost_check["costo_hist_max"],
        },
        "state": state,
        "curve": curve_out,
    }


@app.post("/api/validate-cost")
def validate_cost(req: ValidateCostRequest):
    out = R.validate_cost_input(req.sku, req.new_cost)
    return out


@app.get("/api/skus/{sku}/substitutes")
def substitutes(sku: str, k: int = 5,
                new_price: Optional[float] = Query(None, gt=0,
                    description="Si se pasa, calcula canibalización esperada al cambiar al nuevo precio"),
                ):
    """Top-k sustitutos por similitud Item2Vec + β_ij + (opcional) canibalización
    esperada si pasamos del precio actual al new_price.
    """
    if new_price is not None:
        df = R.estimate_canibalization(sku, new_price)
    else:
        df = R.get_substitutes(sku, k=k)
    if df is None or len(df) == 0:
        return {"items": [], "p_actual": None}
    state = R.get_sku_state(sku)
    # to_dict + clean NaN/inf
    items = _clean_json(df.to_dict(orient="records"))
    return {
        "items": items,
        "p_actual": state["precio_actual"] if state else None,
        "fam_focal": state["fam"] if state else None,
        "desc_focal": state["descripcion"] if state else None,
    }


@app.get("/api/skus/{sku}/promos")
def promos(sku: str, limit: int = 20):
    df = R.get_promo_benchmarks(sku, k=limit)
    if df is None or len(df) == 0:
        return {"items": []}
    df = df.head(limit).copy()
    if "fec_min" in df.columns:
        df["fec_min"] = df.fec_min.astype(str)
        df["fec_max"] = df.fec_max.astype(str)
    return {"items": df.to_dict(orient="records")}


# ---------------------------------------------------------------------------
# RECOMENDACIÓN POR LOTES
# ---------------------------------------------------------------------------
class BatchItem(BaseModel):
    sku: str
    new_cost: float = Field(..., gt=0)

class BatchRequest(BaseModel):
    items: list[BatchItem]
    conservatism: float = Field(0.5, ge=0, le=1)
    objetivo: str = Field("margen", pattern="^(margen|ingreso|cantidad)$")


@app.post("/api/recommend-batch")
def recommend_batch(req: BatchRequest):
    """Procesa una lista de (sku, new_cost) y devuelve recomendaciones masivas.

    El frontend luego puede exportar la tabla a CSV para que el cliente actúe.
    """
    if len(req.items) > 5000:
        raise HTTPException(400, "Máximo 5,000 items por request")

    out_rows = []
    n_ok = 0
    n_err = 0
    margen_total = 0.0
    margen_total_actual = 0.0
    score_buckets = {"ALTA": 0, "MEDIA": 0, "BAJA": 0}
    severity_buckets = {"ok": 0, "warning": 0, "extreme": 0}

    for item in req.items:
        row = {"sku": item.sku, "new_cost": item.new_cost}
        try:
            rec = R.recommend_price(item.sku, item.new_cost,
                                     objetivo=req.objetivo,
                                     conservatism=req.conservatism)
            rng = R.recommended_range(rec)
            card = R.confidence_card(item.sku, rec)
            cost_check = R.validate_cost_input(item.sku, item.new_cost)

            row.update({
                "descripcion": rec["state"]["descripcion"],
                "fam": rec["state"]["fam"],
                "marca": rec["state"]["marca"],
                "precio_actual": rec["precio_actual"],
                "costo_actual": rec["state"]["costo_actual_unitario"],
                "precio_recomendado": rec["precio_recomendado"],
                "rango_lo": rng["lo"],
                "rango_hi": rng["hi"],
                "delta_precio_pct": (rec["precio_recomendado"] / rec["precio_actual"] - 1) * 100,
                "qty_esperada": rec["qty_esperada"],
                "qty_lo": rec["qty_lo"],
                "qty_hi": rec["qty_hi"],
                "margen_esperado": rec["margen_esperado_diario"],
                "margen_actual": rec["margen_al_precio_actual"],
                "delta_margen": rec["delta_margen_diario"],
                "credibilidad": rec["credibilidad"],
                "confianza_score": card["score"],
                "confianza_nivel": card["nivel"],
                "modelo_usado": rec["modelo_usado"],
                "cost_severity": cost_check["severity"],
                "cost_message": cost_check["message"],
                "status": "ok",
                "error": None,
            })
            n_ok += 1
            margen_total += rec["margen_esperado_diario"]
            margen_total_actual += rec["margen_al_precio_actual"]
            score_buckets[card["nivel"]] = score_buckets.get(card["nivel"], 0) + 1
            severity_buckets[cost_check["severity"]] = severity_buckets.get(cost_check["severity"], 0) + 1
        except Exception as e:
            row.update({
                "status": "error",
                "error": str(e),
                "descripcion": None, "fam": None, "marca": None,
                "precio_actual": None, "costo_actual": None,
                "precio_recomendado": None, "rango_lo": None, "rango_hi": None,
                "delta_precio_pct": None, "qty_esperada": None, "qty_lo": None, "qty_hi": None,
                "margen_esperado": None, "margen_actual": None, "delta_margen": None,
                "credibilidad": None, "confianza_score": None, "confianza_nivel": None,
                "modelo_usado": None, "cost_severity": None, "cost_message": None,
            })
            n_err += 1
        out_rows.append(row)

    return {
        "items": out_rows,
        "summary": {
            "n_total": len(req.items),
            "n_ok": n_ok,
            "n_err": n_err,
            "margen_total_recomendado": margen_total,
            "margen_total_actual": margen_total_actual,
            "delta_margen_total": margen_total - margen_total_actual,
            "delta_margen_pct": (margen_total / margen_total_actual - 1) * 100
                if margen_total_actual > 0 else 0.0,
            "confianza_buckets": score_buckets,
            "cost_severity_buckets": severity_buckets,
        }
    }


@app.get("/api/backtest-summary")
def backtest_summary():
    """Métricas globales para el footer de "¿por qué confiar?"."""
    try:
        ev = pd.read_parquet(R.DATA / "backtest_quasi_events.parquet")
        return {
            "n_eventos": int(len(ev)),
            "n_skus": int(ev.cve_art.nunique()),
            "mape_mediano": float(ev.mape_post.median()),
            "cobertura_ic80": float(ev.in_ci_calib_post_rate.mean() * 100),
            "direccion_acertada": float(ev.direccion_correcta.mean() * 100),
            "subidas": int(ev.es_suba.sum()),
            "bajadas": int((~ev.es_suba).sum()),
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/families")
def list_families():
    """Lista de familias con # SKUs y modelo enrutado."""
    c = R._load_all()
    panel = c["panel"]
    fam_n = panel.groupby("fam").cve_art.nunique().reset_index(name="n_skus")
    routing = c.get("fam_routing", {})
    fam_n["modelo"] = fam_n.fam.map(routing).fillna("LightGBM")
    return {"items": fam_n.sort_values("n_skus", ascending=False).to_dict(orient="records")}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
