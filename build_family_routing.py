"""Construye la tabla de routing fam → mejor_modelo para el ensemble por familia.

Lógica:
  1. Para cada (familia, modelo) calcular MAPE mediano en el test set.
  2. Para cada familia: best_model = argmin(mape).
  3. Si la familia tiene <5 SKUs, fallback a LightGBM.
  4. Si la mejora del best_model sobre LightGBM es <2pp, también fallback a LGBM
     (preferimos LGBM porque tiene IC bien calibrados).
  5. Saltamos CatBoost para SKUs donde necesitamos IC (su IC está mal calibrado),
     pero lo usamos para predicción central si gana.

Salida:
  data/family_model_routing.parquet — un row por familia con:
    fam, best_model (lgbm/xgb/cat), n_skus, mape_winner, mape_lgbm, gain_vs_lgbm
"""
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path("data")
MIN_GAIN_PP = 2.0    # mejora mínima sobre LGBM para cambiar (puntos %)
MIN_SKUS    = 5      # familias con menos SKUs → fallback LGBM

# ---------------------------------------------------------------------------
# 1. Cargar comparación por SKU y meta de familias
# ---------------------------------------------------------------------------
per_sku = pd.read_parquet(DATA / "model_comparison_by_sku.parquet")
panel   = pd.read_parquet(DATA / "panel_diario.parquet")

fam_map = panel.drop_duplicates("cve_art")[["cve_art", "fam"]]
per_sku = per_sku.merge(fam_map, on="cve_art", how="left")

print(f"SKUs evaluados: {len(per_sku):,}")
print(f"Familias presentes: {per_sku.fam.nunique():,}")

# ---------------------------------------------------------------------------
# 2. MAPE por (familia, modelo)
# ---------------------------------------------------------------------------
MODELOS = ["LightGBM", "XGBoost", "CatBoost", "MLP", "Stacking"]
mape_cols = [f"mape_{m}" for m in MODELOS]

agg = per_sku.groupby("fam").agg(
    n_skus=("cve_art", "count"),
    **{f"mape_{m}": (f"mape_{m}", "median") for m in MODELOS},
).reset_index()

# Excluir MLP y Stacking del ensemble — preferimos modelos con quantile nativo (LGBM/XGB/CAT)
ENS_MODELS = ["LightGBM", "XGBoost", "CatBoost"]

def pick_best(row):
    mapes = {m: row[f"mape_{m}"] for m in ENS_MODELS}
    best_model = min(mapes, key=mapes.get)
    best_mape  = mapes[best_model]
    lgb_mape   = mapes["LightGBM"]
    gain = lgb_mape - best_mape  # positivo = mejora vs LGBM
    if row.n_skus < MIN_SKUS or gain < MIN_GAIN_PP:
        return pd.Series({
            "best_model": "LightGBM",
            "mape_winner": lgb_mape,
            "gain_vs_lgbm": 0.0,
            "razon": "fallback_pocos_SKUs" if row.n_skus < MIN_SKUS else "mejora_marginal"
        })
    return pd.Series({
        "best_model": best_model,
        "mape_winner": best_mape,
        "gain_vs_lgbm": gain,
        "razon": "mejor_que_lgbm"
    })

best = agg.apply(pick_best, axis=1)
routing = pd.concat([agg, best], axis=1)
routing["mape_lgbm"] = routing["mape_LightGBM"]
routing = routing[["fam","n_skus","best_model","mape_winner","mape_lgbm",
                    "gain_vs_lgbm","razon"]].sort_values("n_skus", ascending=False)

routing.to_parquet(DATA / "family_model_routing.parquet", index=False)

# ---------------------------------------------------------------------------
# 3. Reporte
# ---------------------------------------------------------------------------
print("\n" + "="*80, "\nROUTING POR FAMILIA — RESUMEN\n", "="*80)
print(f"\nFamilias totales: {len(routing):,}")
print(f"\nDistribución de modelos elegidos:")
print(routing.best_model.value_counts().to_string())

print(f"\nDistribución de razón de elección:")
print(routing.razon.value_counts().to_string())

# Familias que sí ganaron con un modelo distinto
ganadoras = routing[routing.best_model != "LightGBM"]
if len(ganadoras) > 0:
    print(f"\n— Familias donde XGB o CAT vencen a LGBM por ≥{MIN_GAIN_PP}pp —")
    print(f"  ({len(ganadoras)} familias, ganancia mediana {ganadoras.gain_vs_lgbm.median():.1f}pp)")
    print(ganadoras.head(15).to_string(index=False))

# SKUs cubiertos por modelo no-LGBM
sku_routing = per_sku[["cve_art","fam"]].merge(
    routing[["fam","best_model"]], on="fam", how="left")
cobertura = sku_routing.best_model.value_counts()
print(f"\nSKUs cubiertos por cada modelo (post-routing):")
for m, n in cobertura.items():
    print(f"  {m:10s}: {n:5d} SKUs ({n/len(sku_routing)*100:.1f}%)")

# Mejora global esperada
mejora_global = (per_sku.merge(routing[["fam","best_model"]], on="fam")
                  .apply(lambda r: r[f"mape_{r['best_model']}"], axis=1).median())
mape_solo_lgbm = per_sku.mape_LightGBM.median()
print(f"\n— Mejora global esperada —")
print(f"  MAPE mediano sólo con LightGBM:           {mape_solo_lgbm:.2f}%")
print(f"  MAPE mediano con routing por familia:     {mejora_global:.2f}%")
print(f"  Mejora:                                   {mape_solo_lgbm - mejora_global:+.2f}pp")
