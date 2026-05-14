"""Construye el dataset analítico canónico: fecha × sku con precio, costo, demanda.

Salidas en data/:
  - ventas_diarias.parquet  : agregado diario por SKU (precio, cantidad, promo)
  - costos_eventos.parquet  : log de eventos de cambio de costo (recepciones)
  - panel_diario.parquet    : panel completo fecha × sku con todo joinado y forward-fill
  - sku_catalog.parquet     : catálogo de SKUs con clasificación
"""
from pathlib import Path
import numpy as np
import pandas as pd

SRC = Path("tablas_merksyst_parquet")
OUT = Path("data")
OUT.mkdir(exist_ok=True)


def strip_str(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip()


# ---------------------------------------------------------------------------
# 1. VENTAS DIARIAS
# ---------------------------------------------------------------------------
print("[1/4] Procesando ventas diarias…")
det = pd.read_parquet(
    SRC / "pdvdet.parquet",
    columns=["FecOpe", "cve_art", "cantidad", "precio", "precio_normal",
             "tipo_promocion", "status"],
)
det["cve_art"] = strip_str(det.cve_art)
det["tp"] = strip_str(det.tipo_promocion)
det = det[(det.status == "01") & (det.cantidad > 0)].copy()
det["fecha"] = pd.to_datetime(det.FecOpe, format="%Y%m%d", errors="coerce")
det["es_promo"] = (det.tp != "")

# Agregado diario por SKU. Separamos sin-promo (precio limpio) y con-promo.
g = det.groupby(["fecha", "cve_art"])
ventas = g.agg(
    qty_total=("cantidad", "sum"),
    qty_promo=("cantidad", lambda s: s[det.loc[s.index, "es_promo"]].sum()),
    n_tickets=("cantidad", "size"),
    precio_min=("precio_normal", "min"),
    precio_max=("precio_normal", "max"),
).reset_index()

# Mediana de precio_normal sólo en líneas SIN promoción (precio "limpio")
no_promo = det[~det.es_promo]
precio_limpio = (
    no_promo.groupby(["fecha", "cve_art"])
    .precio_normal.median()
    .reset_index(name="precio_normal_sinpromo")
)
ventas = ventas.merge(precio_limpio, on=["fecha", "cve_art"], how="left")
ventas["frac_promo"] = ventas.qty_promo / ventas.qty_total

ventas.to_parquet(OUT / "ventas_diarias.parquet", index=False)
print(f"   → ventas_diarias.parquet: {len(ventas):,} filas, "
      f"{ventas.cve_art.nunique():,} SKUs, "
      f"{ventas.fecha.min().date()} → {ventas.fecha.max().date()}")


# ---------------------------------------------------------------------------
# 2. EVENTOS DE COSTO (cprrde ⨝ cprrme)
# ---------------------------------------------------------------------------
print("[2/4] Procesando eventos de costo del proveedor…")
rde = pd.read_parquet(
    SRC / "cprrde.parquet",
    columns=["ord", "n_rec", "cve_art", "uds_ent", "costo_rec", "CostoNeto"],
)
rde["cve_art"] = strip_str(rde.cve_art)
rde["ord"] = strip_str(rde.ord)
rde["n_rec"] = strip_str(rde.n_rec)

rme = pd.read_parquet(
    SRC / "cprrme.parquet",
    columns=["ord", "n_rec", "st", "proveedor", "fec_rec"],
)
rme["ord"] = strip_str(rme.ord)
rme["n_rec"] = strip_str(rme.n_rec)
rme["proveedor"] = strip_str(rme.proveedor)
# Sólo recepciones afectadas (st='11' = válida según value_counts)
rme = rme[rme.st == "11"]

# Join correcto por (ord, n_rec) para no inflar
costos = rde.merge(rme[["ord", "n_rec", "fec_rec", "proveedor"]],
                   on=["ord", "n_rec"], how="inner")
costos["fecha"] = pd.to_datetime(costos.fec_rec, format="%Y%m%d", errors="coerce")
costos = costos[costos.fecha.notna() & (costos.costo_rec > 0)].copy()
costos["uds_ent_num"] = pd.to_numeric(costos.uds_ent, errors="coerce")

eventos = costos[["fecha", "cve_art", "proveedor", "costo_rec",
                  "CostoNeto", "uds_ent_num", "ord", "n_rec"]].rename(
    columns={"CostoNeto": "costo_neto", "uds_ent_num": "uds_ent"}
)
eventos = eventos.sort_values(["cve_art", "fecha"]).reset_index(drop=True)
eventos.to_parquet(OUT / "costos_eventos.parquet", index=False)

print(f"   → costos_eventos.parquet: {len(eventos):,} eventos, "
      f"{eventos.cve_art.nunique():,} SKUs, "
      f"{eventos.proveedor.nunique():,} proveedores")


# ---------------------------------------------------------------------------
# 3. CATÁLOGO DE SKUs
# ---------------------------------------------------------------------------
print("[3/4] Construyendo catálogo de SKUs…")
iar = pd.read_parquet(
    SRC / "inviar.parquet",
    columns=["art", "lin", "s_lin", "fam", "s_fam", "marca", "des1",
             "tipo_articulo", "uds_cont", "moneda"],
)
iar["art"] = strip_str(iar.art)
for c in ["lin", "s_lin", "fam", "s_fam", "marca", "des1", "tipo_articulo"]:
    iar[c] = strip_str(iar[c])
iar = iar.rename(columns={"art": "cve_art", "des1": "descripcion"})

# invart: tomamos por art el row "principal" (alm con costo > 0, primero)
art = pd.read_parquet(
    SRC / "invart.parquet",
    columns=["art", "alm", "cve_pro", "fac_costo", "precio_vta0",
             "precio_lista", "costo_neto", "fec_cam_cos", "fec_cam_pre"],
)
art["art"] = strip_str(art.art)
art["cve_pro"] = strip_str(art.cve_pro)
art_agg = (
    art.sort_values(["art", "alm"])
    .groupby("art", as_index=False)
    .agg(cve_pro=("cve_pro", "first"),
         costo_actual=("fac_costo", "first"),
         precio_vta0=("precio_vta0", "first"),
         precio_lista=("precio_lista", "first"),
         costo_neto_snap=("costo_neto", "first"),
         fec_ult_cam_cos=("fec_cam_cos", "max"),
         fec_ult_cam_pre=("fec_cam_pre", "max"))
    .rename(columns={"art": "cve_art"})
)
catalog = iar.merge(art_agg, on="cve_art", how="left")
catalog.to_parquet(OUT / "sku_catalog.parquet", index=False)
print(f"   → sku_catalog.parquet: {len(catalog):,} SKUs")


# ---------------------------------------------------------------------------
# 4. PANEL DIARIO (fecha × sku) — el dataset canónico para modelado
# ---------------------------------------------------------------------------
print("[4/4] Construyendo panel diario fecha × sku con forward-fill de costo…")

# Universo: SKUs con al menos 1 venta válida
skus = ventas.cve_art.unique()
fechas = pd.date_range(ventas.fecha.min(), ventas.fecha.max(), freq="D")
print(f"   universo: {len(skus):,} SKUs × {len(fechas):,} días = "
      f"{len(skus)*len(fechas):,} celdas teóricas")

# Costo vigente por SKU-día: tomamos el último costo_rec registrado <= fecha
# (forward-fill desde el evento más reciente de cprrde)
costo_diario = (
    eventos.assign(fecha_d=eventos.fecha.dt.normalize())
    .sort_values(["cve_art", "fecha_d"])
    .groupby(["cve_art", "fecha_d"], as_index=False)
    .agg(costo_rec=("costo_rec", "median"),  # si hay varias recep el mismo día
         proveedor=("proveedor", "first"))
)

# Construcción del panel: solo días-SKU donde hubo venta (panel "denso de ventas")
# para no explotar el tamaño. Si quieres panel completo cambiar a producto cartesiano.
panel = ventas.copy()
panel["fecha_d"] = panel.fecha.dt.normalize()

# Para cada SKU, mergear con eventos de costo y forward-fill
def attach_costo(g: pd.DataFrame, eventos_sku: pd.DataFrame) -> pd.DataFrame:
    if len(eventos_sku) == 0:
        g["costo_vigente"] = np.nan
        g["proveedor_ult"] = pd.NA
        g["dias_desde_cambio_costo"] = np.nan
        return g
    eventos_sku = eventos_sku.sort_values("fecha_d")
    # merge_asof: para cada fecha de venta, el último evento con fecha<=
    out = pd.merge_asof(
        g.sort_values("fecha_d"),
        eventos_sku[["fecha_d", "costo_rec", "proveedor"]].rename(
            columns={"costo_rec": "costo_vigente",
                     "proveedor": "proveedor_ult",
                     "fecha_d": "fecha_evento"}),
        left_on="fecha_d", right_on="fecha_evento",
        direction="backward",
    )
    out["dias_desde_cambio_costo"] = (
        (out.fecha_d - out.fecha_evento).dt.days
    )
    return out.drop(columns=["fecha_evento"])

# Procesamos por SKU para mantener memoria controlada
print("   forward-filling costos…")
costo_por_sku = {sku: g for sku, g in costo_diario.groupby("cve_art")}
chunks = []
for sku, g in panel.groupby("cve_art"):
    chunks.append(attach_costo(g, costo_por_sku.get(sku, pd.DataFrame())))
panel_full = pd.concat(chunks, ignore_index=True)

# Adjuntar clasificación
panel_full = panel_full.merge(
    catalog[["cve_art", "descripcion", "lin", "s_lin", "fam", "s_fam",
             "marca", "tipo_articulo", "cve_pro"]],
    on="cve_art", how="left",
)

# Margen estimado y reordenar columnas
panel_full["margen_unit"] = panel_full.precio_normal_sinpromo - panel_full.costo_vigente
panel_full["margen_pct"] = (
    panel_full.margen_unit / panel_full.precio_normal_sinpromo
)

# Calendario
panel_full["dow"] = panel_full.fecha.dt.dayofweek
panel_full["mes"] = panel_full.fecha.dt.month
panel_full["año"] = panel_full.fecha.dt.year

cols = ["fecha", "año", "mes", "dow", "cve_art", "descripcion",
        "lin", "fam", "marca", "tipo_articulo", "cve_pro", "proveedor_ult",
        "qty_total", "qty_promo", "frac_promo", "n_tickets",
        "precio_normal_sinpromo", "precio_min", "precio_max",
        "costo_vigente", "dias_desde_cambio_costo",
        "margen_unit", "margen_pct"]
panel_full = panel_full[cols].sort_values(["cve_art", "fecha"])
panel_full.to_parquet(OUT / "panel_diario.parquet", index=False)

print(f"   → panel_diario.parquet: {len(panel_full):,} filas")
print(f"     SKUs:           {panel_full.cve_art.nunique():,}")
print(f"     Con costo NN:   {panel_full.costo_vigente.notna().sum():,} "
      f"({panel_full.costo_vigente.notna().mean()*100:.1f}%)")
print(f"     Con precio NN:  {panel_full.precio_normal_sinpromo.notna().sum():,} "
      f"({panel_full.precio_normal_sinpromo.notna().mean()*100:.1f}%)")
print(f"     Tamaño en disco: {(OUT/'panel_diario.parquet').stat().st_size/1e6:.1f} MB")

print("\n✓ Listo. Archivos en data/:")
for p in sorted(OUT.glob("*.parquet")):
    print(f"   {p.name}  ({p.stat().st_size/1e6:.1f} MB)")
