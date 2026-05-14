"""Análisis de canastas: co-ocurrencia, lift, sustitutos y complementos.

Pipeline:
  1. Construir tickets desde pdvhdr ⨝ pdvdet (set de SKUs por canasta)
  2. Matriz dispersa ticket × SKU → co-ocurrencia = M.T @ M
  3. Métricas por par: support_A, support_B, support_AB, lift, jaccard
  4. Por SKU: top-K complementos (lift alto, cualquier familia)
              top-K sustitutos  (misma familia, lift < 1, alta sustituibilidad)

Salidas en data/:
  - tickets.parquet              — un row por ticket con set de SKUs
  - sku_pairs.parquet            — pares con métricas (filtrados por support)
  - sku_neighbors.parquet        — top-5 sustitutos y top-5 complementos por SKU
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

SRC = Path("tablas_merksyst_parquet")
DATA = Path("data")

MIN_TICKETS_SKU = 100   # un SKU debe aparecer en al menos N tickets
MIN_PAIR_COUNT = 20     # un par debe aparecer en al menos N tickets juntos
TOP_K = 5

# ---------------------------------------------------------------------------
# 1. TICKETS: una fila por (Suc, FecOpe, Caja, NumTra) con set de SKUs
# ---------------------------------------------------------------------------
print("[1/4] Construyendo tickets…")
hdr = pd.read_parquet(SRC / "pdvhdr.parquet",
                      columns=["Suc","FecOpe","Caja","NumTra","status","subtotal"])
hdr["status"] = hdr.status.str.strip()
hdr = hdr[hdr.status == "TI"].copy()
hdr["ticket_id"] = (hdr.Suc.str.strip() + "|" + hdr.FecOpe.str.strip() + "|"
                    + hdr.Caja.str.strip() + "|" + hdr.NumTra.str.strip())
print(f"   tickets válidos (status=TI): {len(hdr):,}")

det = pd.read_parquet(SRC / "pdvdet.parquet",
                      columns=["Suc","FecOpe","Caja","NumTra","cve_art",
                               "cantidad","status"])
det = det[(det.status == "01") & (det.cantidad > 0)].copy()
det["cve_art"] = det.cve_art.str.strip()
det["ticket_id"] = (det.Suc.str.strip() + "|" + det.FecOpe.str.strip() + "|"
                    + det.Caja.str.strip() + "|" + det.NumTra.str.strip())

# sólo tickets que existen en hdr válido
det = det[det.ticket_id.isin(hdr.ticket_id)]

# set de SKUs únicos por ticket
tickets = det.groupby("ticket_id").cve_art.unique().reset_index()
tickets["n_items"] = tickets.cve_art.apply(len)
print(f"   tickets con detalle: {len(tickets):,}")
print(f"   tamaño de canasta — describe:\n{tickets.n_items.describe().to_string()}")

# Para análisis de co-compra necesitamos canastas con >=2 items
multi = tickets[tickets.n_items >= 2].copy()
print(f"   tickets con >=2 items: {len(multi):,} ({len(multi)/len(tickets)*100:.1f}%)")

# Guardar tickets con lista de SKUs (formato compacto)
tickets_out = tickets.copy()
tickets_out["cve_art"] = tickets_out.cve_art.apply(lambda a: ",".join(sorted(a)))
tickets_out.to_parquet(DATA / "tickets.parquet", index=False)
print(f"   → tickets.parquet ({len(tickets_out):,} filas)")


# ---------------------------------------------------------------------------
# 2. CO-OCURRENCIA con matriz dispersa
# ---------------------------------------------------------------------------
print("\n[2/4] Calculando matriz de co-ocurrencia…")
all_skus = sorted(set(s for arr in multi.cve_art for s in arr))
sku_to_idx = {s: i for i, s in enumerate(all_skus)}
n_skus = len(all_skus)
n_tickets = len(multi)
print(f"   SKUs distintos en canastas multi-item: {n_skus:,}")

# build sparse ticket × SKU
rows, cols = [], []
for tid, arr in enumerate(multi.cve_art):
    for s in arr:
        rows.append(tid)
        cols.append(sku_to_idx[s])
data = np.ones(len(rows), dtype=np.int32)
M = csr_matrix((data, (rows, cols)), shape=(n_tickets, n_skus), dtype=np.int32)
print(f"   matriz {M.shape} con {M.nnz:,} entradas")

# frecuencias por SKU (en cuántos tickets aparece)
freq = np.asarray(M.sum(axis=0)).ravel()  # support_A en conteo
print(f"   support por SKU — describe:\n{pd.Series(freq).describe().to_string()}")

# Filtrar SKUs con baja frecuencia para reducir tamaño
keep = np.where(freq >= MIN_TICKETS_SKU)[0]
print(f"   SKUs con support >= {MIN_TICKETS_SKU}: {len(keep):,}")
M_keep = M[:, keep]
freq_keep = freq[keep]
sku_keep = [all_skus[i] for i in keep]

# Co-ocurrencia: cuenta de tickets donde aparecen ambos
print("   computando M.T @ M …")
C = (M_keep.T @ M_keep).tocoo()  # n_keep × n_keep, entero, simétrico
print(f"   matriz de co-ocurrencia: {C.shape}, nnz={C.nnz:,}")


# ---------------------------------------------------------------------------
# 3. MÉTRICAS POR PAR
# ---------------------------------------------------------------------------
print("\n[3/4] Calculando lift y jaccard por par…")
# Sólo upper triangle, sin diagonal, con conteo >= MIN_PAIR_COUNT
mask = (C.row < C.col) & (C.data >= MIN_PAIR_COUNT)
i = C.row[mask]; j = C.col[mask]; n_ij = C.data[mask]
print(f"   pares con co-ocurrencia >= {MIN_PAIR_COUNT}: {len(i):,}")

n_i = freq_keep[i]
n_j = freq_keep[j]
# Lift = P(AB) / (P(A) P(B)) = (n_ij * N) / (n_i * n_j)
lift = (n_ij.astype(float) * n_tickets) / (n_i * n_j)
# Jaccard = n_ij / (n_i + n_j - n_ij)
jaccard = n_ij / (n_i + n_j - n_ij)
# Confidence A→B = P(B|A) = n_ij / n_i
conf_a_b = n_ij / n_i
conf_b_a = n_ij / n_j

pairs = pd.DataFrame({
    "sku_a": [sku_keep[k] for k in i],
    "sku_b": [sku_keep[k] for k in j],
    "n_a": n_i, "n_b": n_j, "n_ab": n_ij,
    "lift": lift, "jaccard": jaccard,
    "conf_a_b": conf_a_b, "conf_b_a": conf_b_a,
})

# Adjuntar familia de cada SKU para clasificar substituto vs complemento
iar = pd.read_parquet(SRC / "inviar.parquet",
                      columns=["art","fam","s_fam","lin","marca","des1"])
iar["art"] = iar.art.str.strip()
iar = iar.rename(columns={"art":"cve_art","des1":"descripcion"})
for c in ["fam","s_fam","lin","marca","descripcion"]:
    iar[c] = iar[c].str.strip()

meta = iar.set_index("cve_art")[["fam","s_fam","lin","marca","descripcion"]]
pairs["fam_a"] = pairs.sku_a.map(meta.fam)
pairs["fam_b"] = pairs.sku_b.map(meta.fam)
pairs["s_fam_a"] = pairs.sku_a.map(meta.s_fam)
pairs["s_fam_b"] = pairs.sku_b.map(meta.s_fam)
pairs["lin_a"] = pairs.sku_a.map(meta.lin)
pairs["lin_b"] = pairs.sku_b.map(meta.lin)
pairs["misma_familia"] = pairs.fam_a == pairs.fam_b
pairs["misma_subfam"] = pairs.s_fam_a == pairs.s_fam_b

# Score de sustituibilidad: en misma familia, lift bajo Y similar frecuencia individual
# Idea: dos productos sustituibles tienen frecuencias parecidas y aparecen juntos menos de lo esperado
def sub_score(row):
    if not row.misma_familia:
        return 0.0
    if row.lift >= 1:
        return 0.0  # complemento, no sustituto
    # cuanta más diferencia entre P(B|A) y P(A|B), menos sustituibles (uno arrastra al otro)
    sim = 1 - abs(np.log(max(row.n_a,1)) - np.log(max(row.n_b,1))) / 5  # similitud frecuencia
    sim = max(sim, 0)
    return (1 - row.lift) * sim   # mayor cuando lift es muy bajo y frecuencias similares

pairs["score_sustituto"] = pairs.apply(sub_score, axis=1)
pairs.to_parquet(DATA / "sku_pairs.parquet", index=False)
print(f"   → sku_pairs.parquet: {len(pairs):,} pares")


# ---------------------------------------------------------------------------
# 4. VECINOS POR SKU: top-K sustitutos y complementos
# ---------------------------------------------------------------------------
print(f"\n[4/4] Construyendo top-{TOP_K} vecinos por SKU…")

# Para cada SKU: tomar pares donde aparece (en a o b), invertir si necesario
def long_pairs(pairs_df):
    """Devuelve pares en formato largo: una fila por (foco, vecino)."""
    a = pairs_df.rename(columns={
        "sku_a":"foco","sku_b":"vecino",
        "fam_a":"fam_foco","fam_b":"fam_vec",
        "n_a":"n_foco","n_b":"n_vec",
        "conf_a_b":"conf_foco_vec","conf_b_a":"conf_vec_foco"})
    b = pairs_df.rename(columns={
        "sku_b":"foco","sku_a":"vecino",
        "fam_b":"fam_foco","fam_a":"fam_vec",
        "n_b":"n_foco","n_a":"n_vec",
        "conf_b_a":"conf_foco_vec","conf_a_b":"conf_vec_foco"})
    cols = ["foco","vecino","fam_foco","fam_vec","n_foco","n_vec","n_ab",
            "lift","jaccard","conf_foco_vec","conf_vec_foco",
            "misma_familia","misma_subfam","score_sustituto"]
    return pd.concat([a[cols], b[cols]], ignore_index=True)

long = long_pairs(pairs)

# Complementos: lift alto, cualquier familia (incluso misma)
comp = (long[long.lift > 1.2]
        .sort_values(["foco","lift"], ascending=[True, False])
        .groupby("foco").head(TOP_K)
        .assign(tipo="complemento"))

# Sustitutos: misma familia, score alto
sust = (long[(long.misma_familia) & (long.score_sustituto > 0)]
        .sort_values(["foco","score_sustituto"], ascending=[True, False])
        .groupby("foco").head(TOP_K)
        .assign(tipo="sustituto"))

vecinos = pd.concat([comp, sust], ignore_index=True)
# adjuntar descripciones
vecinos["desc_foco"] = vecinos.foco.map(meta.descripcion)
vecinos["desc_vecino"] = vecinos.vecino.map(meta.descripcion)
vecinos = vecinos[["foco","desc_foco","fam_foco","tipo",
                   "vecino","desc_vecino","fam_vec",
                   "n_foco","n_vec","n_ab","lift","jaccard",
                   "conf_foco_vec","score_sustituto"]]
vecinos.to_parquet(DATA / "sku_neighbors.parquet", index=False)
print(f"   → sku_neighbors.parquet: {len(vecinos):,} relaciones "
      f"({(vecinos.tipo=='complemento').sum():,} comp, "
      f"{(vecinos.tipo=='sustituto').sum():,} sust)")


# ---------------------------------------------------------------------------
# DIAGNÓSTICO
# ---------------------------------------------------------------------------
print("\n" + "="*80, "\nDIAGNÓSTICO DE CANASTAS\n", "="*80)
print(f"\nTickets totales: {n_tickets:,}")
print(f"Tamaño promedio canasta: {tickets.n_items.mean():.1f}")
print(f"Tamaño mediano canasta: {tickets.n_items.median():.0f}")

print(f"\nDistribución de lift en pares:")
print(pairs.lift.describe(percentiles=[.1,.25,.5,.75,.9,.95,.99]).to_string())

print(f"\nPares por categoría:")
print(f"  Misma familia y lift>=1.2 (complementos en familia): {((pairs.misma_familia)&(pairs.lift>=1.2)).sum():,}")
print(f"  Misma familia y lift<1 (sustitutos potenciales):     {((pairs.misma_familia)&(pairs.lift<1)).sum():,}")
print(f"  Diferentes familias y lift>=1.2 (complementos):       {((~pairs.misma_familia)&(pairs.lift>=1.2)).sum():,}")

# SKUs cubiertos
n_con_comp = vecinos[vecinos.tipo=="complemento"].foco.nunique()
n_con_sust = vecinos[vecinos.tipo=="sustituto"].foco.nunique()
print(f"\nSKUs con al menos 1 complemento: {n_con_comp:,}")
print(f"SKUs con al menos 1 sustituto:   {n_con_sust:,}")

# Ejemplos concretos
print(f"\n— Ejemplo: COCA COLA 600ML (006913) —")
ej = vecinos[vecinos.foco == "006913"]
print(ej[["tipo","vecino","desc_vecino","fam_vec","n_ab","lift","conf_foco_vec"]].to_string(index=False))

print(f"\n— Ejemplo: PLATANO CHIAPAS KG (006989) —")
ej = vecinos[vecinos.foco == "006989"]
print(ej[["tipo","vecino","desc_vecino","fam_vec","n_ab","lift","conf_foco_vec"]].to_string(index=False))

print(f"\n— Ejemplo: PAN BLANCO BIMBO 620G (000687) —")
ej = vecinos[vecinos.foco == "000687"]
print(ej[["tipo","vecino","desc_vecino","fam_vec","n_ab","lift","conf_foco_vec"]].to_string(index=False))

print(f"\n— Top 10 pares con lift más alto (complementos fuertes) —")
topc = pairs.sort_values("lift", ascending=False).head(10)
topc["desc_a"] = topc.sku_a.map(meta.descripcion)
topc["desc_b"] = topc.sku_b.map(meta.descripcion)
print(topc[["desc_a","desc_b","fam_a","fam_b","n_ab","lift","jaccard"]].to_string(index=False))
