"""Item2Vec: embeddings densos de SKU desde canastas.

Cada ticket (transacción de un cliente en un momento) es una "oración"
de SKUs. Word2Vec / Skip-Gram aprende vectores tales que SKUs que
aparecen en contextos similares (mismas canastas) tienen vectores parecidos.

Salidas:
  data/sku_embeddings.parquet   — cve_art × {emb_0…emb_31}
  data/embedding_neighbors.parquet — vecinos por similitud cosine (top-10 por SKU)
  data/item2vec_model.kv         — modelo gensim (para SKUs nuevos en producción)
"""
from pathlib import Path
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
import warnings
warnings.filterwarnings("ignore")

SRC = Path("tablas_merksyst_parquet")
DATA = Path("data")
EMB_DIM = 32         # tamaño del vector por SKU
WINDOW = 8           # rango de contexto en la canasta
MIN_COUNT = 5        # SKU debe aparecer en >=5 tickets
EPOCHS = 25

# ---------------------------------------------------------------------------
# 1. Reconstruir canastas (tickets) — un ticket = unique (Suc, FecOpe, Caja, NumTra)
# ---------------------------------------------------------------------------
print("[1/4] Cargando líneas de venta y agrupando en tickets…")
det = pd.read_parquet(SRC / "pdvdet.parquet",
    columns=["Suc","FecOpe","Caja","NumTra","cve_art","cantidad","status"])
det = det[(det.status=="01") & (det.cantidad > 0)].copy()
det["cve_art"] = det.cve_art.str.strip()
det["ticket_id"] = (det.Suc + "|" + det.FecOpe + "|" + det.Caja + "|" + det.NumTra)

# Agrupar SKUs por ticket, ordenados por aparición original
tickets = det.groupby("ticket_id")["cve_art"].apply(list)
# Filtrar tickets con ≥2 SKUs (los de 1 SKU no aportan contexto)
tickets = tickets[tickets.apply(len) >= 2]
print(f"   tickets totales: {len(tickets):,}")
print(f"   distribución de tamaño de canasta:")
print(tickets.apply(len).describe(percentiles=[.5,.75,.9,.99]).to_string())


# ---------------------------------------------------------------------------
# 2. Entrenar Word2Vec (Skip-Gram)
# ---------------------------------------------------------------------------
print(f"\n[2/4] Entrenando Item2Vec (dim={EMB_DIM}, window={WINDOW}, epochs={EPOCHS})…")
sentences = tickets.values.tolist()
model = Word2Vec(
    sentences=sentences,
    vector_size=EMB_DIM,
    window=WINDOW,
    min_count=MIN_COUNT,
    sg=1,                  # skip-gram (mejor para items raros)
    negative=10,
    sample=1e-4,
    epochs=EPOCHS,
    workers=4,
    seed=42,
)
print(f"   SKUs en vocabulario: {len(model.wv.key_to_index):,}")
model.wv.save(str(DATA / "item2vec_model.kv"))


# ---------------------------------------------------------------------------
# 3. Exportar embeddings a parquet
# ---------------------------------------------------------------------------
print(f"\n[3/4] Exportando embeddings…")
skus = list(model.wv.key_to_index.keys())
emb_matrix = model.wv.vectors  # (n_skus, EMB_DIM)
emb_df = pd.DataFrame(emb_matrix, columns=[f"emb_{i}" for i in range(EMB_DIM)])
emb_df.insert(0, "cve_art", skus)

# Adjuntar metadata para inspección
iar = pd.read_parquet(SRC / "inviar.parquet", columns=["art","fam","lin","marca","des1"])
iar["art"] = iar.art.str.strip()
for c in ["fam","lin","marca","des1"]:
    iar[c] = iar[c].str.strip()
iar = iar.rename(columns={"art":"cve_art","des1":"descripcion"})
emb_df = emb_df.merge(iar, on="cve_art", how="left")
emb_df.to_parquet(DATA / "sku_embeddings.parquet", index=False)
print(f"   sku_embeddings.parquet: {len(emb_df):,} SKUs × {EMB_DIM} dim")


# ---------------------------------------------------------------------------
# 4. Vecinos por similitud cosine (top-10 por SKU)
# ---------------------------------------------------------------------------
print(f"\n[4/4] Calculando vecinos top-10 por similitud cosine…")
neighbors = []
for sku in skus:
    try:
        sim = model.wv.most_similar(sku, topn=10)
    except KeyError:
        continue
    for vec, score in sim:
        neighbors.append({"focal": sku, "vecino": vec, "similarity": float(score)})

nbr_df = pd.DataFrame(neighbors)
nbr_df = nbr_df.merge(iar.rename(columns={"cve_art":"focal","descripcion":"desc_focal","fam":"fam_focal"})[
    ["focal","desc_focal","fam_focal"]], on="focal", how="left")
nbr_df = nbr_df.merge(iar.rename(columns={"cve_art":"vecino","descripcion":"desc_vecino","fam":"fam_vecino"})[
    ["vecino","desc_vecino","fam_vecino"]], on="vecino", how="left")
nbr_df["misma_fam"] = nbr_df.fam_focal == nbr_df.fam_vecino
nbr_df.to_parquet(DATA / "embedding_neighbors.parquet", index=False)
print(f"   embedding_neighbors.parquet: {len(nbr_df):,} relaciones top-10")

# ---------------------------------------------------------------------------
# DIAGNÓSTICO
# ---------------------------------------------------------------------------
print("\n" + "="*80, "\nDIAGNÓSTICO ITEM2VEC\n", "="*80)

print(f"\n— ¿Los embeddings agrupan por familia? —")
pct_misma_fam = nbr_df.misma_fam.mean() * 100
print(f"   % vecinos top-10 en MISMA familia (semánticamente coherente): {pct_misma_fam:.1f}%")

# Ejemplos: vecinos cercanos para SKUs conocidos
ejemplos = [
    ("006989", "PLATANO"),
    ("006913", "COCA 600ML"),
    ("006931", "AGUACATE HASS"),
    ("000687", "PAN BIMBO"),
    ("002439", "PECHUGA POLLO"),
]
for sku, label in ejemplos:
    if sku in model.wv.key_to_index:
        print(f"\n— TOP 8 vecinos semánticos de {label} ({sku}) —")
        sim = model.wv.most_similar(sku, topn=8)
        for vec, score in sim:
            d = iar[iar.cve_art==vec]
            desc = d.descripcion.iloc[0] if len(d) else "?"
            fam = d.fam.iloc[0] if len(d) else "?"
            print(f"   {score:.3f}  {fam:6s}  {desc}")

# Comparación con vecinos por lift (canastas)
print(f"\n— Cruce con vecinos por LIFT (canastas) —")
vec_lift = pd.read_parquet(DATA / "sku_neighbors.parquet")
overlap = vec_lift.merge(nbr_df, left_on=["foco","vecino"], right_on=["focal","vecino"], how="inner")
print(f"   Pares en común entre lift-top-X y embedding-top-10: {len(overlap):,}")
print(f"   De los pares lift, % también en embedding top-10: "
      f"{len(overlap)/len(vec_lift)*100:.1f}%")

# SKUs nuevos para LightGBM: ahora todo SKU del vocabulario tiene vector
print(f"\n— Cobertura para LightGBM —")
n_skus_emb = emb_df.cve_art.nunique()
panel = pd.read_parquet(DATA / "panel_diario.parquet")
skus_panel = panel.cve_art.unique()
overlap_panel = sum(1 for s in skus_panel if s in model.wv.key_to_index)
print(f"   SKUs en panel:        {len(skus_panel):,}")
print(f"   SKUs en embeddings:   {n_skus_emb:,}")
print(f"   Cobertura del panel:  {overlap_panel:,} ({overlap_panel/len(skus_panel)*100:.1f}%)")
