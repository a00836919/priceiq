const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export type Sku = {
  cve_art: string;
  descripcion: string;
  fam: string;
  lin: string;
  marca: string;
};

export type SkuState = {
  cve_art: string;
  descripcion: string;
  fam: string;
  lin: string;
  marca: string;
  precio_actual: number;
  costo_actual_unitario: number;
  costo_actual_caja: number;
  factor_uds: number;
  qty_diaria_med: number;
  qty_diaria_p75: number;
  n_dias: number;
  fecha_ultima: string;
  elasticidad_estructural: number | null;
};

export type CurvePoint = {
  precio: number;
  qty_med: number;
  qty_lo: number;
  qty_hi: number;
  margen_med: number;
  margen_lo: number;
  margen_hi: number;
  margen_med_penalizado: number;
  credibilidad: number;
};

export type Recommendation = {
  precio_recomendado: number;
  qty_esperada: number;
  qty_lo: number;
  qty_hi: number;
  margen_unit: number;
  margen_esperado_diario: number;
  ingreso_esperado_diario: number;
  credibilidad: number;
  precio_actual: number;
  margen_al_precio_actual: number;
  qty_al_precio_actual: number;
  delta_margen_diario: number;
  modelo_usado: string;
  objetivo: string;
  conservatism: number;
  rango: { lo: number; hi: number; n_puntos: number };
  confianza: {
    score: number;
    nivel: "ALTA" | "MEDIA" | "BAJA";
    emoji: string;
    texto: string;
    componentes: Record<string, number>;
    n_eventos_fam: number;
    n_dias_sku: number;
  };
  cost_check: {
    severity: "ok" | "warning" | "extreme";
    is_outlier: boolean;
    message: string;
    delta_pct: number;
    costo_actual: number;
    costo_hist_min: number;
    costo_hist_max: number;
  };
  state: SkuState;
  curve: CurvePoint[];
};

export type BacktestSummary = {
  n_eventos: number;
  n_skus: number;
  mape_mediano: number;
  cobertura_ic80: number;
  direccion_acertada: number;
  subidas: number;
  bajadas: number;
};

export async function listSkus(q?: string, limit = 50): Promise<{ total: number; items: Sku[] }> {
  const url = new URL(`${API_BASE}/api/skus`);
  if (q) url.searchParams.set("q", q);
  url.searchParams.set("limit", String(limit));
  const r = await fetch(url.toString(), { cache: "no-store" });
  if (!r.ok) throw new Error(`API error ${r.status}`);
  return r.json();
}

export async function getSku(sku: string): Promise<SkuState> {
  const r = await fetch(`${API_BASE}/api/skus/${sku}`, { cache: "no-store" });
  if (!r.ok) throw new Error(`API error ${r.status}`);
  return r.json();
}

export async function recommend(req: {
  sku: string;
  new_cost: number;
  conservatism: number;
  objetivo: string;
}): Promise<Recommendation> {
  const r = await fetch(`${API_BASE}/api/recommend`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!r.ok) throw new Error(`API error ${r.status}`);
  return r.json();
}

export async function getBacktestSummary(): Promise<BacktestSummary> {
  const r = await fetch(`${API_BASE}/api/backtest-summary`, { cache: "no-store" });
  if (!r.ok) throw new Error(`API error ${r.status}`);
  return r.json();
}

export type Substitute = {
  vecino: string;
  descripcion: string;
  fam: string;
  marca: string;
  similitud_embedding: number;
  beta_cross_shrunk: number | null;
  ci_lo: number | null;
  ci_hi: number | null;
  tipo_vecino_canasta: string;
  delta_qty_pct?: number | null;
  delta_qty_pct_lo?: number | null;
  delta_qty_pct_hi?: number | null;
};

export type SubstitutesResponse = {
  items: Substitute[];
  p_actual: number | null;
  fam_focal: string | null;
  desc_focal: string | null;
};

export async function getSubstitutes(sku: string, newPrice?: number): Promise<SubstitutesResponse> {
  const url = new URL(`${API_BASE}/api/skus/${sku}/substitutes`);
  if (newPrice !== undefined) url.searchParams.set("new_price", String(newPrice));
  const r = await fetch(url.toString(), { cache: "no-store" });
  if (!r.ok) throw new Error(`API error ${r.status}`);
  return r.json();
}

export type BatchItem = { sku: string; new_cost: number };
export type BatchResultRow = {
  sku: string;
  new_cost: number;
  descripcion: string | null;
  fam: string | null;
  marca: string | null;
  precio_actual: number | null;
  costo_actual: number | null;
  precio_recomendado: number | null;
  rango_lo: number | null;
  rango_hi: number | null;
  delta_precio_pct: number | null;
  qty_esperada: number | null;
  qty_lo: number | null;
  qty_hi: number | null;
  margen_esperado: number | null;
  margen_actual: number | null;
  delta_margen: number | null;
  credibilidad: number | null;
  confianza_score: number | null;
  confianza_nivel: "ALTA" | "MEDIA" | "BAJA" | null;
  modelo_usado: string | null;
  cost_severity: "ok" | "warning" | "extreme" | null;
  cost_message: string | null;
  status: "ok" | "error";
  error: string | null;
};

export type BatchSummary = {
  n_total: number;
  n_ok: number;
  n_err: number;
  margen_total_recomendado: number;
  margen_total_actual: number;
  delta_margen_total: number;
  delta_margen_pct: number;
  confianza_buckets: { ALTA: number; MEDIA: number; BAJA: number };
  cost_severity_buckets: { ok: number; warning: number; extreme: number };
};

export async function recommendBatch(
  items: BatchItem[], conservatism = 0.5, objetivo = "margen"
): Promise<{ items: BatchResultRow[]; summary: BatchSummary }> {
  const r = await fetch(`${API_BASE}/api/recommend-batch`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ items, conservatism, objetivo }),
  });
  if (!r.ok) throw new Error(`API error ${r.status}`);
  return r.json();
}
