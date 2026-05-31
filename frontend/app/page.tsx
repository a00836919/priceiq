"use client";
import { useEffect, useState } from "react";
import {
  TrendingUp, DollarSign, Package, ShoppingBasket, Cpu,
} from "lucide-react";
import { SkuSelector } from "@/components/SkuSelector";
import { ConfidenceCard } from "@/components/ConfidenceCard";
import { DemandChart } from "@/components/DemandChart";
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Slider } from "@/components/ui/slider";
import { Alert } from "@/components/ui/alert";
import { formatMoney, formatNumber, formatPct, cn } from "@/lib/utils";
import {
  recommend, getSku, getBacktestSummary,
  type Sku, type Recommendation, type SkuState, type BacktestSummary,
} from "@/lib/api";

export default function Home() {
  const [sku, setSku] = useState<Sku | null>(null);
  const [state, setState] = useState<SkuState | null>(null);
  const [costDelta, setCostDelta] = useState<number>(10);
  const [costMode, setCostMode] = useState<"pct" | "abs">("pct");
  const [costAbs, setCostAbs] = useState<number>(0);
  const [conservatism, setConservatism] = useState<number>(0.5);
  const [objetivo, setObjetivo] = useState<"margen" | "ingreso" | "cantidad">("margen");
  const [rec, setRec] = useState<Recommendation | null>(null);
  const [loading, setLoading] = useState(false);
  const [summary, setSummary] = useState<BacktestSummary | null>(null);
  const [chartMode, setChartMode] = useState<"demand" | "margin">("demand");

  useEffect(() => {
    getBacktestSummary().then(setSummary).catch(() => {});
  }, []);

  useEffect(() => {
    if (!sku) return;
    getSku(sku.cve_art).then((s) => {
      setState(s);
      setCostAbs(s.costo_actual_unitario * 1.10);
    });
  }, [sku]);

  useEffect(() => {
    if (!state || !sku) return;
    const newCost = costMode === "pct"
      ? state.costo_actual_unitario * (1 + costDelta / 100)
      : costAbs;
    if (newCost <= 0) return;
    setLoading(true);
    const t = setTimeout(() => {
      recommend({
        sku: sku.cve_art, new_cost: newCost,
        conservatism, objetivo,
      }).then(setRec).finally(() => setLoading(false));
    }, 250);
    return () => clearTimeout(t);
  }, [sku, state, costDelta, costAbs, costMode, conservatism, objetivo]);

  const newCost = state
    ? costMode === "pct"
      ? state.costo_actual_unitario * (1 + costDelta / 100)
      : costAbs
    : 0;

  return (
    <>
      <main className="max-w-7xl mx-auto px-6 py-8 space-y-6 flex-1 w-full">
        <div className="space-y-1">
          <h1 className="text-2xl font-bold tracking-tight text-slate-900">Recomendador de precio</h1>
          <p className="text-sm text-slate-600">Selecciona un producto y simula un cambio de costo del proveedor.</p>
        </div>

        <Card>
          <CardContent className="pt-5 grid grid-cols-1 lg:grid-cols-3 gap-5">
            <div className="lg:col-span-1">
              <SkuSelector value={sku?.cve_art ?? null} onChange={setSku} />
              {state && (
                <div className="mt-4 grid grid-cols-2 gap-3 text-sm">
                  <div className="rounded-lg bg-slate-50 px-3 py-2.5">
                    <div className="text-[10px] uppercase tracking-wide text-slate-500 font-semibold">Precio actual</div>
                    <div className="text-lg font-bold text-slate-900 tabular-nums">{formatMoney(state.precio_actual)}</div>
                  </div>
                  <div className="rounded-lg bg-slate-50 px-3 py-2.5">
                    <div className="text-[10px] uppercase tracking-wide text-slate-500 font-semibold">Costo actual</div>
                    <div className="text-lg font-bold text-slate-900 tabular-nums">{formatMoney(state.costo_actual_unitario)}</div>
                  </div>
                </div>
              )}
            </div>

            <div className="lg:col-span-2 space-y-4">
              <div>
                <div className="flex items-center justify-between mb-1.5">
                  <label className="text-xs font-semibold text-slate-600 uppercase tracking-wide">Nuevo costo del proveedor</label>
                  <div className="flex gap-1 bg-slate-100 rounded-md p-0.5">
                    <button onClick={() => setCostMode("pct")}
                      className={cn("text-[11px] px-2 py-0.5 rounded font-medium transition",
                        costMode === "pct" ? "bg-white shadow-sm text-slate-900" : "text-slate-500")}>% cambio</button>
                    <button onClick={() => setCostMode("abs")}
                      className={cn("text-[11px] px-2 py-0.5 rounded font-medium transition",
                        costMode === "abs" ? "bg-white shadow-sm text-slate-900" : "text-slate-500")}>Costo directo</button>
                  </div>
                </div>
                {costMode === "pct" ? (
                  <div className="space-y-2">
                    <Slider value={[costDelta]} min={-30} max={80} step={1}
                      onValueChange={(v) => setCostDelta(v[0])} />
                    <div className="flex justify-between text-xs text-slate-500 tabular-nums">
                      <span>-30%</span>
                      <span className={cn("font-bold text-base px-2 rounded",
                        costDelta > 0 ? "text-rose-600" : costDelta < 0 ? "text-emerald-600" : "text-slate-700")}>
                        {formatPct(costDelta, 0)}
                      </span>
                      <span>+80%</span>
                    </div>
                  </div>
                ) : (
                  <input type="number" value={costAbs} onChange={(e) => setCostAbs(parseFloat(e.target.value) || 0)}
                    step={0.10} min={0}
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm font-mono tabular-nums focus:outline-none focus:ring-2 focus:ring-indigo-500" />
                )}
                {state && (
                  <div className="mt-2 flex items-center gap-2 text-sm">
                    <span className="text-slate-500">Nuevo costo:</span>
                    <span className="font-bold text-slate-900 tabular-nums">{formatMoney(newCost)}</span>
                  </div>
                )}
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-xs font-semibold text-slate-600 uppercase tracking-wide block mb-1.5">Conservadurismo</label>
                  <Slider value={[conservatism * 100]} min={0} max={100} step={10}
                    onValueChange={(v) => setConservatism(v[0] / 100)} />
                  <div className="flex justify-between text-[10px] text-slate-500 mt-1">
                    <span>Agresivo</span>
                    <span className="font-bold text-slate-700">{(conservatism * 100).toFixed(0)}</span>
                    <span>Conservador</span>
                  </div>
                </div>
                <div>
                  <label className="text-xs font-semibold text-slate-600 uppercase tracking-wide block mb-1.5">Optimizar</label>
                  <div className="flex gap-1 bg-slate-100 rounded-md p-0.5">
                    {(["margen", "ingreso", "cantidad"] as const).map(o => (
                      <button key={o} onClick={() => setObjetivo(o)}
                        className={cn("flex-1 text-xs px-2 py-1.5 rounded font-medium capitalize transition",
                          objetivo === o ? "bg-white shadow-sm text-slate-900" : "text-slate-500")}>{o}</button>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {!sku && (
          <Card>
            <CardContent className="py-16 text-center">
              <ShoppingBasket className="h-12 w-12 text-slate-300 mx-auto mb-3" />
              <p className="text-slate-500">Selecciona un producto arriba para ver la recomendación.</p>
            </CardContent>
          </Card>
        )}

        {rec && state && (
          <>
            {rec.cost_check.severity === "extreme" && (
              <Alert severity="error" title="Costo fuera del rango histórico completo">
                {rec.cost_check.message} El modelo nunca vio costos así para este SKU; la recomendación es una
                extrapolación. Considera escalonar el ajuste o pedir validación humana antes de aplicar.
              </Alert>
            )}
            {rec.cost_check.severity === "warning" && (
              <Alert severity="warning" title="Costo inusual">{rec.cost_check.message}</Alert>
            )}

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
              <div className="lg:col-span-1">
                <ConfidenceCard rec={rec} />
              </div>

              <div className="lg:col-span-2 space-y-4">
                <Card className="bg-gradient-to-br from-emerald-50 to-teal-50 border-emerald-200">
                  <CardContent className="pt-5">
                    <div className="text-[11px] uppercase tracking-wider font-semibold text-emerald-700 mb-1">
                      Zona recomendada de precio
                    </div>
                    <div className="flex items-baseline gap-3 mb-2 flex-wrap">
                      <span className="text-4xl font-bold text-emerald-900 tabular-nums">{formatMoney(rec.rango.lo)}</span>
                      <span className="text-xl text-emerald-700 font-medium">—</span>
                      <span className="text-4xl font-bold text-emerald-900 tabular-nums">{formatMoney(rec.rango.hi)}</span>
                    </div>
                    <div className="text-xs text-emerald-700">
                      Precio óptimo dentro del rango:{" "}
                      <span className="font-bold text-lg tabular-nums text-emerald-900">{formatMoney(rec.precio_recomendado)}</span>
                      {" "}
                      <span className="text-emerald-600">({formatPct((rec.precio_recomendado / state.precio_actual - 1) * 100)} vs actual)</span>
                    </div>
                  </CardContent>
                </Card>

                <div className="grid grid-cols-3 gap-3">
                  <Card>
                    <CardContent className="pt-4">
                      <div className="flex items-center justify-between mb-1">
                        <Package className="h-4 w-4 text-blue-500" />
                        <span className="text-[10px] font-semibold uppercase text-slate-500 tracking-wide">Demanda</span>
                      </div>
                      <div className="text-2xl font-bold tabular-nums text-slate-900">{formatNumber(rec.qty_esperada)}</div>
                      <div className="text-[11px] text-slate-500">IC: {formatNumber(rec.qty_lo, 0)}–{formatNumber(rec.qty_hi, 0)} /día</div>
                    </CardContent>
                  </Card>
                  <Card>
                    <CardContent className="pt-4">
                      <div className="flex items-center justify-between mb-1">
                        <DollarSign className="h-4 w-4 text-emerald-500" />
                        <span className="text-[10px] font-semibold uppercase text-slate-500 tracking-wide">Margen/día</span>
                      </div>
                      <div className="text-2xl font-bold tabular-nums text-slate-900">{formatMoney(rec.margen_esperado_diario, 0)}</div>
                      <div className={cn("text-[11px] tabular-nums",
                        rec.delta_margen_diario > 0 ? "text-emerald-600 font-semibold" : "text-rose-600")}>
                        {rec.delta_margen_diario > 0 ? "▲" : "▼"} {formatMoney(Math.abs(rec.delta_margen_diario), 0)} vs actual
                      </div>
                    </CardContent>
                  </Card>
                  <Card>
                    <CardContent className="pt-4">
                      <div className="flex items-center justify-between mb-1">
                        <TrendingUp className="h-4 w-4 text-violet-500" />
                        <span className="text-[10px] font-semibold uppercase text-slate-500 tracking-wide">Margen unit</span>
                      </div>
                      <div className="text-2xl font-bold tabular-nums text-slate-900">{formatMoney(rec.margen_unit)}</div>
                      <div className="text-[11px] text-slate-500">{((rec.margen_unit / rec.precio_recomendado) * 100).toFixed(1)}% del precio</div>
                    </CardContent>
                  </Card>
                </div>

                <div className="flex items-center gap-2 text-xs">
                  <Badge variant={rec.modelo_usado === "LightGBM" ? "info" : rec.modelo_usado === "XGBoost" ? "warning" : "neutral"}>
                    <Cpu className="h-3 w-3 mr-1 inline" />
                    Modelo: {rec.modelo_usado}
                  </Badge>
                  <span className="text-slate-500">Elegido automáticamente para la familia <b>{state.fam}</b></span>
                </div>
              </div>
            </div>

            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle>Curva precio → {chartMode === "demand" ? "demanda" : "margen"}</CardTitle>
                    <CardDescription>Banda verde: zona recomendada. Línea gris: precio actual. Línea verde: óptimo.</CardDescription>
                  </div>
                  <div className="flex gap-1 bg-slate-100 rounded-md p-0.5">
                    <button onClick={() => setChartMode("demand")}
                      className={cn("text-xs px-3 py-1 rounded font-medium transition",
                        chartMode === "demand" ? "bg-white shadow-sm text-slate-900" : "text-slate-500")}>Demanda</button>
                    <button onClick={() => setChartMode("margin")}
                      className={cn("text-xs px-3 py-1 rounded font-medium transition",
                        chartMode === "margin" ? "bg-white shadow-sm text-slate-900" : "text-slate-500")}>Margen</button>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <DemandChart rec={rec} mode={chartMode} />
              </CardContent>
            </Card>

            {summary && (
              <Card className="bg-slate-50/50">
                <CardContent className="pt-5">
                  <div className="text-xs text-slate-600 leading-relaxed">
                    <b className="text-slate-900">¿Por qué confiar en esta recomendación?</b>{" "}
                    Validamos el modelo en <b>{summary.n_eventos.toLocaleString()}</b> cambios reales de precio en los últimos 60 días.
                    MAPE mediano de <b>{summary.mape_mediano.toFixed(1)}%</b> y cobertura empírica del IC80% de
                    {" "}<b>{summary.cobertura_ic80.toFixed(1)}%</b> (target 80%). El modelo predice mejor las
                    magnitudes que la dirección ({summary.direccion_acertada.toFixed(0)}% de acierto en dirección).
                  </div>
                </CardContent>
              </Card>
            )}
          </>
        )}

        {loading && !rec && (
          <div className="flex items-center justify-center py-12 text-slate-400">
            <div className="animate-spin rounded-full h-6 w-6 border-2 border-indigo-500 border-t-transparent" />
            <span className="ml-2 text-sm">Calculando…</span>
          </div>
        )}
      </main>

      <footer className="border-t border-slate-200 mt-12 py-6">
        <div className="max-w-7xl mx-auto px-6 text-center text-xs text-slate-500">
          PriceIQ · LightGBM + Item2Vec + Conformal Prediction + IV-2SLS · Datos: 22 meses, 7.3K SKUs
        </div>
      </footer>
    </>
  );
}
