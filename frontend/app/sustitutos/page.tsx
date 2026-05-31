"use client";
import { useEffect, useState } from "react";
import { Users, ArrowRight, ArrowDown, ArrowUp, Info, Minus } from "lucide-react";
import { SkuSelector } from "@/components/SkuSelector";
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Slider } from "@/components/ui/slider";
import { Alert } from "@/components/ui/alert";
import { formatMoney, formatPct, cn, formatNumber } from "@/lib/utils";
import {
  getSku, getSubstitutes,
  type Sku, type SkuState, type Substitute,
} from "@/lib/api";

export default function SustitutosPage() {
  const [sku, setSku] = useState<Sku | null>(null);
  const [state, setState] = useState<SkuState | null>(null);
  const [pctChange, setPctChange] = useState<number>(10);  // % cambio de precio focal
  const [subs, setSubs] = useState<Substitute[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!sku) return;
    getSku(sku.cve_art).then(setState);
  }, [sku]);

  useEffect(() => {
    if (!sku || !state) return;
    const newPrice = state.precio_actual * (1 + pctChange / 100);
    setLoading(true);
    const t = setTimeout(() => {
      getSubstitutes(sku.cve_art, newPrice)
        .then((r) => setSubs(r.items))
        .finally(() => setLoading(false));
    }, 200);
    return () => clearTimeout(t);
  }, [sku, state, pctChange]);

  const newPrice = state ? state.precio_actual * (1 + pctChange / 100) : 0;
  const subsConBeta = subs.filter((s) => s.beta_cross_shrunk !== null);
  const subsSinBeta = subs.filter((s) => s.beta_cross_shrunk === null);

  return (
    <main className="max-w-7xl mx-auto px-6 py-8 space-y-6 flex-1 w-full">
      <div className="space-y-1">
        <h1 className="text-2xl font-bold tracking-tight text-slate-900 flex items-center gap-2">
          <Users className="h-6 w-6 text-indigo-600" />
          Sustitutos y canibalización
        </h1>
        <p className="text-sm text-slate-600">
          Si subes el precio de un producto, sus sustitutos cercanos ganan
          demanda. Si lo bajas, los canibalizas.
        </p>
      </div>

      <Card>
        <CardContent className="pt-5 grid grid-cols-1 lg:grid-cols-3 gap-5">
          <div>
            <SkuSelector value={sku?.cve_art ?? null} onChange={setSku} />
            {state && (
              <div className="mt-4 grid grid-cols-2 gap-3 text-sm">
                <div className="rounded-lg bg-slate-50 px-3 py-2.5">
                  <div className="text-[10px] uppercase tracking-wide text-slate-500 font-semibold">Precio actual</div>
                  <div className="text-lg font-bold text-slate-900 tabular-nums">{formatMoney(state.precio_actual)}</div>
                </div>
                <div className="rounded-lg bg-emerald-50 px-3 py-2.5">
                  <div className="text-[10px] uppercase tracking-wide text-emerald-700 font-semibold">Nuevo precio</div>
                  <div className="text-lg font-bold text-emerald-900 tabular-nums">{formatMoney(newPrice)}</div>
                </div>
              </div>
            )}
          </div>

          <div className="lg:col-span-2 space-y-3">
            <label className="text-xs font-semibold text-slate-600 uppercase tracking-wide">
              Simulación: cambio de precio del producto focal
            </label>
            <Slider value={[pctChange]} min={-30} max={50} step={1}
              onValueChange={(v) => setPctChange(v[0])} />
            <div className="flex justify-between text-xs text-slate-500 tabular-nums">
              <span>-30%</span>
              <span className={cn(
                "font-bold text-base px-2 rounded",
                pctChange > 0 ? "text-rose-600" : pctChange < 0 ? "text-emerald-600" : "text-slate-700"
              )}>
                {formatPct(pctChange, 0)}
              </span>
              <span>+50%</span>
            </div>
            {state && (
              <p className="text-xs text-slate-500 pt-1">
                Pregunta: si {state.descripcion} pasa de ${state.precio_actual.toFixed(2)} a
                ${newPrice.toFixed(2)}, ¿qué pasa con sus sustitutos?
              </p>
            )}
          </div>
        </CardContent>
      </Card>

      {!sku && (
        <Card>
          <CardContent className="py-16 text-center">
            <Users className="h-12 w-12 text-slate-300 mx-auto mb-3" />
            <p className="text-slate-500">Selecciona un producto para ver sus sustitutos.</p>
          </CardContent>
        </Card>
      )}

      {sku && state && (
        <>
          <Alert severity="info">
            <b>Cómo se identifican los sustitutos:</b> con embeddings Item2Vec
            entrenados sobre 551,821 tickets. Productos que aparecen en canastas
            similares tienen vectores parecidos. La <b>similitud</b> es coseno
            entre vectores (0-1). Cuando un par tiene <b>β cruzado cuantificado</b>
            por nuestro modelo de cross-elasticidad, lo usamos para estimar la
            canibalización exacta.
          </Alert>

          <Card>
            <CardHeader>
              <CardTitle>Top 5 sustitutos de {state.descripcion}</CardTitle>
              <CardDescription>
                Ordenados por similitud semántica. Si subes el precio del focal,
                los sustitutos ganan demanda; si bajas, los pierden.
              </CardDescription>
            </CardHeader>
            <CardContent className="p-0">
              {loading && subs.length === 0 ? (
                <div className="py-12 text-center text-slate-400">Cargando…</div>
              ) : subs.length === 0 ? (
                <div className="py-12 text-center text-slate-400">
                  No se encontraron sustitutos para este producto.
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead className="bg-slate-50 text-slate-600 text-xs uppercase">
                      <tr>
                        <th className="text-left px-4 py-3 font-semibold">Sustituto</th>
                        <th className="text-center px-3 py-3 font-semibold">Familia</th>
                        <th className="text-right px-3 py-3 font-semibold">Similitud</th>
                        <th className="text-right px-3 py-3 font-semibold">β cruzado</th>
                        <th className="text-right px-4 py-3 font-semibold">
                          Δ demanda esperada
                          <div className="text-[10px] font-normal text-slate-400 normal-case">
                            si el focal pasa a ${newPrice.toFixed(2)}
                          </div>
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {subs.map((s, i) => (
                        <tr key={s.vecino} className={cn(
                          "border-t border-slate-100",
                          i % 2 === 0 ? "bg-white" : "bg-slate-50/30"
                        )}>
                          <td className="px-4 py-3">
                            <div className="font-medium text-slate-900">{s.descripcion}</div>
                            <div className="flex items-center gap-1.5 mt-0.5">
                              <span className="text-[10px] font-mono text-slate-400">{s.vecino}</span>
                              <span className="text-[10px] text-slate-500">{s.marca}</span>
                            </div>
                          </td>
                          <td className="px-3 py-3 text-center">
                            <Badge variant={s.fam === state.fam ? "info" : "neutral"}>{s.fam}</Badge>
                          </td>
                          <td className="px-3 py-3 text-right">
                            <div className="inline-flex items-center gap-2">
                              <div className="w-14 h-1.5 bg-slate-200 rounded-full overflow-hidden">
                                <div
                                  className="h-full bg-indigo-500"
                                  style={{ width: `${s.similitud_embedding * 100}%` }}
                                />
                              </div>
                              <span className="tabular-nums text-xs text-slate-700 font-semibold">
                                {(s.similitud_embedding * 100).toFixed(0)}%
                              </span>
                            </div>
                          </td>
                          <td className="px-3 py-3 text-right">
                            {s.beta_cross_shrunk !== null ? (
                              <div>
                                <div className="font-mono tabular-nums font-semibold text-slate-900">
                                  {s.beta_cross_shrunk > 0 ? "+" : ""}{s.beta_cross_shrunk.toFixed(2)}
                                </div>
                                {s.ci_lo !== null && s.ci_hi !== null && (
                                  <div className="text-[10px] text-slate-400 tabular-nums">
                                    IC [{s.ci_lo.toFixed(2)}, {s.ci_hi.toFixed(2)}]
                                  </div>
                                )}
                              </div>
                            ) : (
                              <span className="text-xs text-slate-400 italic">no cuantificado</span>
                            )}
                          </td>
                          <td className="px-4 py-3 text-right">
                            {s.delta_qty_pct !== null && s.delta_qty_pct !== undefined ? (
                              <CanibBadge value={s.delta_qty_pct} />
                            ) : (
                              <span className="text-xs text-slate-400 italic flex items-center justify-end gap-1">
                                <Minus className="h-3 w-3" /> sin estimar
                              </span>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </CardContent>
          </Card>

          {subsConBeta.length > 0 && (
            <Card className="bg-amber-50/40 border-amber-200">
              <CardContent className="pt-5">
                <div className="text-sm text-amber-900 leading-relaxed">
                  <b>Lectura para decidir:</b> de los {subs.length} sustitutos
                  detectados, {subsConBeta.length} tienen elasticidad cruzada
                  cuantificada con suficiente data histórica. Los demás existen
                  semánticamente (los embeddings los conocen) pero no tienen
                  suficientes co-apariciones para estimar la magnitud — solo
                  podemos confiar en la dirección.
                </div>
              </CardContent>
            </Card>
          )}
        </>
      )}
    </main>
  );
}

function CanibBadge({ value }: { value: number }) {
  const isGain = value > 0;
  const isLoss = value < 0;
  if (Math.abs(value) < 0.5) {
    return <span className="text-xs text-slate-500">sin cambio</span>;
  }
  return (
    <div className={cn(
      "inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-semibold tabular-nums",
      isGain && "bg-emerald-100 text-emerald-700",
      isLoss && "bg-rose-100 text-rose-700"
    )}>
      {isGain ? <ArrowUp className="h-3 w-3" /> : <ArrowDown className="h-3 w-3" />}
      {value > 0 ? "+" : ""}{value.toFixed(1)}%
    </div>
  );
}
