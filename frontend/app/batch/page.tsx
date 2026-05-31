"use client";
import { useState, useRef } from "react";
import {
  Upload, FileSpreadsheet, Download, AlertCircle, CheckCircle2,
  TrendingUp, DollarSign, Package, Loader2, FileDown,
} from "lucide-react";
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Alert } from "@/components/ui/alert";
import { cn, formatMoney, formatPct } from "@/lib/utils";
import { recommendBatch, type BatchItem, type BatchResultRow, type BatchSummary } from "@/lib/api";

const EXAMPLE_CSV = `cve_art,new_cost
006913,18.25
006989,26.00
002439,76.80
000687,38.50
006931,46.50`;

export default function BatchPage() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [items, setItems] = useState<BatchItem[]>([]);
  const [filename, setFilename] = useState<string>("");
  const [results, setResults] = useState<BatchResultRow[]>([]);
  const [summary, setSummary] = useState<BatchSummary | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>("");
  const [conservatism, setConservatism] = useState(0.5);

  function parseCSV(text: string): BatchItem[] {
    const lines = text.trim().split(/\r?\n/);
    if (lines.length < 2) throw new Error("CSV vacío o sin encabezado");
    const header = lines[0].toLowerCase().split(",").map(h => h.trim());
    const skuIdx = header.findIndex(h => h.includes("cve_art") || h.includes("sku") || h.includes("articulo"));
    const costIdx = header.findIndex(h => h.includes("cost") || h.includes("precio_prov"));
    if (skuIdx < 0 || costIdx < 0) {
      throw new Error("CSV debe tener columnas 'cve_art' y 'new_cost' (o sku/costo).");
    }
    const out: BatchItem[] = [];
    for (const ln of lines.slice(1)) {
      const cols = ln.split(",").map(c => c.trim());
      if (cols.length <= Math.max(skuIdx, costIdx)) continue;
      const sku = cols[skuIdx].replace(/^"|"$/g, "");
      const cost = parseFloat(cols[costIdx]);
      if (!sku || !Number.isFinite(cost) || cost <= 0) continue;
      out.push({ sku, new_cost: cost });
    }
    return out;
  }

  async function handleFile(f: File) {
    setError("");
    setFilename(f.name);
    try {
      const text = await f.text();
      const parsed = parseCSV(text);
      if (parsed.length === 0) throw new Error("Sin filas válidas tras parsear");
      setItems(parsed);
      setResults([]);
      setSummary(null);
    } catch (e: any) {
      setError(e.message);
    }
  }

  function loadExample() {
    const parsed = parseCSV(EXAMPLE_CSV);
    setItems(parsed);
    setFilename("ejemplo.csv");
    setResults([]);
    setSummary(null);
    setError("");
  }

  async function runBatch() {
    if (items.length === 0) return;
    setLoading(true);
    setError("");
    try {
      const r = await recommendBatch(items, conservatism, "margen");
      setResults(r.items);
      setSummary(r.summary);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  function downloadCSV() {
    if (results.length === 0) return;
    const cols = [
      "sku","descripcion","fam","marca",
      "precio_actual","costo_actual","new_cost",
      "precio_recomendado","rango_lo","rango_hi","delta_precio_pct",
      "qty_esperada","qty_lo","qty_hi",
      "margen_actual","margen_esperado","delta_margen",
      "confianza_nivel","confianza_score","modelo_usado",
      "cost_severity","status","error",
    ];
    const escape = (v: any) => {
      if (v === null || v === undefined) return "";
      const s = String(v);
      return /[,"\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
    };
    const csv = [
      cols.join(","),
      ...results.map(r => cols.map(c => escape((r as any)[c])).join(",")),
    ].join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `priceiq_recomendaciones_${new Date().toISOString().slice(0,10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <main className="max-w-7xl mx-auto px-6 py-8 space-y-6 flex-1 w-full">
      <div className="space-y-1">
        <h1 className="text-2xl font-bold tracking-tight text-slate-900 flex items-center gap-2">
          <FileSpreadsheet className="h-6 w-6 text-indigo-600" />
          Modo lote — recomendaciones masivas
        </h1>
        <p className="text-sm text-slate-600">
          Sube una lista del proveedor (CSV con SKU + nuevo costo) y obtén
          recomendaciones de precio para todos los productos a la vez.
        </p>
      </div>

      {/* UPLOAD */}
      <Card>
        <CardContent className="pt-5 space-y-4">
          <div
            onClick={() => inputRef.current?.click()}
            onDragOver={(e) => { e.preventDefault(); e.currentTarget.classList.add("border-indigo-500","bg-indigo-50/50"); }}
            onDragLeave={(e) => e.currentTarget.classList.remove("border-indigo-500","bg-indigo-50/50")}
            onDrop={(e) => {
              e.preventDefault();
              e.currentTarget.classList.remove("border-indigo-500","bg-indigo-50/50");
              const f = e.dataTransfer.files[0];
              if (f) handleFile(f);
            }}
            className="border-2 border-dashed border-slate-300 rounded-xl p-8 text-center cursor-pointer hover:border-indigo-400 hover:bg-slate-50 transition"
          >
            <Upload className="h-10 w-10 text-slate-400 mx-auto mb-3" />
            <div className="text-sm font-medium text-slate-700">
              {filename || "Arrastra un CSV aquí o haz clic para seleccionar"}
            </div>
            <div className="text-xs text-slate-500 mt-1">
              Columnas requeridas: <code className="font-mono bg-slate-100 px-1 rounded">cve_art</code>,{" "}
              <code className="font-mono bg-slate-100 px-1 rounded">new_cost</code>
            </div>
            <input
              type="file"
              accept=".csv"
              ref={inputRef}
              className="hidden"
              onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
            />
          </div>

          {/* Acciones */}
          <div className="flex flex-wrap items-center gap-3">
            <button
              onClick={loadExample}
              className="text-xs text-indigo-600 hover:text-indigo-800 underline"
            >
              ¿No tienes CSV? Cargar ejemplo (5 SKUs)
            </button>
            {items.length > 0 && (
              <Badge variant="info">{items.length} filas detectadas</Badge>
            )}
            <div className="flex-1" />
            {items.length > 0 && (
              <button
                onClick={runBatch}
                disabled={loading}
                className="inline-flex items-center gap-2 bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 text-white text-sm font-medium px-4 py-2 rounded-lg shadow-sm transition"
              >
                {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Package className="h-4 w-4" />}
                {loading ? "Procesando..." : `Procesar ${items.length} SKUs`}
              </button>
            )}
          </div>

          {error && <Alert severity="error" title="Error">{error}</Alert>}
        </CardContent>
      </Card>

      {/* SUMMARY KPIs */}
      {summary && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <Card>
            <CardContent className="pt-4">
              <div className="flex items-center justify-between mb-1">
                <CheckCircle2 className="h-4 w-4 text-emerald-500" />
                <span className="text-[10px] font-semibold uppercase text-slate-500 tracking-wide">Procesados</span>
              </div>
              <div className="text-2xl font-bold tabular-nums text-slate-900">{summary.n_ok}</div>
              <div className="text-[11px] text-slate-500">{summary.n_err} con error</div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-4">
              <div className="flex items-center justify-between mb-1">
                <DollarSign className="h-4 w-4 text-emerald-500" />
                <span className="text-[10px] font-semibold uppercase text-slate-500 tracking-wide">Margen actual</span>
              </div>
              <div className="text-2xl font-bold tabular-nums text-slate-900">
                {formatMoney(summary.margen_total_actual, 0)}
              </div>
              <div className="text-[11px] text-slate-500">/día agregado</div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-4">
              <div className="flex items-center justify-between mb-1">
                <TrendingUp className="h-4 w-4 text-indigo-500" />
                <span className="text-[10px] font-semibold uppercase text-slate-500 tracking-wide">Margen con recom.</span>
              </div>
              <div className="text-2xl font-bold tabular-nums text-slate-900">
                {formatMoney(summary.margen_total_recomendado, 0)}
              </div>
              <div className={cn(
                "text-[11px] tabular-nums font-semibold",
                summary.delta_margen_total > 0 ? "text-emerald-600" : "text-rose-600"
              )}>
                {summary.delta_margen_total > 0 ? "▲" : "▼"} {formatPct(summary.delta_margen_pct)}
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-[10px] font-semibold uppercase text-slate-500 tracking-wide">Confianza</span>
              </div>
              <div className="space-y-1 text-xs">
                <ConfBar label="ALTA"  n={summary.confianza_buckets.ALTA}  total={summary.n_ok} color="emerald" />
                <ConfBar label="MEDIA" n={summary.confianza_buckets.MEDIA} total={summary.n_ok} color="amber" />
                <ConfBar label="BAJA"  n={summary.confianza_buckets.BAJA}  total={summary.n_ok} color="rose" />
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* TABLA RESULTADOS */}
      {results.length > 0 && (
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle>Recomendaciones</CardTitle>
                <CardDescription>
                  {results.length} resultados — descarga el CSV para aplicar en tu sistema
                </CardDescription>
              </div>
              <button
                onClick={downloadCSV}
                className="inline-flex items-center gap-2 bg-emerald-600 hover:bg-emerald-700 text-white text-sm font-medium px-4 py-2 rounded-lg shadow-sm transition"
              >
                <FileDown className="h-4 w-4" />
                Descargar CSV
              </button>
            </div>
          </CardHeader>
          <CardContent className="p-0">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-slate-50 text-slate-600 text-xs uppercase sticky top-0">
                  <tr>
                    <th className="text-left px-3 py-3 font-semibold">Producto</th>
                    <th className="text-right px-3 py-3 font-semibold">Costo nuevo</th>
                    <th className="text-right px-3 py-3 font-semibold">P. actual → recom.</th>
                    <th className="text-right px-3 py-3 font-semibold">Rango</th>
                    <th className="text-right px-3 py-3 font-semibold">Δ Margen/día</th>
                    <th className="text-center px-3 py-3 font-semibold">Confianza</th>
                    <th className="text-center px-3 py-3 font-semibold">Modelo</th>
                    <th className="text-center px-3 py-3 font-semibold">Costo</th>
                  </tr>
                </thead>
                <tbody>
                  {results.map((r, i) => (
                    <tr key={`${r.sku}-${i}`} className={cn(
                      "border-t border-slate-100",
                      i % 2 === 0 ? "bg-white" : "bg-slate-50/40",
                      r.status === "error" && "bg-rose-50/50"
                    )}>
                      <td className="px-3 py-2.5">
                        <div className="font-medium text-slate-900 truncate max-w-xs">
                          {r.descripcion || <span className="text-rose-600">{r.error}</span>}
                        </div>
                        <div className="flex items-center gap-1.5 mt-0.5">
                          <span className="text-[10px] font-mono text-slate-400">{r.sku}</span>
                          {r.fam && <Badge variant="info" className="text-[10px] py-0">{r.fam}</Badge>}
                        </div>
                      </td>
                      <td className="px-3 py-2.5 text-right tabular-nums text-slate-700">
                        {r.new_cost ? formatMoney(r.new_cost) : "—"}
                      </td>
                      <td className="px-3 py-2.5 text-right">
                        {r.precio_actual !== null && r.precio_recomendado !== null ? (
                          <div>
                            <div className="text-xs text-slate-500 tabular-nums">{formatMoney(r.precio_actual)}</div>
                            <div className="font-semibold tabular-nums text-slate-900">
                              → {formatMoney(r.precio_recomendado)}
                            </div>
                            <div className={cn(
                              "text-[10px] tabular-nums font-medium",
                              (r.delta_precio_pct ?? 0) > 0 ? "text-rose-600" :
                              (r.delta_precio_pct ?? 0) < 0 ? "text-emerald-600" : "text-slate-500"
                            )}>
                              {formatPct(r.delta_precio_pct ?? 0)}
                            </div>
                          </div>
                        ) : "—"}
                      </td>
                      <td className="px-3 py-2.5 text-right tabular-nums text-xs text-slate-600">
                        {r.rango_lo !== null && r.rango_hi !== null
                          ? `${formatMoney(r.rango_lo)}–${formatMoney(r.rango_hi)}`
                          : "—"}
                      </td>
                      <td className="px-3 py-2.5 text-right">
                        {r.delta_margen !== null ? (
                          <span className={cn(
                            "tabular-nums font-semibold",
                            r.delta_margen > 0 ? "text-emerald-600" : "text-rose-600"
                          )}>
                            {r.delta_margen > 0 ? "+" : ""}{formatMoney(r.delta_margen, 0)}
                          </span>
                        ) : "—"}
                      </td>
                      <td className="px-3 py-2.5 text-center">
                        {r.confianza_nivel ? (
                          <Badge variant={
                            r.confianza_nivel === "ALTA" ? "success" :
                            r.confianza_nivel === "MEDIA" ? "warning" : "error"
                          }>
                            {r.confianza_nivel} {r.confianza_score && `${Math.round(r.confianza_score * 100)}`}
                          </Badge>
                        ) : "—"}
                      </td>
                      <td className="px-3 py-2.5 text-center">
                        {r.modelo_usado && (
                          <Badge variant={
                            r.modelo_usado === "LightGBM" ? "info" :
                            r.modelo_usado === "XGBoost"  ? "warning" : "neutral"
                          }>
                            {r.modelo_usado}
                          </Badge>
                        )}
                      </td>
                      <td className="px-3 py-2.5 text-center">
                        {r.cost_severity === "extreme" && <AlertCircle className="h-4 w-4 text-rose-500 mx-auto" />}
                        {r.cost_severity === "warning" && <AlertCircle className="h-4 w-4 text-amber-500 mx-auto" />}
                        {r.cost_severity === "ok" && <CheckCircle2 className="h-4 w-4 text-emerald-500 mx-auto" />}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}

      {!results.length && (
        <Alert severity="info">
          <b>Caso de uso típico:</b> el proveedor te manda mañana una lista con
          200 SKUs y sus nuevos costos. La cargas aquí, obtienes precios
          recomendados con su zona segura, ordenados por margen ganado, y
          descargas el CSV para aplicar en tu sistema POS o validar con el
          comprador.
        </Alert>
      )}
    </main>
  );
}

function ConfBar({ label, n, total, color }: {
  label: string; n: number; total: number; color: "emerald" | "amber" | "rose";
}) {
  const pct = total > 0 ? (n / total) * 100 : 0;
  const colorClass = {
    emerald: "bg-emerald-500",
    amber: "bg-amber-500",
    rose: "bg-rose-500",
  }[color];
  return (
    <div className="flex items-center gap-2">
      <div className="w-12 text-[10px] font-semibold text-slate-500">{label}</div>
      <div className="flex-1 h-2 bg-slate-100 rounded-full overflow-hidden">
        <div className={cn("h-full transition-all", colorClass)}
          style={{ width: `${pct}%` }} />
      </div>
      <div className="w-8 text-right text-xs font-semibold text-slate-700 tabular-nums">{n}</div>
    </div>
  );
}
