"use client";
import {
  ComposedChart, Area, Line, XAxis, YAxis, Tooltip, ReferenceLine,
  ReferenceArea, ResponsiveContainer, CartesianGrid,
} from "recharts";
import { formatMoney, formatNumber } from "@/lib/utils";
import type { Recommendation } from "@/lib/api";

export function DemandChart({ rec, mode = "demand" }: {
  rec: Recommendation; mode?: "demand" | "margin";
}) {
  const data = rec.curve.map((p) => ({
    precio: p.precio,
    qty: p.qty_med,
    qty_lo: p.qty_lo,
    qty_hi: p.qty_hi,
    margen: p.margen_med,
    margen_lo: p.margen_lo,
    margen_hi: p.margen_hi,
    margen_pen: p.margen_med_penalizado,
    credibilidad: p.credibilidad,
  }));

  const isDemand = mode === "demand";

  return (
    <div className="h-72 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={data} margin={{ top: 10, right: 12, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis
            dataKey="precio"
            tick={{ fontSize: 11, fill: "#64748b" }}
            tickFormatter={(v) => `$${v.toFixed(0)}`}
            label={{ value: "Precio ($)", position: "insideBottom", offset: -5, fontSize: 11, fill: "#64748b" }}
          />
          <YAxis
            tick={{ fontSize: 11, fill: "#64748b" }}
            tickFormatter={(v) => isDemand ? formatNumber(v, 0) : `$${v.toFixed(0)}`}
            label={{ value: isDemand ? "Unidades/día" : "Margen $/día", angle: -90, position: "insideLeft", fontSize: 11, fill: "#64748b" }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "white", border: "1px solid #e2e8f0",
              borderRadius: "8px", fontSize: "12px", padding: "8px 12px",
            }}
            formatter={((value: unknown, name: unknown) => {
              const labels: Record<string, string> = {
                qty: "Demanda esperada",
                margen: "Margen esperado",
                margen_pen: "Margen ajustado",
              };
              const num = typeof value === "number" ? value : 0;
              const v = isDemand ? formatNumber(num, 1) : formatMoney(num);
              return [v, labels[String(name)] || String(name)];
            }) as never}
            labelFormatter={(p) => `Precio: ${formatMoney(Number(p) || 0)}`}
          />

          {/* Banda verde del rango recomendado */}
          <ReferenceArea
            x1={rec.rango.lo}
            x2={rec.rango.hi}
            fill="#10b981"
            fillOpacity={0.10}
            stroke="#10b981"
            strokeOpacity={0.3}
          />

          {/* IC */}
          {isDemand ? (
            <>
              <Area
                type="monotone"
                dataKey="qty_hi"
                stroke="none"
                fill="#3b82f6"
                fillOpacity={0.1}
              />
              <Area
                type="monotone"
                dataKey="qty_lo"
                stroke="none"
                fill="#ffffff"
                fillOpacity={1}
              />
              <Line type="monotone" dataKey="qty" stroke="#2563eb" strokeWidth={2.5} dot={false} />
            </>
          ) : (
            <>
              <Area
                type="monotone"
                dataKey="margen_hi"
                stroke="none"
                fill="#10b981"
                fillOpacity={0.12}
              />
              <Area
                type="monotone"
                dataKey="margen_lo"
                stroke="none"
                fill="#ffffff"
                fillOpacity={1}
              />
              <Line type="monotone" dataKey="margen" stroke="#059669" strokeWidth={2.5} dot={false} />
              <Line type="monotone" dataKey="margen_pen" stroke="#7c3aed" strokeWidth={1.5} strokeDasharray="4 3" dot={false} />
            </>
          )}

          {/* Líneas verticales */}
          <ReferenceLine
            x={rec.precio_actual}
            stroke="#64748b"
            strokeDasharray="5 5"
            label={{ value: "Actual", position: "top", fontSize: 10, fill: "#64748b" }}
          />
          <ReferenceLine
            x={rec.precio_recomendado}
            stroke="#059669"
            strokeWidth={2}
            label={{ value: "Recomendado", position: "top", fontSize: 10, fill: "#059669", fontWeight: 600 }}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
