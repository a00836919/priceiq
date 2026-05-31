import { cn } from "@/lib/utils";
import { ShieldCheck, ShieldAlert, ShieldX } from "lucide-react";
import type { Recommendation } from "@/lib/api";

const STYLES = {
  ALTA: {
    box: "bg-emerald-50 border-emerald-200",
    text: "text-emerald-900",
    label: "text-emerald-700",
    bar: "bg-emerald-500",
    icon: <ShieldCheck className="h-7 w-7 text-emerald-600" />,
  },
  MEDIA: {
    box: "bg-amber-50 border-amber-200",
    text: "text-amber-900",
    label: "text-amber-700",
    bar: "bg-amber-500",
    icon: <ShieldAlert className="h-7 w-7 text-amber-600" />,
  },
  BAJA: {
    box: "bg-rose-50 border-rose-200",
    text: "text-rose-900",
    label: "text-rose-700",
    bar: "bg-rose-500",
    icon: <ShieldX className="h-7 w-7 text-rose-600" />,
  },
};

export function ConfidenceCard({ rec }: { rec: Recommendation }) {
  const c = rec.confianza;
  const s = STYLES[c.nivel];
  const pct = Math.round(c.score * 100);

  return (
    <div className={cn("rounded-xl border p-5", s.box)}>
      <div className="flex items-start gap-4">
        <div className="shrink-0">{s.icon}</div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between gap-3 mb-1">
            <div className={cn("text-xs font-bold uppercase tracking-wider", s.label)}>
              Confianza {c.nivel}
            </div>
            <div className={cn("text-2xl font-bold tabular-nums", s.text)}>
              {pct}<span className="text-base font-medium opacity-60">/100</span>
            </div>
          </div>
          <div className="h-1.5 bg-white/60 rounded-full overflow-hidden mb-3">
            <div
              className={cn("h-full transition-all", s.bar)}
              style={{ width: `${pct}%` }}
            />
          </div>
          <p className={cn("text-sm leading-relaxed", s.text)}>{c.texto}</p>
        </div>
      </div>
    </div>
  );
}
