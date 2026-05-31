"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { Sparkles, BarChart3, Target, Cpu, Home, Users, FileSpreadsheet } from "lucide-react";
import { getBacktestSummary, type BacktestSummary } from "@/lib/api";
import { cn } from "@/lib/utils";

const links = [
  { href: "/",          label: "Recomendar", icon: Home },
  { href: "/sustitutos", label: "Sustitutos", icon: Users },
  { href: "/batch",     label: "Lote (CSV)", icon: FileSpreadsheet },
];

export function Header() {
  const pathname = usePathname();
  const [summary, setSummary] = useState<BacktestSummary | null>(null);

  useEffect(() => {
    getBacktestSummary().then(setSummary).catch(() => {});
  }, []);

  return (
    <header className="border-b border-slate-200 bg-white/80 backdrop-blur-sm sticky top-0 z-40">
      <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between gap-6">
        <Link href="/" className="flex items-center gap-3 shrink-0">
          <div className="h-10 w-10 rounded-xl bg-gradient-to-br from-indigo-500 to-violet-600 flex items-center justify-center shadow-md">
            <Sparkles className="h-5 w-5 text-white" />
          </div>
          <div>
            <div className="text-lg font-bold tracking-tight text-slate-900">PriceIQ</div>
            <div className="text-xs text-slate-500">Recomendación inteligente de precios</div>
          </div>
        </Link>

        <nav className="hidden md:flex items-center gap-1">
          {links.map((l) => {
            const active = pathname === l.href;
            const Icon = l.icon;
            return (
              <Link
                key={l.href}
                href={l.href}
                className={cn(
                  "flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition",
                  active
                    ? "bg-indigo-50 text-indigo-700"
                    : "text-slate-600 hover:bg-slate-100 hover:text-slate-900"
                )}
              >
                <Icon className="h-4 w-4" />
                {l.label}
              </Link>
            );
          })}
        </nav>

        {summary && (
          <div className="hidden lg:flex items-center gap-5 text-xs text-slate-600">
            <div className="flex items-center gap-1.5">
              <BarChart3 className="h-4 w-4 text-indigo-500" />
              <span><b className="text-slate-900">{summary.n_eventos.toLocaleString()}</b> eventos</span>
            </div>
            <div className="flex items-center gap-1.5">
              <Target className="h-4 w-4 text-emerald-500" />
              <span>MAPE <b className="text-slate-900">{summary.mape_mediano.toFixed(1)}%</b></span>
            </div>
            <div className="flex items-center gap-1.5">
              <Cpu className="h-4 w-4 text-violet-500" />
              <span>IC <b className="text-slate-900">{summary.cobertura_ic80.toFixed(0)}%</b></span>
            </div>
          </div>
        )}
      </div>
      <nav className="md:hidden border-t border-slate-100 px-6 py-2 flex gap-2 overflow-x-auto">
        {links.map((l) => {
          const active = pathname === l.href;
          const Icon = l.icon;
          return (
            <Link key={l.href} href={l.href}
              className={cn("flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium whitespace-nowrap",
                active ? "bg-indigo-50 text-indigo-700" : "text-slate-600")}>
              <Icon className="h-3.5 w-3.5" /> {l.label}
            </Link>
          );
        })}
      </nav>
    </header>
  );
}
