"use client";
import { useEffect, useState } from "react";
import { Search } from "lucide-react";
import { listSkus, type Sku } from "@/lib/api";
import { Badge } from "@/components/ui/badge";

export function SkuSelector({
  value,
  onChange,
}: {
  value: string | null;
  onChange: (sku: Sku) => void;
}) {
  const [q, setQ] = useState("");
  const [items, setItems] = useState<Sku[]>([]);
  const [total, setTotal] = useState(0);
  const [open, setOpen] = useState(false);
  const [selected, setSelected] = useState<Sku | null>(null);

  useEffect(() => {
    const t = setTimeout(async () => {
      const r = await listSkus(q || undefined, 30);
      setItems(r.items);
      setTotal(r.total);
    }, 200);
    return () => clearTimeout(t);
  }, [q]);

  return (
    <div className="relative">
      <label className="block text-xs font-semibold text-slate-600 mb-1.5 uppercase tracking-wide">
        Producto
      </label>
      <div className="relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400" />
        <input
          type="text"
          placeholder={selected ? selected.descripcion : "Buscar SKU o descripción..."}
          value={q}
          onChange={(e) => {
            setQ(e.target.value);
            setOpen(true);
          }}
          onFocus={() => setOpen(true)}
          onBlur={() => setTimeout(() => setOpen(false), 200)}
          className="w-full pl-9 pr-3 py-2.5 border border-slate-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 bg-white"
        />
      </div>
      {selected && !open && (
        <div className="mt-2 flex items-center gap-2">
          <Badge variant="info">{selected.fam}</Badge>
          <Badge variant="neutral">{selected.marca}</Badge>
          <span className="text-xs text-slate-500 font-mono">{selected.cve_art}</span>
        </div>
      )}

      {open && items.length > 0 && (
        <div className="absolute z-50 mt-1 w-full max-h-80 overflow-y-auto bg-white border border-slate-200 rounded-lg shadow-xl">
          <div className="px-3 py-2 text-xs text-slate-500 border-b border-slate-100">
            {total} resultados
          </div>
          {items.map((s) => (
            <button
              key={s.cve_art}
              onClick={() => {
                setSelected(s);
                onChange(s);
                setQ("");
                setOpen(false);
              }}
              className="w-full text-left px-3 py-2 hover:bg-indigo-50 transition border-b border-slate-50 last:border-0"
            >
              <div className="text-sm font-medium text-slate-800 truncate">
                {s.descripcion}
              </div>
              <div className="flex items-center gap-1.5 mt-0.5">
                <span className="text-[10px] font-mono text-slate-400">{s.cve_art}</span>
                <Badge variant="info" className="text-[10px] py-0">{s.fam}</Badge>
                <span className="text-[10px] text-slate-500">{s.marca}</span>
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
