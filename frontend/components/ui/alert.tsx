import { cn } from "@/lib/utils";
import { HTMLAttributes } from "react";
import { AlertTriangle, AlertCircle, CheckCircle2, Info } from "lucide-react";

type Severity = "info" | "success" | "warning" | "error";

const styles: Record<Severity, { box: string; icon: React.ReactNode }> = {
  info: {
    box: "bg-blue-50 border-blue-200 text-blue-900",
    icon: <Info className="h-5 w-5 text-blue-600" />,
  },
  success: {
    box: "bg-emerald-50 border-emerald-200 text-emerald-900",
    icon: <CheckCircle2 className="h-5 w-5 text-emerald-600" />,
  },
  warning: {
    box: "bg-amber-50 border-amber-200 text-amber-900",
    icon: <AlertTriangle className="h-5 w-5 text-amber-600" />,
  },
  error: {
    box: "bg-rose-50 border-rose-200 text-rose-900",
    icon: <AlertCircle className="h-5 w-5 text-rose-600" />,
  },
};

export function Alert({
  severity = "info",
  title,
  children,
  className,
}: {
  severity?: Severity;
  title?: string;
  children?: React.ReactNode;
  className?: string;
}) {
  const s = styles[severity];
  return (
    <div
      className={cn(
        "flex items-start gap-3 rounded-lg border p-4",
        s.box,
        className,
      )}
    >
      <div className="mt-0.5 shrink-0">{s.icon}</div>
      <div className="text-sm leading-relaxed">
        {title && <div className="font-semibold mb-1">{title}</div>}
        {children}
      </div>
    </div>
  );
}
