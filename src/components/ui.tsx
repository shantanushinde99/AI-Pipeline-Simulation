import type { ButtonHTMLAttributes, ReactNode } from "react";
import { useId } from "react";
import { motion } from "motion/react";
import { AlertTriangle, CheckCircle2, Circle, CircleDot, XCircle } from "lucide-react";

export const cx = (...c: (string | false | null | undefined)[]) => c.filter(Boolean).join(" ");

type Tone = "neutral" | "accent" | "ok" | "warn" | "err";
const toneCls: Record<Tone, string> = {
  neutral: "bg-surface-2 text-muted border-line",
  accent: "bg-accent-soft text-accent border-accent/30",
  ok: "bg-ok-soft text-ok border-ok/30",
  warn: "bg-warn-soft text-warn border-warn/30",
  err: "bg-err-soft text-err border-err/30",
};

export function Badge({ tone = "neutral", children, className }: { tone?: Tone; children: ReactNode; className?: string }) {
  return (
    <span className={cx("inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 text-[11px] font-medium leading-none whitespace-nowrap", toneCls[tone], className)}>
      {children}
    </span>
  );
}

export type Status = "pending" | "active" | "ok" | "warning" | "error";
/** Status is always shown as icon + text, never color alone. */
export function StatusBadge({ status, label }: { status: Status; label?: string }) {
  const map = {
    pending: { tone: "neutral", Icon: Circle, text: "Pending" },
    active: { tone: "accent", Icon: CircleDot, text: "Running" },
    ok: { tone: "ok", Icon: CheckCircle2, text: "OK" },
    warning: { tone: "warn", Icon: AlertTriangle, text: "Warning" },
    error: { tone: "err", Icon: XCircle, text: "Error" },
  } as const;
  const { tone, Icon, text } = map[status];
  return (
    <Badge tone={tone}>
      <Icon size={11} aria-hidden />
      {label ?? text}
    </Badge>
  );
}

type BtnProps = ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "primary" | "secondary" | "ghost"; size?: "sm" | "md" };
export function Button({ variant = "secondary", size = "md", className, ...rest }: BtnProps) {
  return (
    <button
      {...rest}
      className={cx(
        "inline-flex items-center justify-center gap-1.5 rounded-md font-medium transition-[color,background-color,border-color,transform] duration-150 active:scale-[0.97] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent disabled:opacity-40 disabled:cursor-not-allowed",
        size === "sm" ? "h-7 px-2 text-xs" : "h-8 px-3 text-sm",
        variant === "primary" && "bg-accent text-white dark:text-black hover:opacity-90",
        variant === "secondary" && "border border-line bg-surface hover:bg-surface-2",
        variant === "ghost" && "hover:bg-surface-2 text-muted hover:text-fg",
        className,
      )}
    />
  );
}

export function Panel({ title, actions, children, className }: { title?: ReactNode; actions?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={cx("rounded-lg border border-line bg-surface", className)}>
      {title && (
        <header className="flex items-center justify-between gap-2 border-b border-line px-3 py-2">
          <h3 className="text-xs font-semibold uppercase tracking-wide text-muted">{title}</h3>
          {actions}
        </header>
      )}
      <div className="p-3">{children}</div>
    </section>
  );
}

export function Slider({ label, value, min, max, step = 1, onChange, format = String, hint }: {
  label: ReactNode; value: number; min: number; max: number; step?: number; onChange: (v: number) => void; format?: (v: number) => string; hint?: string;
}) {
  const id = useId();
  return (
    <div className="space-y-1">
      <div className="flex items-baseline justify-between gap-2 text-xs">
        <label htmlFor={id} className="font-medium">{label}</label>
        <span className="font-mono text-muted">{format(value)}</span>
      </div>
      <input id={id} type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} className="w-full accent-[var(--accent)]" />
      {hint && <p className="text-[11px] text-muted">{hint}</p>}
    </div>
  );
}

export function Select<T extends string>({ label, value, options, onChange }: {
  label: string; value: T; options: { value: T; label: string }[]; onChange: (v: T) => void;
}) {
  const id = useId();
  return (
    <div className="space-y-1">
      <label htmlFor={id} className="block text-xs font-medium">{label}</label>
      <select id={id} value={value} onChange={(e) => onChange(e.target.value as T)} className="h-8 w-full rounded-md border border-line bg-surface px-2 text-sm">
        {options.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
      </select>
    </div>
  );
}

export function Toggle({ label, checked, onChange, hint }: { label: string; checked: boolean; onChange: (v: boolean) => void; hint?: string }) {
  return (
    <label className="flex cursor-pointer items-start gap-2 text-xs">
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} className="mt-0.5 accent-[var(--accent)]" />
      <span>
        <span className="font-medium">{label}</span>
        {hint && <span className="block text-[11px] text-muted">{hint}</span>}
      </span>
    </label>
  );
}

export function NumberField({ label, value, onChange, step = 0.1, min, max }: { label: string; value: number; onChange: (v: number) => void; step?: number; min?: number; max?: number }) {
  const id = useId();
  return (
    <div className="space-y-1">
      <label htmlFor={id} className="block text-xs font-medium">{label}</label>
      <input
        id={id}
        type="number"
        value={value}
        step={step}
        min={min}
        max={max}
        onChange={(e) => {
          const v = Number(e.target.value);
          if (e.target.value !== "" && Number.isFinite(v)) onChange(min !== undefined || max !== undefined ? Math.min(max ?? v, Math.max(min ?? v, v)) : v);
        }}
        className="h-8 w-full rounded-md border border-line bg-surface px-2 font-mono text-sm"
      />
    </div>
  );
}

export function KV({ rows }: { rows: [ReactNode, ReactNode][] }) {
  return (
    <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs">
      {rows.map(([k, v], i) => (
        <div key={i} className="contents">
          <dt className="text-muted">{k}</dt>
          <dd className="min-w-0 break-words text-right font-mono">{v}</dd>
        </div>
      ))}
    </dl>
  );
}

export function Formula({ children }: { children: ReactNode }) {
  return <div className="overflow-x-auto rounded-md bg-surface-2 px-2.5 py-2 font-mono text-xs leading-relaxed whitespace-pre">{children}</div>;
}

export function CodeBlock({ code, title }: { code: string; title?: string }) {
  return (
    <figure className="overflow-hidden rounded-lg border border-line">
      {title && <figcaption className="border-b border-line bg-surface-2 px-3 py-1.5 font-mono text-xs text-muted">{title}</figcaption>}
      <pre className="overflow-x-auto bg-surface p-3 font-mono text-xs leading-relaxed"><code>{code}</code></pre>
    </figure>
  );
}

export function Tabs<T extends string>({ tabs, value, onChange, label }: { tabs: { id: T; label: string }[]; value: T; onChange: (v: T) => void; label: string }) {
  const underline = useId();
  return (
    <div role="tablist" aria-label={label} className="flex gap-1 overflow-x-auto border-b border-line">
      {tabs.map((t) => (
        <button
          key={t.id}
          role="tab"
          aria-selected={value === t.id}
          onClick={() => onChange(t.id)}
          className={cx("relative -mb-px shrink-0 px-3 py-2 text-sm font-medium transition-colors", value === t.id ? "text-fg" : "text-muted hover:text-fg")}
        >
          {t.label}
          {value === t.id && <motion.span layoutId={underline} className="absolute inset-x-0 -bottom-px h-0.5 rounded bg-accent" transition={{ type: "spring", stiffness: 500, damping: 40 }} />}
        </button>
      ))}
    </div>
  );
}

export function Callout({ tone = "neutral", title, children }: { tone?: Tone; title?: string; children: ReactNode }) {
  const Icon = tone === "warn" ? AlertTriangle : tone === "err" ? XCircle : tone === "ok" ? CheckCircle2 : CircleDot;
  return (
    <div className={cx("flex gap-2 rounded-md border p-2.5 text-xs leading-relaxed", toneCls[tone])}>
      <Icon size={14} className="mt-0.5 shrink-0" aria-hidden />
      <div className="text-fg">
        {title && <div className="mb-0.5 font-semibold">{title}</div>}
        {children}
      </div>
    </div>
  );
}

export function SimulatedLabel({ children = "Simulated" }: { children?: ReactNode }) {
  return <Badge tone="warn" className="uppercase tracking-wide">{children}</Badge>;
}
