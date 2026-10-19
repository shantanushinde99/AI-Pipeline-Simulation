import { useMemo, useState } from "react";
import type { TraceSpan } from "../../types/simulation";
import { Badge, KV, SimulatedLabel, StatusBadge, cx } from "../ui";
import { EventLog } from "./EventLog";

const show = (v: unknown) => (typeof v === "string" ? v : JSON.stringify(v));

/** Waterfall of spans with filters and a detail pane. Data model mirrors OpenTelemetry spans. */
export function TraceInspector({ spans, initial }: { spans: TraceSpan[]; initial?: string }) {
  const [component, setComponent] = useState("all");
  const [severity, setSeverity] = useState("all");
  const [eventType, setEventType] = useState("all");
  const [sel, setSel] = useState<string | undefined>(initial);

  const components = useMemo(() => [...new Set(spans.map((s) => s.component))], [spans]);
  const types = useMemo(() => [...new Set(spans.flatMap((s) => s.events.map((e) => e.type)))], [spans]);
  const end = Math.max(...spans.map((s) => s.startMs + s.durationMs), 1);
  const depth = (s: TraceSpan): number => (s.parentId ? 1 + depth(spans.find((p) => p.id === s.parentId) ?? { ...s, parentId: undefined }) : 0);
  const sevOf = (s: TraceSpan) => (s.status === "error" || s.events.some((e) => e.severity === "error") ? "error" : s.status === "warning" || s.events.some((e) => e.severity === "warning") ? "warning" : "info");

  const visible = spans.filter(
    (s) => (component === "all" || s.component === component) && (severity === "all" || sevOf(s) === severity) && (eventType === "all" || s.events.some((e) => e.type === eventType)),
  );
  const selected = spans.find((s) => s.id === sel) ?? null;
  const select = (label: string, value: string, set: (v: string) => void, opts: string[]) => (
    <label className="flex items-center gap-1 text-xs text-muted">
      {label}
      <select value={value} onChange={(e) => set(e.target.value)} className="h-7 rounded border border-line bg-surface px-1 text-xs text-fg">
        <option value="all">all</option>
        {opts.map((o) => <option key={o} value={o}>{o}</option>)}
      </select>
    </label>
  );

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-3">
        <SimulatedLabel>Simulated trace</SimulatedLabel>
        {select("Component", component, setComponent, components)}
        {select("Severity", severity, setSeverity, ["info", "warning", "error"])}
        {select("Event type", eventType, setEventType, types)}
        <span className="text-xs text-muted">{visible.length}/{spans.length} spans · times are simulated ms</span>
      </div>
      <div className="grid gap-3 lg:grid-cols-[minmax(0,1.3fr)_minmax(0,1fr)]">
        <ul className="divide-y divide-line overflow-hidden rounded-md border border-line" aria-label="Spans">
          {visible.length === 0 && <li className="p-3 text-xs text-muted">No spans match these filters.</li>}
          {visible.map((s) => (
            <li key={s.id}>
              <button onClick={() => setSel(s.id)} aria-pressed={sel === s.id} className={cx("grid w-full grid-cols-[minmax(0,1fr)_90px] items-center gap-2 px-2 py-1.5 text-left text-xs hover:bg-surface-2", sel === s.id && "bg-accent-soft")}>
                <span className="flex min-w-0 items-center gap-1.5" style={{ paddingLeft: depth(s) * 12 }}>
                  <StatusBadge status={s.status === "ok" ? "ok" : s.status} label={s.status} />
                  <span className="truncate font-mono">{s.name}</span>
                  {s.events.some((e) => e.severity === "warning" || e.severity === "error") && <Badge tone="warn">!</Badge>}
                </span>
                <span className="relative h-3 rounded-sm bg-surface-2" title={`${s.startMs.toFixed(1)}–${(s.startMs + s.durationMs).toFixed(1)} ms`}>
                  <span className={cx("absolute inset-y-0 rounded-sm", s.status === "error" ? "bg-err" : s.status === "warning" ? "bg-warn" : "bg-accent/70")}
                    style={{ left: `${(s.startMs / end) * 100}%`, width: `${Math.max(1.5, (s.durationMs / end) * 100)}%` }} />
                </span>
              </button>
            </li>
          ))}
        </ul>
        <div className="min-w-0 rounded-md border border-line p-3" aria-live="polite">
          {!selected ? (
            <p className="text-xs text-muted">Select a span to see its inputs, outputs and events.</p>
          ) : (
            <div className="space-y-3">
              <div className="flex flex-wrap items-center gap-2"><span className="font-mono text-sm font-semibold">{selected.name}</span><Badge>{selected.component}</Badge></div>
              <KV rows={[["status", selected.status], ["start", `${selected.startMs.toFixed(1)} ms`], ["duration", `${selected.durationMs.toFixed(1)} ms`], ["parent", selected.parentId ?? "—"]]} />
              {selected.input && <div><h4 className="mb-1 text-[11px] font-semibold uppercase text-muted">Input</h4><KV rows={Object.entries(selected.input).map(([k, v]) => [k, show(v)])} /></div>}
              {selected.output && <div><h4 className="mb-1 text-[11px] font-semibold uppercase text-muted">Output</h4><KV rows={Object.entries(selected.output).map(([k, v]) => [k, show(v)])} /></div>}
              {selected.metrics && <div><h4 className="mb-1 text-[11px] font-semibold uppercase text-muted">Metrics</h4><KV rows={Object.entries(selected.metrics).map(([k, v]) => [k, show(v)])} /></div>}
              <div><h4 className="mb-1 text-[11px] font-semibold uppercase text-muted">Events (logs)</h4><EventLog events={selected.events} /></div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
