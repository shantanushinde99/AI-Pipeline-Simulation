import { useMemo, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { CheckCircle2, RotateCcw, Timer, XCircle } from "lucide-react";
import { usePlayback } from "../../hooks/usePlayback";
import { PlaybackControls } from "../../components/visualizer/PlaybackControls";
import { EventLog } from "../../components/visualizer/EventLog";
import { StepExplainer, Workspace } from "../../components/visualizer/Workspace";
import { TraceInspector } from "../../components/visualizer/TraceInspector";
import { AnimatedNumber } from "../../components/visualizer/AnimatedNumber";
import { Badge, Button, KV, Panel, Select, SimulatedLabel, Slider, Toggle, cx } from "../../components/ui";
import { TexBlock } from "../../components/Tex";
import type { TraceSpan } from "../../types/simulation";
import type { VisualizerTab } from "../types";
import { StepCaption, ease, useReveal } from "../monitoring-and-alerting/shared";
import { DEFAULT_INPUT, PRESETS, SERVICES, SLOW_MS, simulate, spanEnd, type ServiceId, type TraceInput, type TraceRun, type TraceState } from "./simulation";

/** One schedule (seconds at 1×): header packets travel, then the simulated clock sweeps t0 → t1 while bars grow. */
const TRAVEL = 0.7;
const clockSecs = (s: TraceState) => (s.phase === "hop" ? Math.min(3, Math.max(0.9, (s.t1 - s.t0) / 350)) : 0);
const stepMs = (s: TraceState) => (s.phase === "hop" ? (TRAVEL + clockSecs(s) + 0.9) * 1000 : s.phase === "bottleneck" ? 3600 : 3800);

export default function TracingVisualizer({ tab }: { tab: VisualizerTab }) {
  const [input, setInput] = useState<TraceInput>(DEFAULT_INPUT);
  const [replay, setReplay] = useState(0);
  const run = useMemo(() => simulate(input), [input]);
  const steps = run.steps;
  const pb = usePlayback(steps.length, (i) => stepMs(steps[i].state));
  const step = steps[pb.index];
  const s = step.state;
  const k = 1 / pb.speed;
  const animKey = `${step.id}-${replay}-${input.preset}-${input.parallel}`;
  const set = (p: Partial<TraceInput>) => setInput((v) => ({ ...v, ...p }));
  const finished = run.spans.filter((x) => spanEnd(x) <= s.t1);

  if (tab === "metrics") return null;

  return (
    <Workspace
      main={
        <>
          <Panel title="One request, many services" actions={<SimulatedLabel>Simulated services & timings</SimulatedLabel>}>
            <StepCaption index={pb.index} total={steps.length} id={step.id} headline={s.headline} sub={s.sub} onReplay={() => setReplay((r) => r + 1)} />
            <div className="overflow-x-auto">
              <div className="min-w-[600px] space-y-2">
                <ServiceMap run={run} s={s} animKey={animKey} k={k} />
                <Waterfall run={run} s={s} animKey={animKey} k={k} />
              </div>
            </div>
          </Panel>
          <Panel title="What's happening in this step">
            <AnimatePresence mode="wait" initial={false}>
              <motion.div key={animKey} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: 0.2 }}>
                {s.phase === "hop" ? <HopCard run={run} s={s} k={k} /> : s.phase === "bottleneck" ? <BottleneckCard run={run} k={k} /> : <SamplingCard run={run} input={input} animKey={animKey} k={k} />}
              </motion.div>
            </AnimatePresence>
          </Panel>
          <PlaybackControls pb={pb} stepTitles={steps.map((x) => x.title)} />
          {pb.index === 0 && !pb.playing && <p className="-mt-1 text-xs text-muted">Press <strong>Play</strong> to follow the request hop by hop. Watch the <code>traceparent</code> header and the waterfall grow.</p>}
          <Panel title="Trace detail (finished spans)">
            {finished.length ? <TraceInspector key={`${animKey}-${finished.length}`} spans={finished} initial={s.focus.find((id) => finished.some((f) => f.id === id)) ?? finished[0].id} />
              : <p className="text-xs text-muted">No span has finished yet. Spans are exported when they end.</p>}
          </Panel>
          <Panel title="Scenario & sampling" actions={<Button size="sm" variant="ghost" onClick={() => { setInput(DEFAULT_INPUT); pb.restart(); }}><RotateCcw size={12} /> Reset</Button>}>
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
              <Select label="Request preset" value={input.preset} options={PRESETS.map((p) => ({ value: p.id, label: p.name }))} onChange={(v) => set({ preset: v })} />
              <Slider label="Head sampling rate" value={input.samplingRate} min={0} max={1} step={0.05} format={(v) => `${Math.round(v * 100)}%`} onChange={(v) => set({ samplingRate: v })} hint="Decided at the gateway from the trace id." />
              <Slider label="Request seed" value={input.seed} min={1} max={20} onChange={(v) => set({ seed: v })} />
              <Toggle label="Parallel fan-out to shards" checked={input.parallel} onChange={(v) => set({ parallel: v })} hint="Break it: turn off and the shard calls run in sequence." />
              <Toggle label={`Tail sampling (keep errors and > ${SLOW_MS} ms)`} checked={input.tailSampling} onChange={(v) => set({ tailSampling: v })} hint="The collector decides after the trace ends." />
            </div>
          </Panel>
        </>
      }
      inspector={
        <>
          <StepExplainer index={pb.index} title={step.title} what={step.description}
            why="A trace turns 'the request was slow' into 'this span on this service was slow', across process boundaries, because every hop carries the same trace id." />
          <Panel title="Latency math">
            <LatencyMath run={run} parallel={input.parallel} />
          </Panel>
          <Panel title="Key values">
            <KV rows={[
              ["trace id", <span className="break-all">{run.traceId}</span>],
              ["sampled flag", run.sampled ? "01 (record & export)" : "00 (head-dropped)"],
              ["spans", run.spans.length],
              ["end-to-end", `${run.total} ms`],
              ["simulated clock", `${Math.round(s.t1)} ms`],
            ]} />
          </Panel>
          <Panel title="Events">
            <EventLog events={step.events} />
          </Panel>
        </>
      }
    />
  );
}

const POS: Record<ServiceId, [number, number]> = { gateway: [90, 112], retriever: [310, 42], "vector-db": [560, 42], reranker: [310, 112], llm: [310, 182], tool: [560, 182] };
const CTRL: Partial<Record<string, [number, number]>> = { "gateway>tool": [330, 300] };
const NW = 136, NH = 38;
const edgeKey = (a: ServiceId, b: ServiceId) => `${a}>${b}`;
const EDGES: [ServiceId, ServiceId][] = [["gateway", "retriever"], ["retriever", "vector-db"], ["gateway", "reranker"], ["gateway", "llm"], ["gateway", "tool"]];
function edgePoints(a: ServiceId, b: ServiceId, n = 14): [number, number][] {
  const [x0, y0] = POS[a], [x2, y2] = POS[b];
  const [x1, y1] = CTRL[edgeKey(a, b)] ?? [(x0 + x2) / 2, (y0 + y2) / 2];
  return Array.from({ length: n }, (_, i) => { const t = i / (n - 1); return [(1 - t) ** 2 * x0 + 2 * (1 - t) * t * x1 + t * t * x2, (1 - t) ** 2 * y0 + 2 * (1 - t) * t * y1 + t * t * y2]; });
}
const shortHeader = (h: string) => { const [v, tr, p, f] = h.split("-"); return `${v}-${tr.slice(0, 4)}…-${p.slice(0, 6)}…-${f}`; };

function ServiceMap({ run, s, animKey, k }: { run: TraceRun; s: TraceState; animKey: string; k: number }) {
  const running = new Set(run.spans.filter((x) => x.startMs < s.t1 && spanEnd(x) > s.t0 && s.phase === "hop").map((x) => x.component));
  const touched = new Set(run.spans.filter((x) => x.startMs < s.t1).map((x) => x.component));
  const errored = new Set(run.spans.filter((x) => x.status === "error" && spanEnd(x) <= s.t1).map((x) => x.component));
  return (
    <svg viewBox="0 0 720 236" className="h-auto w-full" role="img" aria-label="Service map with the trace context travelling between services">
      {EDGES.map(([a, b]) => {
        const pts = edgePoints(a, b);
        const active = s.hops.some((h) => h.from === a && h.to === b);
        return <polyline key={edgeKey(a, b)} points={pts.map((p) => p.join(",")).join(" ")} fill="none" stroke={active ? "var(--accent)" : "var(--line)"} strokeWidth={active ? 2.5 : 1.5} />;
      })}
      {(Object.keys(POS) as ServiceId[]).map((id) => {
        const [x, y] = POS[id];
        const on = running.has(id);
        return (
          <g key={id}>
            {on && <motion.rect x={x - NW / 2} y={y - NH / 2} width={NW} height={NH} rx={9} fill="none" stroke="var(--accent)" strokeWidth={2}
              initial={{ opacity: 0.7, scale: 1 }} animate={{ opacity: 0, scale: 1.18 }} transition={{ duration: 1.3, repeat: Infinity, ease: "easeOut" }} style={{ transformOrigin: `${x}px ${y}px` }} />}
            <rect x={x - NW / 2} y={y - NH / 2} width={NW} height={NH} rx={9} fill={on ? "var(--accent-soft)" : "var(--surface)"} stroke={errored.has(id) ? "var(--err)" : on ? "var(--accent)" : "var(--line)"} strokeWidth={on ? 2 : 1.3} />
            <text x={x} y={y + 4} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--fg)">{SERVICES.find((v) => v.id === id)!.label}</text>
            {touched.has(id) && !on && <text x={x + NW / 2 - 8} y={y - NH / 2 + 12} textAnchor="end" fontSize={10} fill={errored.has(id) ? "var(--err)" : "var(--ok)"}>{errored.has(id) ? "✕" : "✓"}</text>}
          </g>
        );
      })}
      {s.hops.map((h, i) => {
        const pts = edgePoints(h.from, h.to);
        const label = shortHeader(h.header);
        const w = label.length * 6.2 + 14;
        return (
          <motion.g key={`${animKey}-${i}`} initial={{ x: pts[0][0], y: pts[0][1], opacity: 0 }} animate={{ x: pts.map((p) => p[0]), y: pts.map((p) => p[1] + (i - (s.hops.length - 1) / 2) * 16), opacity: [0, 1, 1, 1, 0] }}
            transition={{ duration: TRAVEL * k, delay: i * 0.08 * k, ease: "easeInOut" }}>
            <circle r={6} fill="var(--accent)" stroke="var(--surface)" strokeWidth={2} />
            <rect x={-w / 2} y={-27} width={w} height={17} rx={8} fill="var(--fg)" />
            <text y={-15} textAnchor="middle" fontSize={10.5} fontFamily="var(--font-mono)" fill="var(--surface)">{label}</text>
          </motion.g>
        );
      })}
    </svg>
  );
}

const depthOf = (spans: TraceSpan[], sp: TraceSpan): number => { let d = 0, p = sp.parentId; while (p) { d++; p = spans.find((x) => x.id === p)?.parentId; } return d; };

function Waterfall({ run, s, animKey, k }: { run: TraceRun; s: TraceState; animKey: string; k: number }) {
  const RH = 23, LBL = 220, X0 = LBL + 6, X1 = 700, top = 22;
  const H = top + run.spans.length * RH + 8;
  const x = (ms: number) => X0 + (ms / run.total) * (X1 - X0);
  const D = clockSecs(s) * k;
  const span = Math.max(1e-9, s.t1 - s.t0);
  const crit = new Map(run.critical.map((c) => [c.id, c.self]));
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => Math.round(f * run.total));
  return (
    <svg viewBox={`0 0 720 ${H}`} className="h-auto w-full" role="img" aria-label={`Trace waterfall with ${run.spans.length} spans over ${run.total} ms`}>
      {ticks.map((t) => (
        <g key={t}>
          <line x1={x(t)} x2={x(t)} y1={top - 4} y2={H - 6} stroke="var(--line)" />
          <text x={x(t)} y={top - 8} textAnchor="middle" fontSize={10} fill="var(--muted)" fontFamily="var(--font-mono)">{t} ms</text>
        </g>
      ))}
      {run.spans.map((sp, i) => {
        const y = top + i * RH;
        const end = spanEnd(sp);
        const started = sp.startMs < s.t1 || (sp.startMs === 0 && s.t1 >= 0);
        const w0 = Math.max(0, Math.min(sp.durationMs, s.t0 - sp.startMs));
        const w1 = Math.max(0, Math.min(sp.durationMs, s.t1 - sp.startMs));
        const growStart = Math.max(sp.startMs, s.t0), growEnd = Math.min(end, s.t1);
        const delay = s.phase === "hop" ? TRAVEL * k + (D * (growStart - s.t0)) / span : 0;
        const dur = s.phase === "hop" ? Math.max(0.01, (D * Math.max(0, growEnd - growStart)) / span) : 0;
        const onCrit = crit.has(sp.id);
        const faded = s.phase === "bottleneck" && !onCrit;
        const color = sp.status === "error" ? "var(--err)" : sp.status === "warning" ? "var(--warn)" : sp.component === "llm" ? "var(--violet)" : "var(--accent)";
        const focus = s.focus.includes(sp.id) && s.phase === "hop";
        const pxW = (w: number) => (w / run.total) * (X1 - X0);
        return (
          <g key={sp.id} opacity={started ? (faded ? 0.3 : 1) : 0.35}>
            {focus && <rect x={0} y={y} width={720} height={RH - 2} fill="var(--accent-soft)" rx={3} />}
            <text x={6 + depthOf(run.spans, sp) * 12} y={y + 15} fontSize={11} fontFamily="var(--font-mono)" fill="var(--fg)" fontWeight={focus ? 700 : 400}>
              {sp.status === "error" ? "✕ " : ""}{sp.name}
            </text>
            <text x={LBL} y={y + 15} textAnchor="end" fontSize={9.5} fill="var(--muted)">{sp.component}</text>
            <rect x={x(sp.startMs)} y={y + 5} width={pxW(sp.durationMs)} height={RH - 12} rx={3} fill="var(--surface-2)" />
            {started && (
              <motion.rect key={`${animKey}-bar`} x={x(sp.startMs)} y={y + 4} height={RH - 10} rx={3} fill={color}
                initial={{ width: pxW(s.phase === "hop" ? w0 : w1) }} animate={{ width: pxW(w1) }} transition={{ delay, duration: dur, ease: "linear" }} />
            )}
            {w1 >= sp.durationMs && started && (
              <motion.text key={`${animKey}-lbl`} x={Math.min(x(end) + 4, 690)} y={y + 15} fontSize={10} fontFamily="var(--font-mono)" fill="var(--fg)" textAnchor={x(end) + 4 > 660 ? "end" : "start"}
                initial={{ opacity: s.phase === "hop" && end > s.t0 ? 0 : 1 }} animate={{ opacity: 1 }} transition={{ delay: delay + dur }}>
                {s.phase === "bottleneck" && onCrit ? `self ${crit.get(sp.id)} ms` : `${sp.durationMs} ms`}
              </motion.text>
            )}
          </g>
        );
      })}
      {s.phase === "hop" && (
        <motion.g key={`${animKey}-now`} initial={{ x: x(s.t0) }} animate={{ x: x(s.t1) }} transition={{ delay: TRAVEL * k, duration: D, ease: "linear" }}>
          <line x1={0} x2={0} y1={top - 2} y2={H - 6} stroke="var(--err)" strokeWidth={1.5} />
          <rect x={-34} y={H - 18} width={68} height={15} rx={4} fill="var(--err)" />
          <text y={H - 7} textAnchor="middle" fontSize={10} fontFamily="var(--font-mono)" fill="var(--surface)">
            t=<AnimatedNumber key={animKey} svg value={s.t1} from={s.t0} digits={0} delay={TRAVEL * k} duration={D} />
          </text>
        </motion.g>
      )}
    </svg>
  );
}

function HopCard({ run, s, k }: { run: TraceRun; s: TraceState; k: number }) {
  const header = s.hops[0]?.header ?? `00-${run.traceId}-${"0".repeat(16)}-${run.sampled ? "01" : "00"}`;
  const [v, tr, parent, flags] = header.split("-");
  const started = run.spans.filter((x) => x.startMs < s.t1 || x.startMs === 0);
  const parts = [
    { text: v, label: "version" },
    { text: tr, label: "trace-id · same on every hop" },
    { text: s.hops.length ? parent : "(none: this is the root)", label: "parent-id · the caller's span id", hot: true },
    { text: flags, label: flags === "01" ? "flags · sampled" : "flags · not sampled" },
  ];
  return (
    <div className="space-y-3">
      <div>
        <div className="mb-1 text-xs font-semibold">{s.hops.length ? `HTTP header sent ${s.hops[0].from} → ${s.hops[0].to}${s.hops.length > 1 ? ` (×${s.hops.length})` : ""}` : "No incoming header: the gateway starts a new trace"}</div>
        <div className="flex flex-wrap items-start gap-1 font-mono text-[11px]">
          <span className="py-1 text-muted">traceparent:</span>
          {parts.map((p, i) => (
            <motion.div key={i} className="flex flex-col" initial={p.hot ? { opacity: 0, y: -8 } : false} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.25 * k, duration: 0.4 * k, ease }}>
              <span className={cx("break-all rounded px-1.5 py-1", p.hot ? "bg-accent text-white dark:text-black" : "bg-surface-2")}>{p.text}</span>
              <span className="mt-0.5 font-sans text-[10px] text-muted">{p.label}</span>
            </motion.div>
          ))}
        </div>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[460px] text-xs">
          <thead className="text-left text-muted"><tr><th className="py-1">span</th><th>span id</th><th>parent id</th><th>start</th><th>duration</th></tr></thead>
          <tbody className="font-mono">
            {started.map((x) => {
              const isNew = s.focus.includes(x.id);
              const done = spanEnd(x) <= s.t1;
              return (
                <motion.tr key={x.id} className={cx("border-t border-line", isNew && "bg-accent-soft/60")} initial={isNew ? { opacity: 0, x: -12 } : false} animate={{ opacity: 1, x: 0 }} transition={{ delay: TRAVEL * k, duration: 0.4 * k }}>
                  <td className="py-1 font-sans">{x.name}</td>
                  <td>{x.id.slice(0, 8)}…</td>
                  <td className={cx(x.parentId && s.hops.some((h) => h.header.includes(x.parentId!)) && isNew && "font-semibold text-accent")}>{x.parentId ? `${x.parentId.slice(0, 8)}…` : "—"}</td>
                  <td>{x.startMs} ms</td>
                  <td>{done ? `${x.durationMs} ms` : <span className="text-muted">running…</span>}</td>
                </motion.tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function BottleneckCard({ run, k }: { run: TraceRun; k: number }) {
  const rows = run.critical.filter((c) => c.self > 0).map((c) => ({ ...c, span: run.spans.find((x) => x.id === c.id)! }));
  const sorted = [...rows].sort((a, b) => b.self - a.self);
  const colors = ["var(--err)", "var(--warn)", "var(--violet)", "var(--accent)", "var(--ok)", "var(--muted)"];
  const col = (id: string) => colors[Math.min(sorted.findIndex((r) => r.id === id), colors.length - 1)];
  const before = rows.map((_, i) => rows.slice(0, i).reduce((t, r) => t + r.self, 0));
  return (
    <div className="space-y-2">
      <div className="text-xs font-semibold">Critical-path self time, in request order (sums to {run.total} ms)</div>
      <div className="flex h-7 overflow-hidden rounded bg-surface-2">
        {rows.map((r, i) => {
          const d = before[i];
          return <motion.div key={r.id} className="h-full border-r border-surface" style={{ background: col(r.id) }} title={`${r.span.name}: ${r.self} ms`}
            initial={{ width: 0 }} animate={{ width: `${(r.self / run.total) * 100}%` }} transition={{ delay: (0.2 + (d / run.total) * 2) * k, duration: ((r.self / run.total) * 2 + 0.05) * k, ease: "linear" }} />;
        })}
      </div>
      <ul className="grid gap-x-4 gap-y-1 text-xs sm:grid-cols-2">
        {sorted.map((r, i) => (
          <motion.li key={r.id} className="flex items-center justify-between gap-2" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: (2.3 + i * 0.12) * k }}>
            <span className="flex min-w-0 items-center gap-1.5"><span className="inline-block h-2.5 w-2.5 shrink-0 rounded-sm" style={{ background: col(r.id) }} /><span className="truncate font-mono">{r.span.name}</span>{i === 0 && <Badge tone="err">bottleneck</Badge>}</span>
            <span className="font-mono">{r.self} ms · {Math.round((r.self / run.total) * 100)}%</span>
          </motion.li>
        ))}
      </ul>
      <p className="text-[11px] text-muted">Spans off the critical path (e.g. faster shards in a parallel fan-out) can get slower without changing end-to-end latency, until they become the slowest.</p>
    </div>
  );
}

function SamplingCard({ run, input, animKey, k }: { run: TraceRun; input: TraceInput; animKey: string; k: number }) {
  const decided = useReveal(1, animKey, 0, 1.4 * k) === 1;
  const tile = (p: TraceRun["population"][number]) => (
    <motion.div key={p.i} layoutId={`${animKey}-t${p.i}`} transition={{ type: "spring", stiffness: 220, damping: 26 }}
      className={cx("flex w-[62px] flex-col items-center rounded border px-1 py-0.5 text-[10px]", p.error ? "border-err/50 bg-err-soft" : p.slow ? "border-warn/50 bg-warn-soft" : "border-line bg-surface")}>
      <span className="flex items-center gap-0.5 font-mono">{p.error ? <XCircle size={10} className="text-err" aria-label="error" /> : p.slow ? <Timer size={10} className="text-warn" aria-label="slow" /> : <CheckCircle2 size={10} className="text-ok" aria-label="ok" />}#{p.i}</span>
      <span className="font-mono">{p.totalMs} ms</span>
      <span className="font-mono text-muted">u={p.u.toFixed(2)}</span>
    </motion.div>
  );
  const kept = run.population.filter((p) => p.kept), dropped = run.population.filter((p) => !p.kept);
  return (
    <div className="space-y-2">
      <p className="text-xs">Each trace id hashes to a number <span className="font-mono">u</span> in [0, 1). Head sampling keeps it if <span className="font-mono">u &lt; {input.samplingRate.toFixed(2)}</span>{input.tailSampling ? `; tail sampling also keeps errors and traces over ${SLOW_MS} ms` : ""}.</p>
      {!decided ? (
        <div className="flex flex-wrap gap-1">{run.population.map(tile)}</div>
      ) : (
        <div className="grid gap-2 sm:grid-cols-2">
          <div className="rounded border border-ok/40 p-1.5"><div className="mb-1 text-xs font-semibold text-ok">Kept ({kept.length}) → exported to the tracing backend</div><div className="flex flex-wrap gap-1">{kept.map(tile)}</div></div>
          <div className="rounded border border-line p-1.5"><div className="mb-1 text-xs font-semibold text-muted">Dropped ({dropped.length})</div><div className="flex flex-wrap gap-1">{dropped.map(tile)}</div></div>
        </div>
      )}
    </div>
  );
}

function LatencyMath({ run, parallel }: { run: TraceRun; parallel: boolean }) {
  const shards = run.spans.filter((x) => x.component === "vector-db").map((x) => x.durationMs);
  const embed = run.spans.find((x) => x.name === "embed_query")!.durationMs;
  const retr = run.spans.find((x) => x.name === "retrieve")!.durationMs;
  const core = parallel ? Math.max(...shards) : shards.reduce((a, b) => a + b, 0);
  const shardTex = parallel ? `\\max(${shards.join(", ")})` : shards.join(" + ");
  return (
    <div className="space-y-2">
      <TexBlock>{`T_{\\text{retrieve}} = t_{\\text{embed}} + ${parallel ? "\\max_i t_{s_i}" : "\\sum_i t_{s_i}"} + \\text{gaps} = ${embed} + ${shardTex} + ${retr - embed - core} = ${retr}\\,\\text{ms}`}</TexBlock>
      <TexBlock>{`T_{\\text{request}} = \\sum_{\\text{critical path}} \\text{self}_i = ${run.critical.filter((c) => c.self > 0).map((c) => c.self).join(" + ")} = ${run.total}\\,\\text{ms}`}</TexBlock>
    </div>
  );
}
