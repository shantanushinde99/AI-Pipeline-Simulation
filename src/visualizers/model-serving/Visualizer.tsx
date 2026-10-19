import { useMemo, useState } from "react";
import { AnimatePresence, motion, type MotionValue } from "motion/react";
import { RotateCcw, Shuffle } from "lucide-react";
import { usePlayback } from "../../hooks/usePlayback";
import { PlaybackControls } from "../../components/visualizer/PlaybackControls";
import { EventLog } from "../../components/visualizer/EventLog";
import { LineChart } from "../../components/visualizer/LineChart";
import { StepExplainer, Workspace } from "../../components/visualizer/Workspace";
import { Badge, Button, Callout, KV, Panel, SimulatedLabel, Slider, StatusBadge, Toggle } from "../../components/ui";
import { TexBlock } from "../../components/Tex";
import type { VisualizerTab } from "../types";
import { StepCaption, Stats, f0, f1, pct, useSimClock, useTime } from "../latency-and-throughput/kit";
import {
  BATCH_BASE_MS, COLD_START_MS, DEFAULT_SERVING, PER_ITEM_MS, SCALE_AT_MS, WINDOW_MS, batchMs, metricsAt, simulate, sweepBatch,
  type Batch, type Replica, type SReq, type ServingConfig, type ServingState,
} from "./simulation";

/** 1 simulated ms plays over SLOW real ms at 1×. */
const SLOW = 20;
const STEP_MS = WINDOW_MS * SLOW + 700;

const PRESETS: { label: string; patch: Partial<ServingConfig> }[] = [
  { label: "Balanced batching", patch: { arrivalRate: 300, replicas: 2, maxBatch: 8, maxWaitMs: 10, scaleOut: false, routeBeforeReady: false } },
  { label: "Break it: no batching (max batch 1)", patch: { arrivalRate: 300, replicas: 2, maxBatch: 1, maxWaitMs: 0, scaleOut: false, routeBeforeReady: false } },
  { label: "Long max wait (50 ms), low traffic", patch: { arrivalRate: 40, replicas: 2, maxBatch: 16, maxWaitMs: 50, scaleOut: false, routeBeforeReady: false } },
  { label: "Overload + autoscale (cold start)", patch: { arrivalRate: 260, replicas: 1, maxBatch: 8, maxWaitMs: 10, scaleOut: true, routeBeforeReady: false } },
  { label: "Break it: route before ready", patch: { arrivalRate: 260, replicas: 1, maxBatch: 8, maxWaitMs: 10, scaleOut: true, routeBeforeReady: true } },
];

export default function ModelServingVisualizer({ tab }: { tab: VisualizerTab }) {
  const [cfg, setCfg] = useState<ServingConfig>(DEFAULT_SERVING);
  const [replay, setReplay] = useState(0);
  const run = useMemo(() => simulate(cfg), [cfg]);
  const pb = usePlayback(run.steps.length, () => STEP_MS);
  const step = run.steps[pb.index];
  const s = step.state;
  const k = 1 / pb.speed;
  const animKey = `${step.id}-${replay}`;
  const clock = useSimClock(s.t0, s.t1, (WINDOW_MS * SLOW * k) / 1000, animKey);
  const set = (p: Partial<ServingConfig>) => setCfg((c) => ({ ...c, ...p }));

  if (tab === "metrics") return <Metrics cfg={cfg} />;

  const story = narrate(s, run.cfg, run.replicas);
  const capPerReplica = (run.cfg.maxBatch / batchMs(run.cfg.maxBatch)) * 1000;
  const f = run.final;

  return (
    <Workspace
      main={
        <>
          <Panel title="Serving path" actions={<SimulatedLabel>Simulated GPUs and latencies</SimulatedLabel>}>
            <StepCaption index={pb.index} total={run.steps.length} id={step.id} headline={story.headline} sub={story.sub} onReplay={() => setReplay((r) => r + 1)} />
            <div className="overflow-x-auto">
              <div className="min-w-[600px]">
                <ServingSvg reqs={run.reqs} batches={run.batches} replicas={run.replicas} cfg={run.cfg} clock={clock} t0={s.t0} />
              </div>
            </div>
            <div className="mt-2 flex flex-wrap gap-3 text-[11px] text-muted">
              <span className="flex items-center gap-1"><svg width="10" height="10"><circle cx="5" cy="5" r="4" fill="var(--ok)" /></svg>request age &lt; 50 ms</span>
              <span className="flex items-center gap-1"><svg width="10" height="10"><circle cx="5" cy="5" r="4" fill="var(--warn)" /></svg>50–150 ms</span>
              <span className="flex items-center gap-1"><svg width="10" height="10"><circle cx="5" cy="5" r="4" fill="var(--err)" /></svg>&gt; 150 ms</span>
              <span>{WINDOW_MS} simulated ms play over {(WINDOW_MS * SLOW) / 1000} s at 1×</span>
            </div>
          </Panel>
          <Panel title="What's happening in this step">
            <AnimatePresence mode="wait" initial={false}>
              <motion.div key={animKey} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: 0.2 }}>
                <LiveCard run={run} clock={clock} />
              </motion.div>
            </AnimatePresence>
          </Panel>
          <PlaybackControls pb={pb} stepTitles={run.steps.map((x) => x.title)} />
          <Panel title="Serving configuration" actions={<Button size="sm" variant="ghost" onClick={() => { setCfg(DEFAULT_SERVING); pb.restart(); }}><RotateCcw size={12} /> Reset</Button>}>
            <div className="grid gap-3 sm:grid-cols-2">
              <Slider label="Arrival rate" value={cfg.arrivalRate} min={20} max={400} step={10} format={(v) => `${v} req/s`} onChange={(v) => set({ arrivalRate: v })} />
              <Slider label="Replicas (GPUs)" value={cfg.replicas} min={1} max={4} onChange={(v) => set({ replicas: v })} />
              <Slider label="Max batch size" value={cfg.maxBatch} min={1} max={16} onChange={(v) => set({ maxBatch: v })} hint={`capacity ≈ ${f0(capPerReplica)} req/s per replica at full batches`} />
              <Slider label="Max wait (batch timeout)" value={cfg.maxWaitMs} min={0} max={50} format={(v) => `${v} ms`} onChange={(v) => set({ maxWaitMs: v })} />
              <Toggle label={`Autoscale: add a replica at ${SCALE_AT_MS} ms`} checked={cfg.scaleOut} onChange={(v) => set({ scaleOut: v, routeBeforeReady: v && cfg.routeBeforeReady })} hint={`It needs ${COLD_START_MS} ms to load weights (cold start).`} />
              <Toggle label="Break it: route traffic before the replica is ready" checked={cfg.routeBeforeReady} onChange={(v) => set({ routeBeforeReady: v, scaleOut: v || cfg.scaleOut })} hint="No readiness check: requests queue on a replica that is still loading." />
            </div>
            <div className="mt-3 flex flex-wrap items-center gap-1.5">
              <span className="text-xs text-muted">Presets:</span>
              {PRESETS.map((p) => <Button key={p.label} size="sm" onClick={() => { set(p.patch); pb.restart(); }}>{p.label}</Button>)}
              <Button size="sm" variant="ghost" onClick={() => set({ seed: cfg.seed + 1 })}><Shuffle size={12} /> Seed {cfg.seed}</Button>
            </div>
          </Panel>
        </>
      }
      inspector={
        <>
          <StepExplainer index={pb.index} title={step.title} what={step.description}
            why="Dynamic batching is the main lever for GPU serving cost: the fixed per-launch cost is shared by every request in the batch." />
          <Panel title="Calculation">
            <div className="space-y-2">
              <TexBlock>{`t_{\\text{batch}}(b) = ${BATCH_BASE_MS} + ${PER_ITEM_MS}\\,b \\;\\Rightarrow\\; t(${run.cfg.maxBatch}) = ${batchMs(run.cfg.maxBatch).toFixed(1)}\\,\\text{ms}`}</TexBlock>
              <TexBlock>{`\\begin{aligned} \\text{cap}_{\\text{replica}} &= \\frac{b_{\\max}}{t(b_{\\max})} = \\frac{${run.cfg.maxBatch}}{${(batchMs(run.cfg.maxBatch) / 1000).toFixed(4)}\\,\\text{s}} \\\\ &= ${capPerReplica.toFixed(0)}\\ \\text{req/s} \\end{aligned}`}</TexBlock>
              <TexBlock>{`\\text{offered per replica} = \\frac{${run.cfg.arrivalRate}}{${run.cfg.replicas}} = ${(run.cfg.arrivalRate / run.cfg.replicas).toFixed(0)}\\ \\text{req/s}`}</TexBlock>
            </div>
          </Panel>
          <Panel title="Whole run, 0–1.2 s (simulated)">
            <KV rows={[
              ["requests", run.reqs.length],
              ["throughput", `${f0(f.throughput)} req/s`],
              ["p50 / p95", `${f0(f.p50)} / ${f0(f.p95)} ms`],
              ["GPU busy", pct(f.gpuUtil)],
              ["batches / avg size", `${f.batches} / ${f1(f.meanBatch)}`],
            ]} />
          </Panel>
          <Panel title="Events"><EventLog events={step.events} /></Panel>
        </>
      }
    />
  );
}

function narrate(s: ServingState, cfg: ServingConfig, replicas: Replica[]) {
  const scaled = replicas.find((p) => p.createdAt > 0);
  if (s.coldEvent === "created")
    return { headline: `Autoscaler adds replica ${scaled!.id + 1}: cold start begins`, sub: cfg.routeBeforeReady ? "The load balancer already sends it traffic, but it cannot run anything until its weights are loaded." : `It must load model weights for ${COLD_START_MS} ms; the load balancer keeps it out of rotation until then.` };
  if (s.coldEvent === "ready") return { headline: `Replica ${scaled!.id + 1} is warm and joins the rotation`, sub: cfg.routeBeforeReady ? "The requests stuck in its queue finally run: they waited through the whole cold start." : "From now on the load balancer spreads traffic over one more GPU." };
  if (s.t0 === 0) return { headline: "Requests arrive and the load balancer spreads them over replicas", sub: `Each replica queues requests and launches a batch when it has ${cfg.maxBatch} or the oldest has waited ${cfg.maxWaitMs} ms.` };
  if (s.queuedAtEnd > 3 * cfg.maxBatch * replicas.length) return { headline: `GPUs can't keep up: ${s.queuedAtEnd} requests are waiting`, sub: cfg.maxBatch === 1 ? "With batch size 1 every request pays the full launch cost. Raise max batch size." : "Arrivals exceed capacity. Add replicas or allow bigger batches." };
  const top = (Object.entries(s.closed) as [string, number][]).sort((a, b) => b[1] - a[1])[0];
  if (!top[1]) return { headline: "GPUs finish batches; responses go back", sub: "No new batch launched in this window." };
  if (top[0] === "full") return { headline: `Batches fill up to ${cfg.maxBatch} before the timer fires`, sub: "High traffic: the batch is full quickly, so waiting is short and GPU work is shared by many requests." };
  if (top[0] === "max wait") return { headline: `Batches close on the ${cfg.maxWaitMs} ms timer, not full`, sub: "Traffic is light, so the batcher gives up waiting and runs a partial batch to protect latency." };
  return { headline: "Requests pile up while the GPU is busy", sub: "When the GPU frees up, everything waiting (up to max batch) goes in the next batch." };
}

const LB: [number, number] = [118, 0];
const QHEAD = 372, SLOT = 11, SLOTS = 15, GX = 392, GW = 128, RESP = 742;

function ServingSvg({ reqs, batches, replicas, cfg, clock, t0 }: { reqs: SReq[]; batches: Batch[]; replicas: Replica[]; cfg: ServingConfig; clock: MotionValue<number>; t0: number }) {
  const t = useTime(clock);
  const n = replicas.length;
  const laneH = 66;
  const H = 36 + n * laneH;
  const ly = (r: number) => 40 + r * laneH + laneH / 2 - 6;
  const mid = (ly(0) + ly(n - 1)) / 2;
  const age = (q: SReq) => Math.min(t, q.end) - q.arrive;
  const color = (q: SReq) => (age(q) < 50 ? "var(--ok)" : age(q) < 150 ? "var(--warn)" : "var(--err)");
  const live = reqs.filter((q) => q.arrive <= t && q.end > t);
  const lastRouted = [...reqs].reverse().find((q) => q.arrive <= t);
  const pos = (q: SReq): [number, number] | null => {
    if (q.start <= t) {
      const b = batches[batches.findIndex((x) => x.id === q.batch)];
      const j = b.reqIds.indexOf(q.id);
      return [GX + 12 + (j % 8) * 13, ly(q.replica) - 6 + Math.floor(j / 8) * 13];
    }
    const queue = reqs.filter((x) => x.replica === q.replica && x.arrive <= t && x.start > t);
    const i = queue.indexOf(q);
    return i < SLOTS ? [QHEAD - i * SLOT, ly(q.replica)] : null;
  };
  return (
    <svg viewBox={`0 0 760 ${H}`} className="h-auto w-full" role="img" aria-label={`Model serving at ${t.toFixed(0)} ms: ${live.length} requests in flight across ${n} replicas`}>
      <text x={752} y={16} textAnchor="end" fontSize={12} fontFamily="var(--font-mono)" fill="var(--muted)">t = {t.toFixed(0)} ms</text>
      <text x={756} y={30} textAnchor="end" fontSize={10} fill="var(--muted)">responses →</text>
      <circle cx={30} cy={mid} r={22} fill="var(--surface-2)" stroke="var(--line)" />
      <text x={30} y={mid + 4} textAnchor="middle" fontSize={10}>clients</text>
      <text x={30} y={mid - 28} textAnchor="middle" fontSize={10} fontFamily="var(--font-mono)" fill="var(--accent)">{cfg.arrivalRate}/s</text>
      {replicas.map((p) => (
        <line key={p.id} x1={LB[0] + 34} y1={mid} x2={QHEAD - SLOTS * SLOT} y2={ly(p.id)} stroke="var(--line)" strokeDasharray={p.createdAt > t ? "3 4" : undefined} />
      ))}
      <rect x={LB[0] - 34} y={mid - 22} width={68} height={44} rx={8} fill="var(--surface)" stroke="var(--accent)" />
      <text x={LB[0]} y={mid - 4} textAnchor="middle" fontSize={10} fontWeight={600}>load</text>
      <text x={LB[0]} y={mid + 9} textAnchor="middle" fontSize={10} fontWeight={600}>balancer</text>
      {lastRouted && t - lastRouted.arrive < 12 && (
        <text x={LB[0]} y={mid + 36} textAnchor="middle" fontSize={10} fontFamily="var(--font-mono)" fill="var(--accent)">→ replica {lastRouted.replica + 1}</text>
      )}
      {replicas.map((p) => {
        const y = ly(p.id);
        const queued = reqs.filter((q) => q.replica === p.id && q.arrive <= t && q.start > t).length;
        const running = batches.find((b) => b.replica === p.id && b.start <= t && b.end > t);
        const lastDone = [...batches].reverse().find((b) => b.replica === p.id && b.end <= t);
        const absent = p.createdAt > t;
        const cold = !absent && p.readyAt > t;
        return (
          <g key={p.id}>
            <text x={QHEAD - SLOTS * SLOT - 6} y={y - 20} fontSize={10} fill="var(--muted)">replica {p.id + 1} queue · <tspan fontFamily="var(--font-mono)" fill={queued > 2 * cfg.maxBatch ? "var(--err)" : "var(--fg)"}>{queued}</tspan></text>
            <rect x={QHEAD - SLOTS * SLOT - 8} y={y - 11} width={SLOTS * SLOT + 18} height={22} rx={11} fill="var(--surface-2)" stroke="var(--line)" strokeDasharray={absent ? "3 4" : undefined} />
            {queued > SLOTS && <text x={QHEAD - SLOTS * SLOT - 8} y={y + 24} fontSize={9} fill="var(--err)">+{queued - SLOTS} more</text>}
            <rect x={GX} y={y - 16} width={GW} height={32} rx={6} fill={absent ? "transparent" : cold ? "var(--warn-soft)" : running ? "var(--accent-soft)" : "var(--surface)"}
              stroke={absent ? "var(--line)" : cold ? "var(--warn)" : running ? "var(--accent)" : "var(--line)"} strokeWidth={running ? 2 : 1} strokeDasharray={absent ? "3 4" : undefined} />
            <text x={GX} y={y - 20} fontSize={10} fontWeight={600} fill="var(--fg)">GPU {p.id + 1}</text>
            {absent && <text x={GX + GW / 2} y={y + 4} textAnchor="middle" fontSize={9} fill="var(--muted)">added at {p.createdAt} ms</text>}
            {cold && (
              <>
                <text x={GX + GW / 2} y={y} textAnchor="middle" fontSize={9} fontWeight={600} fill="var(--warn)">⚠ cold start: loading weights</text>
                <rect x={GX + 10} y={y + 6} width={GW - 20} height={4} rx={2} fill="var(--line)" />
                <rect x={GX + 10} y={y + 6} width={(GW - 20) * ((t - p.createdAt) / (p.readyAt - p.createdAt))} height={4} rx={2} fill="var(--warn)" />
              </>
            )}
            {running && (
              <>
                <rect x={GX} y={y + 19} width={GW} height={3} rx={1.5} fill="var(--line)" />
                <rect x={GX} y={y + 19} width={GW * ((t - running.start) / (running.end - running.start))} height={3} rx={1.5} fill="var(--accent)" />
              </>
            )}
            <text x={GX + GW + 8} y={y - 3} fontSize={10} fill="var(--fg)" fontFamily="var(--font-mono)">
              {running ? `batch #${running.id} · ${running.size} req` : cold ? `ready at ${p.readyAt} ms` : absent ? "" : "idle"}
            </text>
            <text x={GX + GW + 8} y={y + 11} fontSize={9} fill={"var(--muted)"}>
              {running ? `closed: ${running.reason} · ${(running.end - running.start).toFixed(1)} ms` : lastDone ? `last: #${lastDone.id}, ${lastDone.size} req` : ""}
            </text>
          </g>
        );
      })}
      <AnimatePresence>
        {live.map((q) => {
          const p = pos(q);
          if (!p) return null;
          const fresh = q.arrive >= t0;
          return (
            <motion.g key={q.id} initial={fresh ? { x: LB[0], y: mid, opacity: 0 } : { x: p[0], y: p[1], opacity: 1 }} animate={{ x: p[0], y: p[1], opacity: 1 }}
              exit={{ x: RESP, y: ly(q.replica), opacity: 0, transition: { duration: 0.7, ease: "easeIn" } }} transition={{ type: "spring", stiffness: 150, damping: 21 }}>
              <circle r={5} fill={color(q)} stroke="var(--surface)" strokeWidth={1.2} />
            </motion.g>
          );
        })}
      </AnimatePresence>
    </svg>
  );
}

function LiveCard({ run, clock }: { run: ReturnType<typeof simulate>; clock: MotionValue<number> }) {
  const t = useTime(clock);
  const m = metricsAt(run, t);
  return (
    <div className="space-y-3">
      <Stats items={[
        { label: "throughput so far", value: `${f0(m.throughput)} req/s`, tone: "accent" },
        { label: "p50 latency", value: `${f0(m.p50)} ms`, tone: "ok" },
        { label: "p95 latency", value: `${f0(m.p95)} ms`, tone: m.p95 > 150 ? "err" : m.p95 > 60 ? "warn" : "ok" },
        { label: "GPU busy", value: pct(m.gpuUtil), hint: `avg batch ${f1(m.meanBatch)} / ${run.cfg.maxBatch}` },
      ]} />
      <BatchLog batches={run.batches} t={t} maxBatch={run.cfg.maxBatch} />
      <p className="text-xs text-muted">
        One forward pass costs {BATCH_BASE_MS} ms + {PER_ITEM_MS} ms per request, so a batch of 1 costs {batchMs(1)} ms per request but a batch of {run.cfg.maxBatch} costs{" "}
        {(batchMs(run.cfg.maxBatch) / run.cfg.maxBatch).toFixed(1)} ms per request. Bigger batches = more throughput, but each request may wait up to {run.cfg.maxWaitMs} ms for the batch to fill.
      </p>
    </div>
  );
}

function BatchLog({ batches, t, maxBatch }: { batches: Batch[]; t: number; maxBatch: number }) {
  const rows = batches.filter((b) => b.start <= t).slice(-6).reverse();
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[420px] text-xs">
        <thead className="text-left text-muted"><tr><th className="py-1">batch</th><th>GPU</th><th className="w-[38%]">size (of {maxBatch})</th><th>closed because</th><th className="text-right">status</th></tr></thead>
        <tbody>
          <AnimatePresence initial={false}>
            {rows.map((b) => (
              <motion.tr key={b.id} layout initial={{ opacity: 0, backgroundColor: "var(--accent-soft)" }} animate={{ opacity: 1, backgroundColor: "rgba(0,0,0,0)" }} transition={{ duration: 0.6 }} className="border-t border-line">
                <td className="py-1 font-mono">#{b.id}</td>
                <td className="font-mono">{b.replica + 1}</td>
                <td>
                  <div className="flex items-center gap-1.5">
                    <div className="h-2.5 flex-1 rounded-sm bg-surface-2"><motion.div className="h-full rounded-sm bg-accent" initial={{ width: 0 }} animate={{ width: `${(b.size / maxBatch) * 100}%` }} transition={{ duration: 0.4 }} /></div>
                    <span className="w-5 text-right font-mono">{b.size}</span>
                  </div>
                </td>
                <td className="pl-2"><Badge tone={b.reason === "full" ? "ok" : "neutral"}>{b.reason}</Badge></td>
                <td className="text-right">{b.end <= t ? <StatusBadge status="ok" label={`done ${(b.end - b.start).toFixed(1)} ms`} /> : <StatusBadge status="active" label="running" />}</td>
              </motion.tr>
            ))}
          </AnimatePresence>
        </tbody>
      </table>
      {!rows.length && <p className="py-2 text-xs text-muted">No batch launched yet: the batcher is waiting for the batch to fill or the timer to fire.</p>}
    </div>
  );
}

function Metrics({ cfg }: { cfg: ServingConfig }) {
  const pts = useMemo(() => sweepBatch(cfg), [cfg]);
  return (
    <div className="space-y-3">
      <Callout title="Same traffic, different max batch size (simulated)">
        Each point re-runs the 1.2 s simulation with your current arrival rate ({cfg.arrivalRate} req/s), replicas ({cfg.replicas}) and max wait ({cfg.maxWaitMs} ms). Throughput is completions within 1.2 s; latency covers every request.
      </Callout>
      <div className="grid gap-3 lg:grid-cols-2">
        <Panel title="Throughput vs max batch size">
          <LineChart xLabel="max batch size" yLabel="req/s" series={[{ name: "throughput", color: "var(--accent)", points: pts.map((p) => [p.maxBatch, p.throughput]) }]} />
        </Panel>
        <Panel title="Latency vs max batch size">
          <LineChart xLabel="max batch size" yLabel="ms" logY series={[
            { name: "p50", color: "var(--ok)", points: pts.map((p) => [p.maxBatch, p.p50]) },
            { name: "p95", color: "var(--err)", points: pts.map((p) => [p.maxBatch, p.p95]) },
          ]} />
        </Panel>
      </div>
      <Panel title="Sweep table">
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead className="text-left text-muted"><tr><th className="py-1">max batch</th><th>throughput</th><th>p50 ms</th><th>p95 ms</th><th>GPU busy</th><th>avg batch</th></tr></thead>
            <tbody className="font-mono">{pts.map((p) => (
              <tr key={p.maxBatch} className="border-t border-line"><td className="py-1">{p.maxBatch}</td><td>{f0(p.throughput)}</td><td>{f0(p.p50)}</td><td>{f0(p.p95)}</td><td>{pct(p.gpuUtil)}</td><td>{f1(p.meanBatch)}</td></tr>
            ))}</tbody>
          </table>
        </div>
      </Panel>
    </div>
  );
}
