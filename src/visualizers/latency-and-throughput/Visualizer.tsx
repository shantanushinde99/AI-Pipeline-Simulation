import { useMemo, useState } from "react";
import { AnimatePresence, motion, type MotionValue } from "motion/react";
import { RotateCcw, Shuffle } from "lucide-react";
import { usePlayback } from "../../hooks/usePlayback";
import { PlaybackControls } from "../../components/visualizer/PlaybackControls";
import { EventLog } from "../../components/visualizer/EventLog";
import { LineChart } from "../../components/visualizer/LineChart";
import { AnimatedNumber } from "../../components/visualizer/AnimatedNumber";
import { StepExplainer, Workspace } from "../../components/visualizer/Workspace";
import { Button, Callout, KV, Panel, Select, SimulatedLabel, Slider, Toggle } from "../../components/ui";
import { TexBlock } from "../../components/Tex";
import type { VisualizerTab } from "../types";
import { BURST, DEFAULT_LT, HORIZON_MS, rateAt, simulate, snapshot, sweep, type LTConfig, type LTState, type Req, type ServiceDist } from "./simulation";
import { Histogram, StepCaption, Stats, TimeBars, f0, f1, pct, useSimClock, useTime } from "./kit";
import { histogram, percentile } from "./queueing";

/** 1 simulated second plays over SLOW real seconds at 1× speed. */
const SLOW = 4;
const stepMs = (s: LTState) => (s.kind === "window" ? ((s.t1 - s.t0) / 1000) * SLOW * 1000 + 700 : 3500);

const PRESETS: { label: string; patch: Partial<LTConfig> }[] = [
  { label: "Comfortable (ρ ≈ 0.5)", patch: { arrivalRate: 6, serviceMs: 250, workers: 3, burst: false, dist: "exponential" } },
  { label: "Near capacity (ρ ≈ 0.95)", patch: { arrivalRate: 11.5, serviceMs: 250, workers: 3, burst: false, dist: "exponential" } },
  { label: "Overloaded (ρ ≈ 1.25)", patch: { arrivalRate: 15, serviceMs: 250, workers: 3, burst: false, dist: "exponential" } },
  { label: "Burst", patch: { arrivalRate: 6, serviceMs: 250, workers: 3, burst: true, dist: "exponential" } },
];

export default function LatencyThroughputVisualizer({ tab }: { tab: VisualizerTab }) {
  const [cfg, setCfg] = useState<LTConfig>(DEFAULT_LT);
  const [replay, setReplay] = useState(0);
  const run = useMemo(() => simulate(cfg), [cfg]);
  const durations = useMemo(() => run.steps.map((s) => stepMs(s.state)), [run]);
  const pb = usePlayback(run.steps.length, (i) => durations[i] ?? 3000);
  const step = run.steps[pb.index];
  const s = step.state;
  const k = 1 / pb.speed;
  const animKey = `${step.id}-${replay}`;
  const clock = useSimClock(s.kind === "window" ? s.t0 : s.t1, s.t1, s.kind === "window" ? ((s.t1 - s.t0) / 1000) * SLOW * k : 0, animKey);
  const set = (p: Partial<LTConfig>) => setCfg((c) => ({ ...c, ...p }));
  const sum = run.sum;

  if (tab === "metrics") return <Metrics cfg={cfg} />;

  const story = narrate(s, run.cfg, s.queueAtEnd, sum.rho);

  return (
    <Workspace
      main={
        <>
          <Panel title="Queue" actions={<SimulatedLabel>Simulated queue, not a real server</SimulatedLabel>}>
            <StepCaption index={pb.index} total={run.steps.length} id={step.id} headline={story.headline} sub={story.sub} onReplay={() => setReplay((r) => r + 1)} />
            <div className="overflow-x-auto">
              <div className="min-w-[560px]">
                <QueueSvg reqs={run.reqs} clock={clock} t0={s.t0} cfg={run.cfg} />
              </div>
            </div>
            <div className="mt-2 flex flex-wrap gap-3 text-[11px] text-muted">
              <span className="flex items-center gap-1"><svg width="10" height="10"><circle cx="5" cy="5" r="4" fill="var(--ok)" /></svg>waited &lt; 1× service time</span>
              <span className="flex items-center gap-1"><svg width="10" height="10"><circle cx="5" cy="5" r="4" fill="var(--warn)" /></svg>1–4×</span>
              <span className="flex items-center gap-1"><svg width="10" height="10"><circle cx="5" cy="5" r="4" fill="var(--err)" /></svg>&gt; 4× (tail)</span>
              <span>1 simulated second plays over {SLOW} s at 1× speed</span>
            </div>
          </Panel>
          <Panel title="What's happening in this step">
            <AnimatePresence mode="wait" initial={false}>
              <motion.div key={animKey} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: 0.2 }}>
                {s.kind === "window" ? <LiveCard reqs={run.reqs} clock={clock} cfg={run.cfg} /> : <LittleCard sum={sum} reqs={run.reqs} k={k} />}
              </motion.div>
            </AnimatePresence>
          </Panel>
          <PlaybackControls pb={pb} stepTitles={run.steps.map((x) => x.title)} />
          <Panel title="Workload & capacity" actions={<Button size="sm" variant="ghost" onClick={() => { setCfg(DEFAULT_LT); pb.restart(); }}><RotateCcw size={12} /> Reset</Button>}>
            <div className="grid gap-3 sm:grid-cols-2">
              <Slider label="Arrival rate λ" value={cfg.arrivalRate} min={1} max={40} step={0.5} format={(v) => `${v} req/s`} onChange={(v) => set({ arrivalRate: v })} />
              <Slider label="Mean service time E[S]" value={cfg.serviceMs} min={50} max={800} step={10} format={(v) => `${v} ms`} onChange={(v) => set({ serviceMs: v })} />
              <Slider label="Workers c" value={cfg.workers} min={1} max={8} onChange={(v) => set({ workers: v })} hint={`capacity = c / E[S] = ${f1((cfg.workers * 1000) / cfg.serviceMs)} req/s`} />
              <Select<ServiceDist> label="Service time distribution" value={cfg.dist} onChange={(v) => set({ dist: v })} options={[
                { value: "constant", label: "Constant (every request same)" },
                { value: "exponential", label: "Exponential (random)" },
                { value: "heavy-tail", label: "Heavy tail (10% are 10× slower)" },
              ]} />
              <Toggle label={`Burst: ×4 arrivals from ${BURST[0] / 1000} s to ${BURST[1] / 1000} s`} checked={cfg.burst} onChange={(v) => set({ burst: v })} hint="Same average capacity, but traffic arrives in a spike." />
              <div className="flex items-end"><Button size="sm" onClick={() => set({ seed: cfg.seed + 1 })}><Shuffle size={12} /> New random seed ({cfg.seed})</Button></div>
            </div>
            <div className="mt-3 flex flex-wrap items-center gap-1.5">
              <span className="text-xs text-muted">Presets:</span>
              {PRESETS.map((p) => <Button key={p.label} size="sm" onClick={() => { set(p.patch); pb.restart(); }}>{p.label}</Button>)}
            </div>
          </Panel>
        </>
      }
      inspector={
        <>
          <StepExplainer index={pb.index} title={step.title} what={step.description}
            why={sum.rho >= 1 ? "Offered load exceeds capacity: the queue grows without bound and latency keeps rising as long as the overload lasts." : "Queues absorb randomness. The closer utilization gets to 1, the longer the queue that randomness builds."} />
          <Panel title="Calculation">
            <div className="space-y-2">
              <TexBlock>{`\\rho = \\frac{\\lambda \\cdot E[S]}{c} = \\frac{${sum.offeredRate.toFixed(2)} \\times ${(cfg.serviceMs / 1000).toFixed(3)}}{${run.cfg.workers}} = ${sum.rho.toFixed(2)}`}</TexBlock>
              <TexBlock>{`\\text{capacity} = \\frac{c}{E[S]} = \\frac{${run.cfg.workers}}{${(cfg.serviceMs / 1000).toFixed(3)}\\,\\text{s}} = ${sum.capacity.toFixed(1)}\\ \\text{req/s}`}</TexBlock>
              <TexBlock>{`L = \\lambda W:\\ ${sum.L.toFixed(2)} = ${sum.lambdaEff.toFixed(2)} \\times ${(sum.meanW / 1000).toFixed(3)}\\,\\text{s}`}</TexBlock>
            </div>
            <p className="mt-2 text-[11px] text-muted">λ in the first line is the arrival rate actually generated (requests ÷ 8 s), which includes any burst. Little's law uses the whole run until the queue drains.</p>
          </Panel>
          <Panel title="Whole run (simulated)">
            <KV rows={[
              ["requests", sum.n],
              ["offered load λ", `${sum.offeredRate.toFixed(2)} req/s`],
              ["capacity", `${sum.capacity.toFixed(2)} req/s`],
              ["utilization ρ", sum.rho.toFixed(2)],
              ["throughput (0–8 s)", `${sum.throughput.toFixed(2)} req/s`],
              ["p50 / p95 / p99", `${f0(sum.p50)} / ${f0(sum.p95)} / ${f0(sum.p99)} ms`],
              ["max queue length", sum.maxQueue],
            ]} />
          </Panel>
          <Panel title="Events"><EventLog events={step.events} /></Panel>
        </>
      }
    />
  );
}

function narrate(s: LTState, cfg: LTConfig, queueNow: number, rho: number) {
  if (s.kind === "summary")
    return { headline: `Little's law: L = λW holds with these exact numbers`, sub: rho >= 1 ? "Overloaded: the backlog took a long time to drain, so W (and L) are huge." : "Average requests in the system = arrival rate × average time each spends there." };
  const lam = rateAt(cfg, s.t0);
  if (s.t0 === 0) return { headline: `Requests arrive at random, about ${lam} per second`, sub: `${cfg.workers} worker${cfg.workers > 1 ? "s" : ""} take them one at a time; anyone who finds every worker busy waits in the queue.` };
  if (s.inBurst) return { headline: `Burst: ${lam} requests/s arrive, far above capacity`, sub: `The queue (${queueNow} waiting now) grows because ${lam}/s > ${((cfg.workers * 1000) / cfg.serviceMs).toFixed(1)}/s capacity. Late arrivals will wait the longest.` };
  if (s.queueAtEnd > 12) return { headline: `The queue keeps growing: ${s.queueAtEnd} waiting at ${s.t1 / 1000} s`, sub: "Arrivals outpace the workers. Every new request waits behind the whole line, so latency climbs." };
  if (s.queueAtEnd === 0 && s.arrivals > 0) return { headline: "Workers keep up: the queue empties again", sub: `${s.completions} requests completed in this second. Waiting only happens when several arrive close together.` };
  return { headline: `${s.arrivals} arrivals, ${s.completions} completions this second`, sub: "Random arrival gaps make the queue grow and shrink even when average capacity is enough." };
}

const QY = 150, QX0 = 120, QHEAD = 468, SLOT = 15, SLOTS = 23, WX = 590, EXIT = 712;

function QueueSvg({ reqs, clock, t0, cfg }: { reqs: Req[]; clock: MotionValue<number>; t0: number; cfg: LTConfig }) {
  const t = useTime(clock);
  const snap = snapshot(reqs, t);
  const c = cfg.workers;
  const boxH = Math.min(46, 250 / c - 6);
  const wy = (w: number) => QY + (w - (c - 1) / 2) * (boxH + 6);
  const lam = rateAt(cfg, t);
  const color = (q: Req) => {
    const waited = Math.min(t, q.start) - q.arrive;
    return waited < cfg.serviceMs ? "var(--ok)" : waited < 4 * cfg.serviceMs ? "var(--warn)" : "var(--err)";
  };
  const pos = (q: Req, i: number): [number, number] => (q.start <= t ? [WX - 34, wy(q.worker)] : [QHEAD - i * SLOT, QY]);
  const visibleQueue = snap.queued.slice(0, SLOTS);
  const top = Math.min(QY - 70, wy(0) - boxH / 2 - 24), bottom = Math.max(QY + 60, wy(c - 1) + boxH / 2 + 10);
  return (
    <svg viewBox={`0 ${top} 730 ${bottom - top}`} className="h-auto w-full" role="img" aria-label={`Queue with ${snap.queued.length} waiting and ${snap.serving.length} of ${c} workers busy at ${(t / 1000).toFixed(2)} s`}>
      <text x={720} y={top + 16} textAnchor="end" fontSize={12} fontFamily="var(--font-mono)" fill="var(--muted)">t = {(t / 1000).toFixed(2)} s</text>
      {/* source */}
      <circle cx={48} cy={QY} r={26} fill="var(--surface-2)" stroke="var(--line)" />
      <text x={48} y={QY + 4} textAnchor="middle" fontSize={11} fill="var(--fg)">clients</text>
      <text x={48} y={QY - 36} textAnchor="middle" fontSize={11} fontFamily="var(--font-mono)" fontWeight={600} fill={lam > cfg.arrivalRate ? "var(--warn)" : "var(--accent)"}>λ = {lam}/s</text>
      {lam > cfg.arrivalRate && <text x={48} y={QY + 44} textAnchor="middle" fontSize={10} fontWeight={600} fill="var(--warn)">⚠ burst</text>}
      {/* queue lane */}
      <rect x={QX0 - 10} y={QY - 16} width={QHEAD - QX0 + 22} height={32} rx={16} fill="var(--surface-2)" stroke="var(--line)" />
      <text x={(QX0 + QHEAD) / 2} y={QY - 26} textAnchor="middle" fontSize={11} fill="var(--muted)">FIFO queue · <tspan fontFamily="var(--font-mono)" fontWeight={600} fill={snap.queued.length > 10 ? "var(--err)" : "var(--fg)"}>{snap.queued.length} waiting</tspan></text>
      {snap.queued.length > SLOTS && <text x={QX0 - 4} y={QY + 32} fontSize={10} fontWeight={600} fill="var(--err)">+{snap.queued.length - SLOTS} more not shown</text>}
      <text x={QHEAD} y={QY + 32} textAnchor="end" fontSize={10} fill="var(--muted)">head →</text>
      {/* workers */}
      {Array.from({ length: c }, (_, w) => {
        const job = snap.serving.find((q) => q.worker === w);
        const prog = job ? (t - job.start) / job.service : 0;
        return (
          <g key={w}>
            <rect x={WX - 56} y={wy(w) - boxH / 2} width={136} height={boxH} rx={8} fill={job ? "var(--accent-soft)" : "var(--surface)"} stroke={job ? "var(--accent)" : "var(--line)"} strokeWidth={job ? 2 : 1} />
            <text x={WX - 12} y={wy(w) - (boxH > 30 ? 4 : -4)} fontSize={10} fill="var(--fg)">worker {w + 1}</text>
            {boxH > 30 && <text x={WX - 12} y={wy(w) + 9} fontSize={9} fill="var(--muted)">{job ? `busy ${Math.round(t - job.start)}/${Math.round(job.service)} ms` : "idle"}</text>}
            <rect x={WX - 12} y={wy(w) + boxH / 2 - 7} width={84} height={3} rx={1.5} fill="var(--line)" />
            {job && <rect x={WX - 12} y={wy(w) + boxH / 2 - 7} width={84 * Math.min(1, prog)} height={3} rx={1.5} fill="var(--accent)" />}
          </g>
        );
      })}
      <line x1={QHEAD + 14} y1={QY} x2={WX - 58} y2={QY} stroke="var(--line)" strokeDasharray="3 3" />
      <text x={EXIT - 4} y={QY + 4} textAnchor="end" fontSize={10} fill="var(--muted)">done →</text>
      <AnimatePresence>
        {[...visibleQueue, ...snap.serving].map((q) => {
          const i = visibleQueue.indexOf(q);
          const [x, y] = pos(q, i);
          const start: [number, number] = q.arrive < t0 ? [x, y] : [48, QY];
          return (
            <motion.g key={q.id} initial={{ x: start[0], y: start[1], opacity: q.arrive < t0 ? 1 : 0 }} animate={{ x, y, opacity: 1 }}
              exit={{ x: EXIT - 50, opacity: 0, transition: { duration: 0.5 } }} transition={{ type: "spring", stiffness: 140, damping: 20 }}>
              <circle r={6} fill={color(q)} stroke="var(--surface)" strokeWidth={1.5} />
            </motion.g>
          );
        })}
      </AnimatePresence>
    </svg>
  );
}

function LiveCard({ reqs, clock, cfg }: { reqs: Req[]; clock: MotionValue<number>; cfg: LTConfig }) {
  const t = useTime(clock);
  const snap = snapshot(reqs, t);
  const W = snap.done.map((q) => q.end - q.arrive);
  const allW = reqs.map((q) => q.end - q.arrive);
  const hi = Math.max(cfg.serviceMs * 2, percentile(allW, 99) * 1.05);
  const qlen = useMemo(() => Array.from({ length: HORIZON_MS / 100 }, (_, i) => snapshot(reqs, i * 100).queued.length), [reqs]);
  const buckets = qlen.map((v, i) => ({ t0: i * 100, t1: i * 100 + 100, values: [i * 100 <= t ? v : 0] }));
  const p = [50, 95, 99].map((x) => percentile(W, x));
  return (
    <div className="space-y-3">
      <Stats items={[
        { label: "arrived", value: String(snap.arrived) },
        { label: "completed", value: String(snap.done.length), tone: "ok" },
        { label: "waiting in queue", value: String(snap.queued.length), tone: snap.queued.length > 10 ? "err" : snap.queued.length > 3 ? "warn" : "neutral" },
        { label: "workers busy", value: `${snap.serving.length}/${cfg.workers}`, tone: "accent" },
      ]} />
      <div className="grid gap-3 md:grid-cols-2">
        <Histogram counts={histogram(W.filter((w) => w <= hi), 0, hi, 24)} lo={0} hi={hi} label={`Latency of the ${W.length} finished requests (wait + service)`}
          markers={[{ v: p[0], label: `p50 ${f0(p[0])}`, color: "var(--ok)" }, { v: p[1], label: `p95 ${f0(p[1])}`, color: "var(--warn)" }, { v: p[2], label: `p99 ${f0(p[2])}`, color: "var(--err)" }]} />
        <TimeBars buckets={buckets} series={[{ name: "requests waiting", color: "var(--violet)" }]} yMax={Math.max(5, ...qlen)} t={t} label="Queue length over time"
          shade={cfg.burst ? [BURST] : undefined} />
      </div>
      <p className="text-xs text-muted">
        The histogram fills in as requests finish. A request's latency is its <strong>wait in the queue</strong> plus its <strong>service time</strong>, so whenever the purple bars rise, the right side of the histogram (p95, p99) stretches out.
      </p>
    </div>
  );
}

function LittleCard({ sum, reqs, k }: { sum: ReturnType<typeof simulate>["sum"]; reqs: Req[]; k: number }) {
  const W = reqs.map((q) => q.end - q.arrive);
  const hi = Math.max(1, percentile(W, 99) * 1.05);
  const boxes = [
    { label: "L (time-average in system)", v: sum.L, d: 2, note: "measured by integrating N(t)" },
    { label: "λ (requests ÷ run length)", v: sum.lambdaEff, d: 2, note: `${sum.n} ÷ ${(sum.tEnd / 1000).toFixed(2)} s` },
    { label: "W (mean time in system)", v: sum.meanW / 1000, d: 3, note: "seconds" },
  ];
  return (
    <div className="space-y-3">
      <div className="grid gap-2 sm:grid-cols-3">
        {boxes.map((b, i) => (
          <motion.div key={b.label} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: (0.2 + i * 0.5) * k, duration: 0.5 * k }} className="rounded-md border border-line bg-surface-2/60 p-2">
            <div className="text-[10px] uppercase tracking-wide text-muted">{b.label}</div>
            <div className="font-mono text-xl font-semibold"><AnimatedNumber value={b.v} from={0} digits={b.d} delay={(0.2 + i * 0.5) * k} duration={0.8 * k} /></div>
            <div className="text-[10px] text-muted">{b.note}</div>
          </motion.div>
        ))}
      </div>
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1.8 * k }}>
        <TexBlock>{`L = \\lambda W \\;\\Rightarrow\\; ${sum.L.toFixed(3)} = ${sum.lambdaEff.toFixed(3)} \\times ${(sum.meanW / 1000).toFixed(3)} = ${sum.lambdaW.toFixed(3)}`}</TexBlock>
      </motion.div>
      <Histogram counts={histogram(W.filter((w) => w <= hi), 0, hi, 30)} lo={0} hi={hi} label={`All ${sum.n} latencies`}
        markers={[{ v: sum.p50, label: `p50 ${f0(sum.p50)}`, color: "var(--ok)" }, { v: sum.p95, label: `p95 ${f0(sum.p95)}`, color: "var(--warn)" }, { v: sum.p99, label: `p99 ${f0(sum.p99)}`, color: "var(--err)" }]} />
      <p className="text-xs text-muted">Utilization was <strong>{pct(sum.rho)}</strong>. Try the "Near capacity" preset: the mean barely moves but p99 explodes. Little's law holds for any distribution and any queue discipline.</p>
    </div>
  );
}

function Metrics({ cfg }: { cfg: LTConfig }) {
  const pts = useMemo(() => sweep(cfg), [cfg]);
  const cap = (cfg.workers * 1000) / cfg.serviceMs;
  return (
    <div className="space-y-3">
      <Callout title="Sweep of the arrival rate (simulated)">
        Each point is a full 8-second run of the same queue ({cfg.workers} workers, E[S] = {cfg.serviceMs} ms, {cfg.dist}) at a different arrival rate. Burst is off for the sweep. Capacity is {cap.toFixed(1)} req/s.
      </Callout>
      <div className="grid gap-3 lg:grid-cols-2">
        <Panel title="Throughput saturates at capacity">
          <LineChart xLabel="arrival rate λ (req/s)" yLabel="completed req/s" series={[
            { name: "throughput", color: "var(--accent)", points: pts.map((p) => [p.lambda, p.throughput]) },
            { name: "capacity c / E[S]", color: "var(--err)", dashed: true, points: pts.map((p) => [p.lambda, cap]) },
          ]} />
        </Panel>
        <Panel title="Tail latency explodes as ρ → 1">
          <LineChart xLabel="utilization ρ" yLabel="latency ms" logY series={[
            { name: "p50", color: "var(--ok)", points: pts.map((p) => [Number(p.rho.toFixed(2)), p.p50]) },
            { name: "p99", color: "var(--err)", points: pts.map((p) => [Number(p.rho.toFixed(2)), p.p99]) },
          ]} />
        </Panel>
      </div>
      <Panel title="Sweep table">
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead className="text-left text-muted"><tr><th className="py-1">λ (req/s)</th><th>ρ</th><th>throughput</th><th>p50 ms</th><th>p99 ms</th></tr></thead>
            <tbody className="font-mono">{pts.map((p) => (
              <tr key={p.lambda} className="border-t border-line"><td className="py-1">{p.lambda}</td><td>{p.rho.toFixed(2)}</td><td>{p.throughput.toFixed(2)}</td><td>{f0(p.p50)}</td><td>{f0(p.p99)}</td></tr>
            ))}</tbody>
          </table>
        </div>
      </Panel>
    </div>
  );
}

