import { useMemo, useState } from "react";
import { motion } from "motion/react";
import { CreditCard, Dice5, RotateCcw, Scissors } from "lucide-react";
import { usePlayback } from "../../hooks/usePlayback";
import { PlaybackControls } from "../../components/visualizer/PlaybackControls";
import { EventLog } from "../../components/visualizer/EventLog";
import { StepExplainer, Workspace } from "../../components/visualizer/Workspace";
import { AnimatedNumber } from "../../components/visualizer/AnimatedNumber";
import { LineChart } from "../../components/visualizer/LineChart";
import { Badge, Button, Callout, KV, NumberField, Panel, Select, SimulatedLabel, Slider, StatusBadge, Toggle, cx } from "../../components/ui";
import { Tex, TexBlock } from "../../components/Tex";
import { clamp } from "../../lib/math";
import type { VisualizerTab } from "../types";
import { EASE, StepCaption } from "../agent-termination/kit";
import {
  DEFAULT_RETRY, SCENARIOS, backoffCeiling, expectedBackoff, reliability, sanitize, simulate,
  type Jitter, type RetryConfig, type RetryRun, type ScenarioId, type Segment,
} from "./simulation";

/** One schedule: a segment of `ms` simulated milliseconds animates for segAnim(ms) seconds (at 1×). */
const segAnim = (ms: number) => clamp((ms / 1000) * 0.9, 0.6, 2.4);
const LEAD = 0.25;
const stepMs = (seg: Segment | undefined, final: boolean) => (final ? 2600 : !seg ? 2200 : (LEAD + segAnim(seg.end - seg.start) * Math.max(1, seg.kind === "attempt" ? (seg.serverEnd - seg.start) / Math.max(1, seg.end - seg.start) : 1) + 1.1) * 1000);

const outcomeText = (s: Extract<Segment, { kind: "attempt" }>) =>
  s.outcome === 200 ? "200 OK" : s.outcome === "timeout" ? "timeout" : s.outcome === "deadline" ? "deadline" : `HTTP ${s.outcome}`;
const outcomeTone = (s: Extract<Segment, { kind: "attempt" }>) => (s.outcome === 200 ? "ok" : s.retryable ? "warn" : "err");

export default function RetriesVisualizer({ tab }: { tab: VisualizerTab }) {
  const [cfg, setCfg] = useState<RetryConfig>(DEFAULT_RETRY);
  const [replay, setReplay] = useState(0);
  const [failRate, setFailRate] = useState(0.3);
  const c = sanitize(cfg);
  const { run, steps } = useMemo(() => simulate(cfg), [cfg]);
  const pb = usePlayback(steps.length, (i) => stepMs(run.segments[i - 1], i === steps.length - 1));
  const step = steps[pb.index];
  const s = step.state;
  const prev = pb.index > 0 ? steps[pb.index - 1].state : s;
  const k = 1 / pb.speed;
  const animKey = `${step.id}-${replay}`;
  const current = pb.index > 0 && !s.done ? run.segments[s.visible - 1] : undefined;
  const set = (p: Partial<RetryConfig>) => setCfg((x) => ({ ...x, ...p }));

  if (tab === "metrics") return <ReliabilityMetrics cfg={c} f={failRate} setF={setFailRate} />;

  const story = narrate(step.id, current, run, c);
  return (
    <Workspace
      main={
        <>
          <Panel title="Call timeline" actions={<SimulatedLabel>Simulated server &amp; latency</SimulatedLabel>}>
            <StepCaption index={pb.index} total={steps.length} stepKey={step.id} headline={story.headline} sub={story.sub} onReplay={() => setReplay((r) => r + 1)} />
            <DeadlineBar from={prev.now} to={s.now} deadline={c.deadlineMs} delay={LEAD * k} duration={(current ? segAnim(current.end - current.start) : 0.4) * k} animKey={animKey} />
            <div className="mt-3 overflow-x-auto">
              <div className="min-w-[560px]">
                <Gantt run={run} visible={s.visible} current={current} cfg={c} k={k} animKey={animKey} done={s.done} />
              </div>
            </div>
            <div className="mt-2 flex flex-wrap gap-3 text-[11px] text-muted">
              <span className="flex items-center gap-1"><span className="inline-block h-2.5 w-4 rounded-sm bg-ok/70" />success</span>
              <span className="flex items-center gap-1"><span className="inline-block h-2.5 w-4 rounded-sm bg-warn/70" />retryable failure</span>
              <span className="flex items-center gap-1"><span className="inline-block h-2.5 w-4 rounded-sm bg-err/70" />permanent failure</span>
              <span className="flex items-center gap-1"><span className="inline-block h-2.5 w-4 rounded-sm border border-dashed border-muted" />backoff wait</span>
              <span className="flex items-center gap-1"><Scissors size={11} />client stopped waiting</span>
            </div>
            {SCENARIOS[c.scenario].sideEffect && <ChargeLedger run={run} visible={s.visible} current={current} k={k} animKey={animKey} />}
          </Panel>
          <Panel title="What's happening in this step">
            <motion.div key={animKey} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.2 }}>
              <StepCard current={current} run={run} cfg={c} done={s.done} k={k} />
            </motion.div>
          </Panel>
          <PlaybackControls pb={pb} stepTitles={steps.map((x) => x.title)} />
          <Panel title="Retry policy & faults" actions={<Button size="sm" variant="ghost" onClick={() => { setCfg(DEFAULT_RETRY); pb.restart(); }}><RotateCcw size={12} /> Reset</Button>}>
            <div className="grid gap-4 sm:grid-cols-2">
              <Select label="Server behaviour (scenario)" value={cfg.scenario} onChange={(v: ScenarioId) => set({ scenario: v })} options={(Object.keys(SCENARIOS) as ScenarioId[]).map((id) => ({ value: id, label: SCENARIOS[id].label }))} />
              <Select label="Jitter" value={cfg.jitter} onChange={(v: Jitter) => set({ jitter: v })} options={[{ value: "none", label: "None (exact exponential)" }, { value: "full", label: "Full jitter: random 0…ceiling" }, { value: "equal", label: "Equal jitter: ceiling/2 + random" }]} />
              <Slider label="Max attempts" value={cfg.maxAttempts} min={1} max={6} onChange={(v) => set({ maxAttempts: v })} />
              <Slider label="Per-attempt timeout" value={cfg.timeoutMs} min={200} max={4000} step={100} format={(v) => `${v} ms`} onChange={(v) => set({ timeoutMs: v })} />
              <Slider label="Base backoff" value={cfg.baseBackoffMs} min={0} max={1000} step={50} format={(v) => `${v} ms`} onChange={(v) => set({ baseBackoffMs: v })} hint="Doubles after every failure, capped at 4000 ms." />
              <Slider label="Overall deadline" value={cfg.deadlineMs} min={1000} max={15000} step={500} format={(v) => `${v} ms`} onChange={(v) => set({ deadlineMs: v })} />
              <Toggle label="Send an idempotency key" checked={cfg.idempotencyKey} onChange={(v) => set({ idempotencyKey: v })} hint="Server stores the result per key and replays it on a retry." />
              <Toggle label="Break it: retry every error (incl. 400/404)" checked={cfg.retryAllErrors} onChange={(v) => set({ retryAllErrors: v })} />
              {cfg.jitter !== "none" && <NumberField label="Jitter seed" value={cfg.seed} step={1} min={1} max={999} onChange={(v) => set({ seed: Math.round(v) })} />}
            </div>
            <p className="mt-3 text-xs text-muted">{SCENARIOS[c.scenario].note}</p>
          </Panel>
        </>
      }
      inspector={
        <>
          <StepExplainer index={pb.index} title={step.title} what={step.description}
            why={current?.kind === "backoff" ? "Waiting longer after each failure gives an overloaded service room to recover; jitter stops many clients retrying in lock-step." : current?.kind === "attempt" && (current.outcome === "timeout" || current.outcome === "deadline") ? "A timeout only stops the client waiting. Whatever the server was doing may still complete." : undefined} />
          <Panel title="Calculation">
            <Calc current={current} cfg={c} />
          </Panel>
          <Panel title="Run so far">
            <KV rows={[
              ["elapsed", `${s.now} ms`],
              ["deadline left", `${Math.max(0, c.deadlineMs - s.now)} ms`],
              ["attempts made", run.segments.slice(0, s.visible).filter((x) => x.kind === "attempt").length],
              ["time spent waiting", `${run.segments.slice(0, s.visible).filter((x) => x.kind === "backoff").reduce((a, x) => a + x.end - x.start, 0)} ms`],
              ...(SCENARIOS[c.scenario].sideEffect ? [["card charges", s.charges] as [string, number]] : []),
              ["status", s.done ? <StatusBadge key="st" status={run.status === "success" ? (run.charges > 1 ? "warning" : "ok") : "error"} label={run.status} /> : "running"],
            ]} />
          </Panel>
          <Panel title="Events"><EventLog events={step.events} /></Panel>
        </>
      }
    />
  );
}

function narrate(id: string, cur: Segment | undefined, run: RetryRun, c: RetryConfig) {
  if (id === "start") return { headline: "One logical request, several possible attempts", sub: `Budget: ${c.maxAttempts} attempts × ${c.timeoutMs} ms each, inside a ${c.deadlineMs} ms deadline.` };
  if (!cur) {
    const dup = run.charges > 1;
    return {
      headline: run.status === "success" ? (dup ? `Succeeded, but the card was charged ${run.charges} times` : `Succeeded after ${run.attempts} attempt${run.attempts > 1 ? "s" : ""} in ${run.endMs} ms`) : `Gave up: ${run.status === "non-retryable" ? "the error is permanent" : run.status === "deadline" ? "the deadline ran out" : "every attempt failed"}`,
      sub: dup ? "The first call timed out on the client but completed on the server. Retrying it charged again." : run.reason,
    };
  }
  if (cur.kind === "backoff") return { headline: `Back off for ${cur.end - cur.start} ms before attempt ${cur.n + 1}`, sub: c.jitter === "none" ? `Exponential: ${c.baseBackoffMs} × 2^${cur.n - 1}, capped at ${c.maxBackoffMs} ms.` : `Jitter picks a random point below the ${cur.ceiling} ms ceiling (seeded).` };
  const cut = cur.outcome === "timeout" || cur.outcome === "deadline";
  return {
    headline: cut ? `Attempt ${cur.n} hits the ${cur.outcome === "timeout" ? `${c.timeoutMs} ms timeout` : "deadline"}: the client stops waiting` : `Attempt ${cur.n} returns ${outcomeText(cur)}`,
    sub: cur.charge === "new" && cut ? "…but the server keeps going and charges the card anyway (dashed bar)." : cur.charge === "replayed" ? "Same idempotency key: the server returns the original charge instead of charging again." : cur.charge === "new" && run.segments.some((x) => x.kind === "attempt" && x.n < cur.n && x.charge) ? "No idempotency key, so the server treats this as a brand-new purchase." : cur.outcome === 200 ? "The response arrived inside the timeout." : cur.retryable ? `${cut ? "Timeouts" : `HTTP ${cur.outcome}`} ${cut ? "are" : "is"} transient: worth retrying.` : `HTTP ${cur.outcome} is permanent: retrying cannot help.`,
  };
}

function DeadlineBar({ from, to, deadline, delay, duration, animKey }: { from: number; to: number; deadline: number; delay: number; duration: number; animKey: string }) {
  const pct = (v: number) => `${Math.min(100, (v / deadline) * 100)}%`;
  const frac = to / deadline;
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs"><span className="font-medium">Deadline budget used</span><span className="font-mono text-muted"><AnimatedNumber key={animKey} value={to} from={from} digits={0} delay={delay} duration={duration} /> / {deadline} ms</span></div>
      <div className="relative h-3 overflow-hidden rounded bg-surface-2">
        <motion.div key={animKey} className={cx("absolute inset-y-0 left-0 rounded", frac >= 1 ? "bg-err/80" : frac > 0.75 ? "bg-warn/80" : "bg-accent/70")} initial={{ width: pct(from) }} animate={{ width: pct(to) }} transition={{ delay, duration, ease: "linear" }} />
      </div>
    </div>
  );
}

const W = 720, L = 74, R = 18, CLIENT_Y = 22, SERVER_Y = 86, BAR_H = 34, SRV_H = 24, AXIS_Y = 132, H = 166;

function Gantt({ run, visible, current, cfg, k, animKey, done }: { run: RetryRun; visible: number; current?: Segment; cfg: RetryConfig; k: number; animKey: string; done: boolean }) {
  const maxT = Math.max(run.endMs, ...run.segments.map((x) => (x.kind === "attempt" ? x.serverEnd : x.end)), 500);
  const domain = Math.min(Math.max(maxT * 1.08, 1000), Math.max(cfg.deadlineMs, maxT * 1.08));
  const sx = (t: number) => L + (t / domain) * (W - L - R);
  const tickStep = [100, 200, 250, 500, 1000, 2000, 2500, 5000, 10000].find((v) => domain / v <= 6) ?? 20000;
  const ticks = Array.from({ length: Math.floor(domain / tickStep) + 1 }, (_, i) => i * tickStep);
  const shown = run.segments.slice(0, visible);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label={`Timeline of ${shown.length} segments; total ${run.endMs} ms`}>
      <text x={8} y={CLIENT_Y + 21} fontSize={11} fontWeight={600} fill="var(--fg)">Client</text>
      <text x={8} y={SERVER_Y + 16} fontSize={11} fontWeight={600} fill="var(--fg)">Server</text>
      <line x1={L} x2={W - R} y1={CLIENT_Y + BAR_H / 2} y2={CLIENT_Y + BAR_H / 2} stroke="var(--line)" />
      <line x1={L} x2={W - R} y1={SERVER_Y + SRV_H / 2} y2={SERVER_Y + SRV_H / 2} stroke="var(--line)" />
      {ticks.map((t) => (
        <g key={t}>
          <line x1={sx(t)} x2={sx(t)} y1={AXIS_Y - 4} y2={AXIS_Y} stroke="var(--muted)" />
          <text x={sx(t)} y={AXIS_Y + 13} fontSize={10} textAnchor={sx(t) > W - R - 20 ? "end" : "middle"} fill="var(--muted)" fontFamily="var(--font-mono)">{t}</text>
        </g>
      ))}
      <line x1={L} x2={W - R} y1={AXIS_Y} y2={AXIS_Y} stroke="var(--muted)" />
      <text x={L} y={H - 2} fontSize={10} fill="var(--muted)">time (simulated ms) →</text>
      {cfg.deadlineMs <= domain && (
        <g>
          <line x1={sx(cfg.deadlineMs)} x2={sx(cfg.deadlineMs)} y1={8} y2={AXIS_Y} stroke="var(--err)" strokeDasharray="4 3" />
          <text x={sx(cfg.deadlineMs) - 4} y={14} fontSize={10} textAnchor="end" fill="var(--err)">deadline</text>
        </g>
      )}
      {shown.map((seg) => {
        const isCur = seg === current;
        const dur = segAnim(seg.end - seg.start) * k;
        const x0 = sx(seg.start), w = Math.max(2, sx(seg.end) - x0);
        const grow = (width: number, d: number) => (isCur ? { initial: { width: 0 }, animate: { width }, transition: { delay: LEAD * k, duration: d, ease: "linear" as const } } : { initial: false as const, animate: { width } });
        if (seg.kind === "backoff") {
          return (
            <g key={`${seg.kind}${seg.n}-${isCur ? animKey : "s"}`}>
              <motion.rect x={x0} y={CLIENT_Y + 6} height={BAR_H - 12} rx={4} fill="none" stroke="var(--muted)" strokeDasharray="4 3" {...grow(w, dur)} />
              <motion.text x={x0 + w / 2} y={CLIENT_Y + BAR_H / 2 + 4} fontSize={10} textAnchor="middle" fill="var(--muted)" fontFamily="var(--font-mono)"
                initial={isCur ? { opacity: 0 } : false} animate={{ opacity: w > 56 ? 1 : 0 }} transition={{ delay: (LEAD + 0.2) * k }}>wait {seg.end - seg.start}</motion.text>
            </g>
          );
        }
        const tone = outcomeTone(seg);
        const fill = tone === "ok" ? "var(--ok)" : tone === "warn" ? "var(--warn)" : "var(--err)";
        const cut = seg.outcome === "timeout" || seg.outcome === "deadline";
        const srvW = Math.max(2, sx(seg.serverEnd) - x0);
        const srvDur = dur * ((seg.serverEnd - seg.start) / Math.max(1, seg.end - seg.start));
        // Alternate sub-rows so overlapping server work (a retry while the old request still runs) stays visible.
        const srvY = SERVER_Y + (seg.n % 2 === 0 ? 12 : 0);
        return (
          <g key={`a${seg.n}-${isCur ? animKey : "s"}`}>
            <motion.rect x={x0} y={CLIENT_Y} height={BAR_H} rx={5} fill={fill} fillOpacity={0.75} {...grow(w, dur)} />
            <motion.text x={x0 + 6} y={CLIENT_Y + 14} fontSize={10} fontWeight={700} fill="var(--fg)" initial={isCur ? { opacity: 0 } : false} animate={{ opacity: 1 }} transition={{ delay: LEAD * k }}>#{seg.n}</motion.text>
            <motion.text x={x0 + 6} y={CLIENT_Y + 27} fontSize={10} fill="var(--fg)" fontFamily="var(--font-mono)" initial={isCur ? { opacity: 0 } : false} animate={{ opacity: w > 40 ? 1 : 0 }} transition={{ delay: (LEAD + dur / k) * k }}>{outcomeText(seg)}</motion.text>
            {/* Server-side processing: keeps going past the client's cut-off. */}
            <motion.rect x={x0} y={srvY} height={SRV_H - 6} rx={4} fill="var(--violet)" fillOpacity={0.18} stroke="var(--violet)" strokeDasharray={cut ? "5 3" : undefined} {...grow(srvW, srvDur)} />
            {cut && (
              <motion.g initial={isCur ? { opacity: 0, scale: 0.4 } : false} animate={{ opacity: 1, scale: 1 }} transition={{ delay: (LEAD * k) + dur, duration: 0.3 * k }} style={{ transformOrigin: `${sx(seg.end)}px ${CLIENT_Y + BAR_H / 2}px` }}>
                <line x1={sx(seg.end)} x2={sx(seg.end)} y1={CLIENT_Y - 6} y2={SERVER_Y + SRV_H + 8} stroke="var(--err)" strokeWidth={2} />
                <rect x={sx(seg.end) - 26} y={CLIENT_Y - 18} width={52} height={14} rx={7} fill="var(--err)" />
                <text x={sx(seg.end)} y={CLIENT_Y - 8} fontSize={9} textAnchor="middle" fill="var(--surface)" fontWeight={700}>✂ {seg.outcome}</text>
              </motion.g>
            )}
            {seg.charge && (
              <motion.g initial={isCur ? { opacity: 0, y: -10 } : false} animate={{ opacity: 1, y: 0 }} transition={{ delay: LEAD * k + srvDur, duration: 0.35 * k, ease: EASE }}>
                <circle cx={sx(seg.serverEnd)} cy={srvY + (SRV_H - 6) / 2} r={9} fill={seg.charge === "new" ? "var(--warn)" : "var(--ok)"} stroke="var(--surface)" strokeWidth={2} />
                <text x={sx(seg.serverEnd)} y={srvY + (SRV_H - 6) / 2 + 4} fontSize={10} textAnchor="middle" fontWeight={700} fill="var(--fg)">{seg.charge === "new" ? "$" : "↺"}</text>
              </motion.g>
            )}
          </g>
        );
      })}
      {current && (
        <motion.line key={`cursor-${animKey}`} y1={8} y2={AXIS_Y} stroke="var(--accent)" strokeWidth={1.5}
          initial={{ x1: sx(current.start), x2: sx(current.start) }} animate={{ x1: sx(current.end), x2: sx(current.end) }} transition={{ delay: LEAD * k, duration: segAnim(current.end - current.start) * k, ease: "linear" }} />
      )}
      {done && (
        <motion.g key={`done-${animKey}`} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 * k }}>
          <line x1={sx(run.endMs)} x2={sx(run.endMs)} y1={8} y2={AXIS_Y} stroke="var(--fg)" strokeWidth={2} />
          <text x={Math.min(sx(run.endMs) + 4, W - 120)} y={CLIENT_Y + BAR_H + 16} fontSize={10} fontWeight={700} fill="var(--fg)">{run.status === "success" ? "✓ done" : "✕ gave up"} at {run.endMs} ms</text>
        </motion.g>
      )}
    </svg>
  );
}

function ChargeLedger({ run, visible, current, k, animKey }: { run: RetryRun; visible: number; current?: Segment; k: number; animKey: string }) {
  const charged = run.segments.slice(0, visible).filter((x): x is Extract<Segment, { kind: "attempt" }> => x.kind === "attempt" && x.charge !== undefined);
  let count = 0;
  return (
    <div className="mt-3 rounded-md border border-line p-2">
      <div className="mb-1.5 flex items-center gap-1.5 text-xs font-medium"><CreditCard size={13} /> Customer's card statement (side effect)</div>
      <div className="flex flex-wrap gap-2">
        {charged.length === 0 && <span className="text-xs text-muted">No charges yet.</span>}
        {charged.map((x) => {
          const isNew = x.charge === "new";
          if (isNew) count++;
          const dup = isNew && count > 1;
          const isCur = x === current;
          const delay = isCur ? (LEAD + segAnim(x.end - x.start) * ((x.serverEnd - x.start) / Math.max(1, x.end - x.start))) * k : 0;
          return (
            <motion.div key={`${x.n}-${isCur ? animKey : "s"}`} initial={isCur ? { opacity: 0, scale: 0.6, y: -14 } : false} animate={{ opacity: 1, scale: 1, y: 0 }} transition={{ delay, type: "spring", stiffness: 260, damping: 18 }}
              className={cx("rounded-md border px-2 py-1 font-mono text-xs", dup ? "border-err bg-err-soft" : isNew ? "border-line bg-surface-2" : "border-ok/40 bg-ok-soft")}>
              {isNew ? `ch_${count}  −$49.00` : "ch_1 replayed  $0.00"} <span className="font-sans">{dup ? "✕ duplicate" : isNew ? "✓" : "✓ deduplicated"}</span>
              <span className="block text-[10px] text-muted">from attempt #{x.n}</span>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}

function StepCard({ current, run, cfg, done, k }: { current?: Segment; run: RetryRun; cfg: RetryConfig; done: boolean; k: number }) {
  if (done) {
    const attempts = run.segments.filter((x) => x.kind === "attempt");
    const waited = run.segments.filter((x) => x.kind === "backoff").reduce((a, x) => a + x.end - x.start, 0);
    const parts = [
      { label: "time in attempts", v: run.endMs - waited, cls: "bg-accent/70" },
      { label: "time waiting (backoff)", v: waited, cls: "bg-muted/40" },
    ];
    return (
      <div className="space-y-3 text-sm">
        <div className="flex flex-wrap items-center gap-2">
          <StatusBadge status={run.status === "success" ? (run.charges > 1 ? "warning" : "ok") : "error"} label={run.status} />
          <span>{attempts.length} attempt{attempts.length > 1 ? "s" : ""} · {run.endMs} ms total{SCENARIOS[cfg.scenario].sideEffect ? ` · ${run.charges} charge${run.charges === 1 ? "" : "s"}` : ""}</span>
        </div>
        <div className="flex h-6 overflow-hidden rounded bg-surface-2">
          {parts.map((p, i) => (
            <motion.div key={p.label} className={p.cls} initial={{ width: 0 }} animate={{ width: `${(p.v / Math.max(1, run.endMs)) * 100}%` }} transition={{ delay: (0.2 + i * 0.6) * k, duration: 0.6 * k, ease: EASE }} />
          ))}
        </div>
        <div className="flex flex-wrap gap-3 text-xs text-muted">{parts.map((p) => <span key={p.label}><span className={cx("mr-1 inline-block h-2 w-3 rounded-sm", p.cls)} />{p.label}: {p.v} ms</span>)}</div>
        {run.charges > 1 && <Callout tone="err" title="Duplicate side effect">The timed-out attempt still charged the card, and the retry charged it again. Turn on <strong>Send an idempotency key</strong> and replay.</Callout>}
        {run.status === "non-retryable" && <Callout tone="ok" title="Failing fast is correct here">A 400 means the request is wrong; retrying only burns time and load.</Callout>}
        {run.status === "exhausted" && cfg.retryAllErrors && cfg.scenario === "badRequest" && <Callout tone="err" title="Wasted retries">Every retry of a 400 fails identically. Retry only transient errors.</Callout>}
      </div>
    );
  }
  if (!current) {
    return (
      <div className="space-y-2">
        <p className="text-sm">The client will make at most these attempts. Each box is the most time an attempt is allowed to take:</p>
        <div className="flex flex-wrap items-center gap-1.5">
          {Array.from({ length: cfg.maxAttempts }, (_, i) => (
            <motion.div key={i} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: (0.2 + i * 0.25) * k, duration: 0.35 * k, ease: EASE }} className="flex items-center gap-1.5">
              <span className="rounded border border-accent/40 bg-accent-soft px-2 py-1 font-mono text-xs">#{i + 1} ≤ {cfg.timeoutMs} ms</span>
              {i < cfg.maxAttempts - 1 && <span className="font-mono text-[11px] text-muted">wait ~{backoffCeiling(i + 1, cfg.baseBackoffMs, cfg.maxBackoffMs)}</span>}
            </motion.div>
          ))}
        </div>
        <p className="text-xs text-muted">Worst case without a deadline: {cfg.maxAttempts * cfg.timeoutMs + Array.from({ length: cfg.maxAttempts - 1 }, (_, i) => backoffCeiling(i + 1, cfg.baseBackoffMs, cfg.maxBackoffMs)).reduce((a, b) => a + b, 0)} ms. The deadline ({cfg.deadlineMs} ms) caps it.</p>
      </div>
    );
  }
  if (current.kind === "backoff") {
    const pct = (v: number) => `${(v / Math.max(1, current.ceiling)) * 100}%`;
    const wait = current.end - current.start;
    const lo = cfg.jitter === "equal" ? current.ceiling / 2 : 0;
    return (
      <div className="space-y-2 text-sm">
        <p>Ceiling for this wait: <span className="font-mono">min({cfg.maxBackoffMs}, {cfg.baseBackoffMs} × 2^{current.n - 1}) = {current.ceiling} ms</span></p>
        <div className="relative h-8 rounded bg-surface-2">
          {cfg.jitter !== "none" && <div className="absolute inset-y-0 rounded bg-accent/15" style={{ left: pct(lo), right: 0 }} title="possible range" />}
          <motion.div className="absolute inset-y-1 left-0 rounded bg-muted/40" initial={{ width: 0 }} animate={{ width: pct(wait) }} transition={{ delay: 0.5 * k, duration: 0.8 * k, ease: EASE }} />
          {cfg.jitter !== "none" && (
            <motion.div className="absolute top-1/2 -translate-y-1/2" initial={{ left: pct(lo), opacity: 0 }} animate={{ left: [pct(lo), pct(current.ceiling), pct(lo + (current.ceiling - lo) * 0.3), pct(wait)], opacity: 1 }} transition={{ duration: 0.9 * k, ease: "easeInOut" }}>
              <Dice5 size={18} className="-ml-2 text-accent" aria-hidden />
            </motion.div>
          )}
        </div>
        <div className="flex justify-between font-mono text-[10px] text-muted"><span>0 ms</span><span>{current.ceiling} ms ceiling</span></div>
        <p className="text-xs text-muted">{cfg.jitter === "none" ? "No jitter: every client waits exactly the ceiling, so clients that failed together retry together." : `Random u = ${current.u.toFixed(3)} (seeded) picks ${wait} ms inside the shaded range, spreading clients apart.`}</p>
      </div>
    );
  }
  const cut = current.outcome === "timeout" || current.outcome === "deadline";
  const span = Math.max(current.serverEnd - current.start, cfg.timeoutMs) * 1.1;
  const pct = (v: number) => `${(v / span) * 100}%`;
  const d = segAnim(current.end - current.start);
  const srvD = d * ((current.serverEnd - current.start) / Math.max(1, current.end - current.start));
  return (
    <div className="space-y-2 text-sm">
      <div className="grid grid-cols-[90px_minmax(0,1fr)] items-center gap-2 text-xs">
        <span className="text-muted">client waits</span>
        <div className="relative h-5 rounded bg-surface-2">
          <motion.div className={cx("absolute inset-y-0 left-0 rounded", outcomeTone(current) === "ok" ? "bg-ok/70" : outcomeTone(current) === "warn" ? "bg-warn/70" : "bg-err/70")} initial={{ width: 0 }} animate={{ width: pct(current.end - current.start) }} transition={{ delay: LEAD * k, duration: d * k, ease: "linear" }} />
          <div className="absolute inset-y-[-4px] w-0.5 bg-err" style={{ left: pct(cfg.timeoutMs) }} title="timeout" />
        </div>
        <span className="text-muted">server works</span>
        <div className="relative h-5 rounded bg-surface-2">
          <motion.div className="absolute inset-y-0 left-0 rounded border border-violet bg-violet/20" initial={{ width: 0 }} animate={{ width: pct(current.serverEnd - current.start) }} transition={{ delay: LEAD * k, duration: srvD * k, ease: "linear" }} />
          <div className="absolute inset-y-[-4px] w-0.5 bg-err" style={{ left: pct(cfg.timeoutMs) }} />
        </div>
      </div>
      <p className="text-right font-mono text-[10px] text-err">red line = {cfg.timeoutMs} ms timeout</p>
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: (LEAD + d) * k }} className="flex flex-wrap items-center gap-2 text-xs">
        <Badge tone={outcomeTone(current) === "ok" ? "ok" : outcomeTone(current) === "warn" ? "warn" : "err"}>{outcomeText(current)}</Badge>
        {current.outcome !== 200 && <span>retryable? <strong>{current.retryable ? "yes" : "no"}</strong>{cfg.retryAllErrors && typeof current.outcome === "number" && ![429, 500, 503].includes(current.outcome) ? " (only because 'retry every error' is on)" : ""}</span>}
        {cut && current.charge === "new" && <Badge tone="warn">server charged the card anyway</Badge>}
        {current.charge === "replayed" && <Badge tone="ok">idempotent replay: no new charge</Badge>}
      </motion.div>
    </div>
  );
}

function Calc({ current, cfg }: { current?: Segment; cfg: RetryConfig }) {
  if (current?.kind === "backoff") {
    const w = current.end - current.start;
    const body = cfg.jitter === "none" ? `${current.ceiling}` : cfg.jitter === "full" ? `u \\cdot c = ${current.u.toFixed(3)} \\times ${current.ceiling} \\approx ${w}` : `\\tfrac{c}{2} + u \\cdot \\tfrac{c}{2} = ${current.ceiling / 2} + ${current.u.toFixed(3)} \\times ${current.ceiling / 2} \\approx ${w}`;
    return <TexBlock>{`\\begin{aligned} c_{${current.n}} &= \\min(\\text{cap},\\, b \\cdot 2^{n-1}) = \\min(${cfg.maxBackoffMs},\\, ${cfg.baseBackoffMs} \\cdot 2^{${current.n - 1}}) = ${current.ceiling} \\\\ \\text{wait} &= ${body}\\,\\text{ms} \\end{aligned}`}</TexBlock>;
  }
  if (current?.kind === "attempt") {
    return <TexBlock>{`\\begin{aligned} t_{\\text{wait}} &= \\min(\\text{latency},\\ \\text{timeout},\\ \\text{deadline} - t_0) \\\\ &= \\min(${current.serverEnd - current.start},\\ ${cfg.timeoutMs},\\ ${cfg.deadlineMs} - ${current.start}) \\\\ &= ${current.end - current.start}\\,\\text{ms} \\end{aligned}`}</TexBlock>;
  }
  return (
    <div className="space-y-2 text-xs">
      <TexBlock>{`\\text{wait}_n = \\operatorname{jitter}\\big(\\min(\\text{cap},\\ b \\cdot 2^{\\,n-1})\\big)`}</TexBlock>
      <p className="text-muted">Base <Tex>{`b = ${cfg.baseBackoffMs}`}</Tex> ms, cap {cfg.maxBackoffMs} ms. See the Metrics tab for success probability and expected latency.</p>
    </div>
  );
}

function ReliabilityMetrics({ cfg, f, setF }: { cfg: RetryConfig; f: number; setF: (v: number) => void }) {
  const ns = [1, 2, 3, 4, 5, 6];
  const rows = ns.map((n) => ({ n, ...reliability(f, n, cfg) }));
  const cur = rows[cfg.maxAttempts - 1] ?? rows.at(-1)!;
  const terms = Array.from({ length: cfg.maxAttempts }, (_, i) => i + 1);
  return (
    <div className="space-y-3">
      <Callout title="An analytic model, not the scripted scenario">Assume each attempt fails independently with probability f, a failure costs the full timeout ({cfg.timeoutMs} ms) and a success takes 300 ms. Backoff uses your base, cap and jitter (expected value). Change them on the Visualize tab.</Callout>
      <Panel title="Per-attempt failure probability">
        <Slider label="f" value={f} min={0} max={0.9} step={0.05} format={(v) => v.toFixed(2)} onChange={setF} />
      </Panel>
      <div className="grid gap-3 lg:grid-cols-2">
        <Panel title="P(success) vs max attempts">
          <LineChart series={[{ name: "P(success)", points: rows.map((r) => [r.n, r.pSuccess]), color: "var(--ok)" }]} xLabel="max attempts" yLabel="P(success)" markerX={cfg.maxAttempts} height={200} />
        </Panel>
        <Panel title="Expected latency vs max attempts">
          <LineChart series={[{ name: "E[latency] (ms)", points: rows.map((r) => [r.n, r.expectedMs]), color: "var(--accent)" }]} xLabel="max attempts" yLabel="ms" markerX={cfg.maxAttempts} height={200} />
        </Panel>
      </div>
      <Panel title={`With ${cfg.maxAttempts} attempts`}>
        <TexBlock>{`P(\\text{success}) = 1 - f^{n} = 1 - ${f.toFixed(2)}^{${cfg.maxAttempts}} = ${cur.pSuccess.toFixed(4)}`}</TexBlock>
        <div className="mt-2"><TexBlock>{`\\mathbb{E}[T] = \\sum_{i=1}^{n} f^{\\,i-1}\\big(\\bar w_{i-1} + (1-f)\\,t_{ok} + f\\,t_{\\text{timeout}}\\big) = ${terms.map((i) => `${f.toFixed(2)}^{${i - 1}}(${i > 1 ? `${expectedBackoff(i - 1, cfg).toFixed(0)} + ` : ""}${((1 - f) * 300 + f * cfg.timeoutMs).toFixed(0)})`).join(" + ")} \\approx ${cur.expectedMs.toFixed(0)}\\,\\text{ms}`}</TexBlock></div>
        <p className="mt-2 text-xs text-muted">More attempts raise the success probability with diminishing returns, while the expected (and worst-case) latency keeps growing. The deadline is what keeps the tail bounded.</p>
      </Panel>
    </div>
  );
}
