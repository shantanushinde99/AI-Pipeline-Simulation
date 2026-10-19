import { useMemo, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { Clock, RotateCcw, ShieldAlert, ShieldCheck } from "lucide-react";
import { usePlayback } from "../../hooks/usePlayback";
import { PlaybackControls } from "../../components/visualizer/PlaybackControls";
import { EventLog } from "../../components/visualizer/EventLog";
import { StepExplainer, Workspace } from "../../components/visualizer/Workspace";
import { AnimatedNumber } from "../../components/visualizer/AnimatedNumber";
import { Badge, Button, KV, Panel, Select, SimulatedLabel, Slider, cx } from "../../components/ui";
import { TexBlock } from "../../components/Tex";
import type { VisualizerTab } from "../types";
import { StepCaption, ease } from "../monitoring-and-alerting/shared";
import {
  BATCHES, BINS, DEFAULT_INPUT, DRIFTS, N_BATCH, N_REF, ksCritical, simulate, truthCurve,
  type BatchStats, type DriftInput, type DriftRun, type DriftState, type ScoredPt,
} from "./simulation";

/** One schedule (seconds at 1×): points slide, labels flip, PSI bars grow, delayed labels land, alerts pop. */
const T = { move: 1.1, flip: 0.75, psi: 1.3, labels: 1.9, alert: 2.5 };
const stepMs = (i: number, total: number) => (i === 0 ? 3000 : i === total - 1 ? 3600 : 3800);

export default function DriftVisualizer({ tab }: { tab: VisualizerTab }) {
  const [input, setInput] = useState<DriftInput>(DEFAULT_INPUT);
  const [replay, setReplay] = useState(0);
  const run = useMemo(() => simulate(input), [input]);
  const steps = run.steps;
  const pb = usePlayback(steps.length, (i) => stepMs(i, steps.length));
  const step = steps[pb.index];
  const s = step.state;
  const prev = pb.index > 0 ? steps[pb.index - 1].state : undefined;
  const k = 1 / pb.speed;
  const animKey = `${step.id}-${replay}`;
  const set = (p: Partial<DriftInput>) => setInput((v) => ({ ...v, ...p }));
  const cur = s.stats.at(-1);

  if (tab === "metrics") return null;

  return (
    <Workspace
      main={
        <>
          <Panel title="Production traffic vs the trained model" actions={<SimulatedLabel>Synthetic 2-D data</SimulatedLabel>}>
            <StepCaption index={pb.index} total={steps.length} id={step.id} headline={s.headline} sub={s.sub} onReplay={() => setReplay((r) => r + 1)} />
            <div className="grid gap-4 lg:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)]">
              <Plane run={run} s={s} prev={prev} animKey={animKey} k={k} />
              <Detectors s={s} input={input} animKey={animKey} k={k} />
            </div>
          </Panel>
          <Panel title="What's happening in this step">
            <AnimatePresence mode="wait" initial={false}>
              <motion.div key={animKey} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: 0.2 }}>
                {s.phase === "reference" ? <ReferenceCard run={run} k={k} />
                  : s.phase === "summary" ? <SummaryCard run={run} input={input} k={k} />
                  : <BatchCard run={run} s={s} prev={prev} k={k} />}
              </motion.div>
            </AnimatePresence>
          </Panel>
          <PlaybackControls pb={pb} stepTitles={steps.map((x) => x.title)} />
          {pb.index === 0 && !pb.playing && <p className="-mt-1 text-xs text-muted">Press <strong>Play</strong> to stream 8 production batches. Then switch between covariate shift and concept drift and compare which detector fires.</p>}
          <Panel title="Drift & detectors" actions={<Button size="sm" variant="ghost" onClick={() => { setInput(DEFAULT_INPUT); pb.restart(); }}><RotateCcw size={12} /> Reset</Button>}>
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
              <Select label="What changes in the world" value={input.kind} options={DRIFTS.map((d) => ({ value: d.id, label: d.name }))} onChange={(v) => set({ kind: v })} />
              <Slider label="Drift magnitude" value={input.magnitude} min={0} max={1} step={0.05} format={(v) => `${Math.round(v * 100)}%`} onChange={(v) => set({ magnitude: v })} />
              <Slider label="Label delay" value={input.labelDelay} min={0} max={4} format={(v) => `${v} batch${v === 1 ? "" : "es"}`} onChange={(v) => set({ labelDelay: v })} hint="Break it: real labels (chargebacks, churn) can take weeks." />
              <Slider label="PSI alert threshold" value={input.psiThreshold} min={0.05} max={0.5} step={0.05} format={(v) => v.toFixed(2)} onChange={(v) => set({ psiThreshold: v })} hint="Common rule of thumb: 0.1 watch, 0.2 act." />
              <Slider label="Accuracy alert below" value={input.accThreshold} min={0.7} max={0.95} step={0.01} format={(v) => `${Math.round(v * 100)}%`} onChange={(v) => set({ accThreshold: v })} />
              <Slider label="Random seed" value={input.seed} min={1} max={30} onChange={(v) => set({ seed: v })} />
            </div>
          </Panel>
        </>
      }
      inspector={
        <>
          <StepExplainer index={pb.index} title={step.title} what={step.description}
            why="Input drift tests are cheap and immediate but only see P(x). Only labels reveal whether P(y|x) changed, and labels arrive late." />
          <Panel title="Detector math">
            {cur ? <DetectorMath run={run} b={cur} /> : <TexBlock>{`\\text{PSI} = \\sum_{b=1}^{${BINS}} (a_b - e_b)\\ln\\frac{a_b}{e_b}, \\quad e_b = ${(1 / BINS).toFixed(2)}`}</TexBlock>}
          </Panel>
          <Panel title="Key values">
            <KV rows={[
              ["model (standardized)", `w = [${run.model.weights.map((w) => w.toFixed(2)).join(", ")}], b = ${run.model.bias.toFixed(2)}`],
              ["reference accuracy", `${(run.refAcc * 100).toFixed(1)}%`],
              ["true accuracy (hidden)", cur ? `${(cur.acc * 100).toFixed(1)}%` : "—"],
              ["latest measured accuracy", s.measured.length ? `${(s.measured.at(-1)!.acc * 100).toFixed(1)}% (batch ${s.measured.at(-1)!.t})` : "no labels yet"],
              ["input shift / boundary rotation", cur ? `(${cur.shift.map((v) => v.toFixed(2)).join(", ")}) / ${cur.angle.toFixed(1)}°` : "—"],
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

const X0 = -2.6, X1 = 4.4, Y0 = -2.6, Y1 = 4.0, W = 440, H = 400, P = 26;
const sx = (v: number) => P + ((v - X0) / (X1 - X0)) * (W - 2 * P);
const sy = (v: number) => H - P - ((v - Y0) / (Y1 - Y0)) * (H - 2 * P);
const pathOf = (pts: [number, number][]) => pts.map(([a, b], i) => `${i ? "L" : "M"}${sx(a).toFixed(1)} ${sy(b).toFixed(1)}`).join(" ");

function modelLine(run: DriftRun): [[number, number], [number, number]] {
  const { weights: w, bias: b, mean: m, std: sd } = run.model;
  const x2 = (x1: number) => m[1] - (sd[1] / w[1]) * (b + (w[0] * (x1 - m[0])) / sd[0]);
  return [[X0, x2(X0)], [X1, x2(X1)]];
}

function Plane({ run, s, prev, animKey, k }: { run: DriftRun; s: DriftState; prev?: DriftState; animKey: string; k: number }) {
  const angle = s.stats.at(-1)?.angle ?? 0;
  const prevAngle = prev?.stats.at(-1)?.angle ?? 0;
  const [[a1, b1], [a2, b2]] = modelLine(run);
  const prevPts = prev && prev.phase !== "reference" ? prev.points : null;
  const showBatch = s.phase !== "reference";
  const wrong = s.points.filter((p) => p.pred !== p.y).length;
  return (
    <figure className="min-w-0">
      <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label={`Scatter of ${showBatch ? N_BATCH : N_REF} points with the model boundary and the true boundary rotated ${angle.toFixed(0)} degrees`}>
        <defs><clipPath id="drift-clip"><rect x={P} y={P} width={W - 2 * P} height={H - 2 * P} /></clipPath></defs>
        <rect x={P} y={P} width={W - 2 * P} height={H - 2 * P} fill="var(--surface-2)" rx={4} />
        {[-2, 0, 2, 4].map((v) => (
          <g key={v}>
            {v <= X1 && <text x={sx(v)} y={H - 8} textAnchor="middle" fontSize={10} fill="var(--muted)" fontFamily="var(--font-mono)">{v}</text>}
            {v <= Y1 && <text x={P - 6} y={sy(v) + 3} textAnchor="end" fontSize={10} fill="var(--muted)" fontFamily="var(--font-mono)">{v}</text>}
          </g>
        ))}
        <text x={W - P} y={H - 8} textAnchor="end" fontSize={10} fill="var(--muted)">x1</text>
        <text x={P + 4} y={P + 12} fontSize={10} fill="var(--muted)">x2</text>
        <g clipPath="url(#drift-clip)">
          {/* Reference cloud: where the model was trained. */}
          {run.reference.map((p, i) => <circle key={i} cx={sx(p.x[0])} cy={sy(p.x[1])} r={showBatch ? 1.8 : 3} fill="var(--muted)" opacity={showBatch ? 0.25 : 0.55} />)}
          <line x1={sx(a1)} y1={sy(b1)} x2={sx(a2)} y2={sy(b2)} stroke="var(--fg)" strokeWidth={2} strokeDasharray="7 5" />
          <motion.path key={`truth-${animKey}`} fill="none" stroke="var(--ok)" strokeWidth={2.5}
            initial={{ d: pathOf(truthCurve(prevAngle)) }} animate={{ d: pathOf(truthCurve(angle)) }} transition={{ duration: T.move * k, ease }} />
          {showBatch && s.points.map((p, i) => <BatchDot key={`${animKey}-${i}`} p={p} from={prevPts?.[i]} k={k} />)}
        </g>
        <g fontSize={10}>
          <rect x={W - P - 150} y={P + 6} width={144} height={44} rx={4} fill="var(--surface)" opacity={0.92} />
          <line x1={W - P - 142} x2={W - P - 120} y1={P + 18} y2={P + 18} stroke="var(--fg)" strokeWidth={2} strokeDasharray="5 4" />
          <text x={W - P - 114} y={P + 21} fill="var(--fg)">model boundary</text>
          <line x1={W - P - 142} x2={W - P - 120} y1={P + 38} y2={P + 38} stroke="var(--ok)" strokeWidth={2.5} />
          <text x={W - P - 114} y={P + 41} fill="var(--fg)">true boundary</text>
        </g>
      </svg>
      <figcaption className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-[11px] text-muted">
        <span className="flex items-center gap-1"><svg width="10" height="10" aria-hidden><circle cx="5" cy="5" r="4" fill="var(--accent)" /></svg>class 1</span>
        <span className="flex items-center gap-1"><svg width="10" height="10" aria-hidden><circle cx="5" cy="5" r="3.5" fill="none" stroke="var(--accent)" strokeWidth="1.5" /></svg>class 0</span>
        <span className="flex items-center gap-1"><svg width="12" height="12" aria-hidden><circle cx="6" cy="6" r="5" fill="none" stroke="var(--err)" strokeWidth="1.5" /></svg>model is wrong</span>
        <span className="flex items-center gap-1"><svg width="10" height="10" aria-hidden><circle cx="5" cy="5" r="2" fill="var(--muted)" opacity={0.5} /></svg>reference (training) data</span>
        {showBatch && <span className="font-mono">wrong: {wrong}/{N_BATCH}</span>}
      </figcaption>
    </figure>
  );
}

function BatchDot({ p, from, k }: { p: ScoredPt; from?: ScoredPt; k: number }) {
  const [x, y] = [sx(p.x[0]), sy(p.x[1])];
  // Class is shown by shape (filled vs hollow), so a label flip is a fill-opacity change.
  const start = from ? { cx: sx(from.x[0]), cy: sy(from.x[1]), opacity: 1, fillOpacity: from.y, scale: 1 } : { cx: x, cy: y, opacity: 0, fillOpacity: p.y, scale: 0 };
  return (
    <g>
      <motion.circle r={3.6} fill="var(--accent)" stroke="var(--accent)" strokeWidth={1.4} initial={start}
        animate={{ cx: x, cy: y, opacity: 1, fillOpacity: p.y, scale: 1 }}
        transition={{ duration: T.move * k, ease, fillOpacity: { delay: T.flip * k, duration: 0.35 * k } }} />
      {p.pred !== p.y && (
        <motion.circle r={6.5} fill="none" stroke="var(--err)" strokeWidth={1.3} initial={{ cx: x, cy: y, opacity: 0 }} animate={{ cx: x, cy: y, opacity: 1 }} transition={{ delay: (T.move + 0.1) * k, duration: 0.3 * k }} />
      )}
    </g>
  );
}

function Detectors({ s, input, animKey, k }: { s: DriftState; input: DriftInput; animKey: string; k: number }) {
  const b = s.stats.at(-1);
  const scale = Math.max(input.psiThreshold * 2.5, 0.5);
  const crit = ksCritical(N_REF, N_BATCH);
  const measured = new Map(s.measured.map((m) => [m.t, m.acc]));
  return (
    <div className="min-w-0 space-y-3">
      <section className="rounded-md border border-line p-2.5">
        <div className="mb-2 flex items-center justify-between gap-2">
          <h4 className="text-xs font-semibold">Feature-distribution test (PSI / KS)</h4>
          <AlertBadge key={`f-${animKey}`} on={s.featureAlert && !!b} delay={T.alert * k} label={b ? (s.featureAlert ? "DRIFT" : "no drift") : "idle"} />
        </div>
        {(["x1", "x2"] as const).map((f, j) => {
          const v = b?.psi[j] ?? 0;
          return (
            <div key={f} className="mb-1.5 grid grid-cols-[22px_minmax(0,1fr)_56px] items-center gap-2 text-xs">
              <span className="font-mono">{f}</span>
              <div className="relative h-4 rounded bg-surface-2">
                <motion.div className={cx("absolute inset-y-0 left-0 rounded", v >= input.psiThreshold ? "bg-err/70" : "bg-accent/60")} initial={false}
                  animate={{ width: `${Math.min(1, v / scale) * 100}%` }} transition={{ delay: T.psi * k, duration: 0.6 * k, ease }} />
                <div className="absolute inset-y-[-3px] w-0.5 bg-err" style={{ left: `${(input.psiThreshold / scale) * 100}%` }} title="threshold" />
              </div>
              <span className="text-right font-mono"><AnimatedNumber value={v} digits={2} delay={T.psi * k} duration={0.6 * k} /></span>
            </div>
          );
        })}
        <p className="text-[11px] text-muted">{b ? <>KS D = {b.ks.map((d) => d.toFixed(2)).join(" / ")} (5% critical value {crit.toFixed(3)}). Red tick = PSI threshold {input.psiThreshold}.</> : "Compares each feature's distribution in production with the reference data. Needs no labels."}</p>
      </section>
      <section className="rounded-md border border-line p-2.5">
        <div className="mb-2 flex items-center justify-between gap-2">
          <h4 className="text-xs font-semibold">Label-based monitor (accuracy, {input.labelDelay}-batch delay)</h4>
          <AlertBadge key={`l-${animKey}`} on={s.labelAlert} delay={T.alert * k} label={s.measured.length ? (s.labelAlert ? "ACCURACY ALERT" : "accuracy OK") : "waiting"} />
        </div>
        <div className="grid grid-cols-8 gap-1">
          {Array.from({ length: BATCHES }, (_, i) => i + 1).map((t) => {
            const acc = measured.get(t);
            const seen = t <= s.t;
            const justArrived = acc !== undefined && s.phase === "batch" && t === s.t - input.labelDelay;
            return (
              <div key={t} className={cx("rounded border px-0.5 py-1 text-center text-[10px]", !seen ? "border-dashed border-line text-muted/50" : "border-line")}>
                <div className="text-muted">b{t}</div>
                {acc !== undefined ? (
                  <motion.div key={justArrived ? animKey : "static"} className={cx("font-mono font-semibold", acc < input.accThreshold ? "text-err" : "text-ok")}
                    initial={justArrived ? { y: -16, opacity: 0 } : false} animate={{ y: 0, opacity: 1 }} transition={{ delay: T.labels * k, duration: 0.45 * k, ease }}>
                    {Math.round(acc * 100)}%
                  </motion.div>
                ) : seen ? <Clock size={11} className="mx-auto text-muted" aria-label="labels pending" /> : <span>·</span>}
              </div>
            );
          })}
        </div>
        <AccuracyChart stats={s.stats} measured={s.measured} thr={input.accThreshold} animKey={animKey} k={k} />
      </section>
    </div>
  );
}

function AlertBadge({ on, delay, label }: { on: boolean; delay: number; label: string }) {
  return (
    <motion.span initial={{ scale: on ? 0.6 : 1, opacity: on ? 0 : 1 }} animate={{ scale: 1, opacity: 1 }} transition={{ delay: on ? delay : 0, type: "spring", stiffness: 300, damping: 15 }}>
      <Badge tone={on ? "err" : "neutral"}>{on ? <ShieldAlert size={11} aria-hidden /> : <ShieldCheck size={11} aria-hidden />}{label}</Badge>
    </motion.span>
  );
}

function AccuracyChart({ stats, measured, thr, animKey, k }: { stats: BatchStats[]; measured: { t: number; acc: number }[]; thr: number; animKey: string; k: number }) {
  const w = 300, h = 110, l = 30, r = 8, t = 8, b = 18;
  const x = (i: number) => l + ((i - 1) / (BATCHES - 1)) * (w - l - r);
  const y = (v: number) => t + (1 - (v - 0.2) / 0.8) * (h - t - b);
  const truePts = stats.map((s) => `${x(s.t)},${y(s.acc)}`).join(" ");
  const measPts = measured.map((m) => `${x(m.t)},${y(m.acc)}`).join(" ");
  return (
    <figure className="mt-2">
    <svg viewBox={`0 0 ${w} ${h}`} className="h-auto w-full" role="img" aria-label="Accuracy over batches: true versus measured">
      {[0.2, 0.6, 1].map((v) => <text key={v} x={l - 4} y={y(v) + 3} textAnchor="end" fontSize={9} fill="var(--muted)" fontFamily="var(--font-mono)">{v}</text>)}
      <line x1={l} x2={w - r} y1={y(thr)} y2={y(thr)} stroke="var(--err)" strokeDasharray="4 3" />
      <text x={w - r} y={y(thr) - 3} textAnchor="end" fontSize={9} fill="var(--err)">alert {Math.round(thr * 100)}%</text>
      <polyline points={truePts} fill="none" stroke="var(--muted)" strokeWidth={1.5} strokeDasharray="3 3" />
      {stats.map((s) => <circle key={s.t} cx={x(s.t)} cy={y(s.acc)} r={2} fill="var(--muted)" />)}
      <polyline points={measPts} fill="none" stroke="var(--accent)" strokeWidth={2} />
      {measured.map((m, i) => (
        <motion.circle key={i === measured.length - 1 ? `${animKey}-${m.t}` : m.t} cx={x(m.t)} cy={y(m.acc)} r={3.5} fill={m.acc < thr ? "var(--err)" : "var(--accent)"}
          initial={i === measured.length - 1 ? { scale: 0 } : false} animate={{ scale: 1 }} transition={{ delay: T.labels * k + 0.3 * k, duration: 0.3 * k }} />
      ))}
      {Array.from({ length: BATCHES }, (_, i) => <text key={i} x={x(i + 1)} y={h - 4} textAnchor="middle" fontSize={9} fill="var(--muted)">{i + 1}</text>)}
    </svg>
    <figcaption className="flex flex-wrap gap-3 text-[10px] text-muted">
      <span className="flex items-center gap-1"><svg width="16" height="4" aria-hidden><line x1="0" x2="16" y1="2" y2="2" stroke="var(--muted)" strokeDasharray="3 3" /></svg>true accuracy (unknown in production)</span>
      <span className="flex items-center gap-1"><svg width="16" height="4" aria-hidden><line x1="0" x2="16" y1="2" y2="2" stroke="var(--accent)" strokeWidth="2" /></svg>measured, once labels arrive</span>
    </figcaption>
    </figure>
  );
}

function ShareBars({ expected: e, cur: a, label, k }: { expected: number[]; cur: number[]; label: string; k: number }) {
  const max = Math.max(...e, ...a, 0.01);
  return (
    <div className="min-w-0">
      <div className="mb-1 text-xs font-semibold">{label}: share of rows per reference-quintile bin</div>
      <div className="flex h-24 items-end gap-2 rounded bg-surface-2 p-1.5">
        {e.map((ev, i) => (
          <div key={i} className="flex h-full flex-1 items-end gap-0.5">
            <div className="flex-1 rounded-t bg-muted/40" style={{ height: `${(ev / max) * 100}%` }} title={`reference ${(ev * 100).toFixed(0)}%`} />
            <motion.div className="flex-1 rounded-t bg-accent" initial={{ height: `${(ev / max) * 100}%` }} animate={{ height: `${(a[i] / max) * 100}%` }} transition={{ delay: (0.3 + i * 0.12) * k, duration: 0.6 * k, ease }} title={`now ${(a[i] * 100).toFixed(0)}%`} />
          </div>
        ))}
      </div>
      <div className="mt-1 flex gap-2 font-mono text-[10px] text-muted">{a.map((v, i) => <span key={i} className="flex-1 text-center">{Math.round(v * 100)}%</span>)}</div>
    </div>
  );
}

function ReferenceCard({ run, k }: { run: DriftRun; k: number }) {
  return (
    <div className="grid gap-4 sm:grid-cols-2">
      <motion.div initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.2 * k, duration: 0.5 * k }} className="space-y-1.5 text-sm">
        <p><strong>Training data</strong>: {N_REF} points around (0, 0). Labels come from the green curve.</p>
        <p>The model is a straight line, so it can only approximate the curve. Near the data that costs little: reference accuracy is <strong className="font-mono">{(run.refAcc * 100).toFixed(1)}%</strong>.</p>
        <p className="text-muted">The reference data is split into {BINS} equal-sized bins per feature. Every production batch will be compared against those bins.</p>
      </motion.div>
      <ShareBars expected={run.refShares[0]} cur={run.refShares[0]} label="x1 (reference)" k={k} />
    </div>
  );
}

function BatchCard({ run, s, prev, k }: { run: DriftRun; s: DriftState; prev?: DriftState; k: number }) {
  const b = s.stats.at(-1)!;
  const j = b.psi[0] >= b.psi[1] ? 0 : 1;
  const flipped = prev && prev.phase === "batch" ? s.points.filter((p, i) => p.y !== prev.points[i].y).length : 0;
  const terms = run.refShares[j].map((e, i) => { const a = Math.max(b.shares[j][i], 1e-4), ee = Math.max(e, 1e-4); return (a - ee) * Math.log(a / ee); });
  return (
    <div className="grid gap-4 sm:grid-cols-2">
      <div>
        <ShareBars expected={run.refShares[j]} cur={b.shares[j]} label={`x${j + 1} (grey = reference, blue = batch ${b.t})`} k={k} />
        <div className="mt-2 flex flex-wrap gap-1 font-mono text-[10px]">
          {terms.map((v, i) => (
            <motion.span key={i} initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: (T.psi + i * 0.12) * k }} className="rounded bg-surface-2 px-1">bin{i + 1}: {v.toFixed(3)}</motion.span>
          ))}
          <motion.span initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: (T.psi + 0.7) * k }} className="rounded bg-accent-soft px-1 font-semibold">PSI = {b.psi[j].toFixed(3)}</motion.span>
        </div>
      </div>
      <div className="space-y-2 text-sm">
        <Row label="inputs moved by" value={`(${b.shift[0].toFixed(2)}, ${b.shift[1].toFixed(2)})`} hint="covariate shift: P(x)" />
        <Row label="true boundary rotated" value={`${b.angle.toFixed(1)}°`} hint="concept drift: P(y|x)" />
        <Row label="labels flipped since last batch" value={String(flipped)} hint="same place, new answer" />
        <Row label="model wrong on" value={`${s.points.filter((p) => p.pred !== p.y).length} / ${N_BATCH}`} hint={`true accuracy ${(b.acc * 100).toFixed(1)}%`} />
        <p className="text-xs text-muted">{b.angle > 0 && b.shift[0] === 0 ? "The points barely move, so their histograms match the reference: PSI cannot see this." : b.shift[0] > 0 ? "The histogram slides right: PSI and KS react immediately, before any label exists." : "Nothing changed: small PSI values are just sampling noise."}</p>
      </div>
    </div>
  );
}

function Row({ label, value, hint }: { label: string; value: string; hint: string }) {
  return (
    <div className="flex items-baseline justify-between gap-2 border-b border-line pb-1 text-xs">
      <span>{label} <span className="text-[10px] text-muted">· {hint}</span></span>
      <span className="font-mono font-semibold">{value}</span>
    </div>
  );
}

function SummaryCard({ run, input, k }: { run: DriftRun; input: DriftInput; k: number }) {
  const stats = run.steps.at(-1)!.state.stats;
  const cell = (on: boolean, i: number, row: number) => (
    <motion.div key={i} initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ delay: (0.2 + row * 0.5 + i * 0.06) * k, duration: 0.25 * k }}
      className={cx("flex h-6 items-center justify-center rounded text-[10px] font-semibold", on ? "bg-err-soft text-err" : "bg-surface-2 text-muted")}>{on ? "!" : "·"}</motion.div>
  );
  return (
    <div className="space-y-2 overflow-x-auto">
      <div className="grid min-w-[420px] grid-cols-[130px_repeat(8,minmax(0,1fr))] gap-1 text-xs">
        <span />{stats.map((s) => <span key={s.t} className="text-center text-muted">batch {s.t}</span>)}
        <span className="font-medium">PSI ≥ {input.psiThreshold}</span>{stats.map((s, i) => cell(Math.max(...s.psi) >= input.psiThreshold, i, 0))}
        <span className="font-medium">true accuracy low</span>{stats.map((s, i) => cell(s.acc < input.accThreshold, i, 1))}
        <span className="font-medium">label monitor knows</span>{stats.map((s, i) => cell(run.firstLabelAlert !== undefined && s.t >= run.firstLabelAlert, i, 2))}
      </div>
      <p className="text-xs text-muted">
        {input.kind === "concept" ? "Concept drift: the inputs look identical, so the input test is blind. Accuracy is already low before the label monitor can know it: that gap is the label delay."
          : input.kind === "covariate" ? "Covariate shift: the input test fires as soon as the data moves, even before accuracy falls. That is an early warning, not proof of harm: the model only fails where it has to extrapolate."
          : input.kind === "both" ? "Both: the input test fires early for the shift; the label monitor confirms the accuracy loss later."
          : "No drift: every detector stays quiet. Small PSI values are sampling noise."}
      </p>
    </div>
  );
}

function DetectorMath({ run, b }: { run: DriftRun; b: BatchStats }) {
  const j = b.psi[0] >= b.psi[1] ? 0 : 1;
  const e = run.refShares[j], a = b.shares[j];
  const terms = e.slice(0, 1).map((ev, i) => `(${a[i].toFixed(2)} - ${ev.toFixed(2)})\\ln\\frac{${a[i] < 0.005 ? "10^{-4}" : a[i].toFixed(2)}}{${ev.toFixed(2)}}`).join(" + ");
  const crit = ksCritical(N_REF, N_BATCH);
  return (
    <div className="space-y-2">
      <TexBlock>{`\\begin{aligned} \\text{PSI}_{x_${j + 1}} &= \\textstyle\\sum_b (a_b - e_b)\\ln\\frac{a_b}{e_b} \\\\ &= ${terms} + \\dots \\\\ &= ${b.psi[j].toFixed(3)} \\end{aligned}`}</TexBlock>
      <TexBlock>{`D_{x_${j + 1}} = \\max_x \\lvert F_{\\text{ref}}(x) - F_{\\text{now}}(x) \\rvert = ${b.ks[j].toFixed(3)} ${b.ks[j] > crit ? ">" : "\\le"} ${crit.toFixed(3)}`}</TexBlock>
      <p className="text-[11px] text-muted">e = reference share per bin (0.20 each), a = this batch's share. The accuracy monitor needs ground-truth labels, which arrive later.</p>
    </div>
  );
}
