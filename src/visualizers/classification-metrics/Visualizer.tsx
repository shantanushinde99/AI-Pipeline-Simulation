import { useMemo, useRef, useState, type PointerEvent as RPointerEvent } from "react";
import { AnimatePresence, motion } from "motion/react";
import { RotateCcw } from "lucide-react";
import { usePlayback } from "../../hooks/usePlayback";
import { PlaybackControls } from "../../components/visualizer/PlaybackControls";
import { EventLog } from "../../components/visualizer/EventLog";
import { StepExplainer, Workspace } from "../../components/visualizer/Workspace";
import { AnimatedNumber } from "../../components/visualizer/AnimatedNumber";
import { Button, Callout, KV, Panel, SimulatedLabel, Slider, cx } from "../../components/ui";
import { Tex, TexBlock } from "../../components/Tex";
import { clamp } from "../../lib/math";
import type { VisualizerTab } from "../types";
import { MetricBar, StepCaption, ease, pct } from "./kit";
import { DEFAULT_CLS, simulate, type ClsConfig, type ClsPhase, type ClsSim, type Example } from "./simulation";

/* ---------- one timing schedule (seconds at 1×) ---------- */
const DROP = (i: number, n: number) => 0.2 + i * Math.min(0.05, 2 / n);
const SPLIT = 0.7;
const SWEEP = 3;
const DURATIONS: Record<ClsPhase, number> = { scores: 3800, threshold: 3400, count: 3400, precision: 3600, recall: 3600, summary: 3400, roc: 5200, pr: 5200, baseline: 3800 };

const PRESETS: { label: string; cfg: Partial<ClsConfig> }[] = [
  { label: "Balanced (50% positive)", cfg: { prevalence: 0.5, separation: 2.5, n: 40, threshold: 0.5 } },
  { label: "Imbalanced (8% positive)", cfg: { prevalence: 0.08, separation: 2.5, n: 50, threshold: 0.5 } },
  { label: "Weak model (overlapping)", cfg: { prevalence: 0.5, separation: 0.6, n: 40, threshold: 0.5 } },
  { label: "Break it: always say negative", cfg: { threshold: 1.001 } },
];

const NARRATE: Record<ClsPhase, (s: ClsSim) => { headline: string; sub: string }> = {
  scores: (s) => ({ headline: `${s.ex.length} examples land on the score line`, sub: "Filled dots are truly positive, hollow dots truly negative. The model only gave each a score." }),
  threshold: (s) => ({ headline: `Everything right of ${s.t.toFixed(2)} jumps up to "predicted positive"`, sub: "The threshold turns scores into decisions. Drag the line and watch dots flip." }),
  count: (s) => ({ headline: `TP ${s.m.tp} · FP ${s.m.fp} · FN ${s.m.fn} · TN ${s.m.tn}`, sub: "Top lane: filled = true positive, hollow = false alarm. Bottom lane: filled = missed, hollow = correctly ignored." }),
  precision: (s) => ({ headline: `Precision ${pct(s.m.precision)}: of the dots flagged positive, how many were real?`, sub: "Only the top lane matters here. Raise the threshold to make alarms rarer and usually more trustworthy." }),
  recall: (s) => ({ headline: `Recall ${pct(s.m.recall)}: of all truly positive dots, how many were caught?`, sub: "Only the filled dots matter here. Ringed ones are the misses (false negatives)." }),
  summary: (s) => ({ headline: `F1 ${pct(s.m.f1)}, accuracy ${pct(s.m.accuracy)}, specificity ${pct(s.m.specificity)}`, sub: "Each metric answers a different question about the same four counts." }),
  roc: (s) => ({ headline: `Sliding the threshold from 1 to 0 traces the ROC curve (AUC ${s.auc.toFixed(3)})`, sub: "Each threshold is one point: false-positive rate across, true-positive rate up." }),
  pr: (s) => ({ headline: `The same sweep as precision vs recall (AP ${s.ap.toFixed(3)})`, sub: `A coin-flip model sits on the dashed line at the prevalence, ${s.prevalence.toFixed(2)}. Under imbalance this curve is the honest one.` }),
  baseline: (s) => ({
    headline: s.allNeg.accuracy >= (s.m.accuracy || 0) - 0.03 ? `Warning: saying "negative" to everyone already scores ${pct(s.allNeg.accuracy)} accuracy` : `Your threshold beats the always-negative baseline (${pct(s.allNeg.accuracy)})`,
    sub: "Accuracy rewards the majority class. Always compare against the dumbest possible model.",
  }),
};

const WHY: Record<ClsPhase, string> = {
  scores: "Most classifiers output a score, not a decision. Every metric below depends on where you cut.",
  threshold: "The threshold is a business decision: it trades false alarms for misses.",
  count: "All of precision, recall, F1, accuracy and specificity are ratios of these four numbers.",
  precision: "High precision matters when false alarms are expensive (spam filter deleting real mail).",
  recall: "High recall matters when misses are expensive (cancer screening, fraud).",
  summary: "F1 is the harmonic mean of precision and recall; it ignores true negatives entirely.",
  roc: "AUC is threshold-free: it measures ranking quality, not the decision you will ship.",
  pr: "When positives are rare, ROC can look great while precision is poor. PR exposes that.",
  baseline: "A metric is only meaningful relative to a baseline.",
};

export default function ClassificationMetricsVisualizer({ tab }: { tab: VisualizerTab }) {
  const [cfg, setCfg] = useState(DEFAULT_CLS);
  const [replay, setReplay] = useState(0);
  const set = (p: Partial<ClsConfig>) => setCfg((c) => ({ ...c, ...p }));
  const sim = useMemo(() => simulate(cfg), [cfg]);
  const pb = usePlayback(sim.steps.length, (i) => DURATIONS[sim.steps[i]?.state.phase ?? "scores"]);
  const step = sim.steps[pb.index];
  const phase = step.state.phase;
  const k = 1 / pb.speed;
  const animKey = `${step.id}-${replay}`;
  if (tab === "metrics") return null;
  const story = NARRATE[phase](sim);

  return (
    <Workspace
      main={
        <>
          <Panel title="Scores, threshold and curves" actions={<SimulatedLabel>Synthetic scores</SimulatedLabel>}>
            <StepCaption index={pb.index} total={sim.steps.length} stepId={step.id} headline={story.headline} sub={story.sub} onReplay={() => setReplay((r) => r + 1)} />
            <div className="overflow-x-auto">
              <div className="min-w-[560px]">
                <ScoreStrip sim={sim} phase={phase} k={k} animKey={animKey} replay={replay} onThreshold={(t) => set({ threshold: t })} />
              </div>
            </div>
            <div className="mt-2 grid gap-3 sm:grid-cols-2">
              <RocChart sim={sim} phase={phase} k={k} animKey={animKey} />
              <PrChart sim={sim} phase={phase} k={k} animKey={animKey} />
            </div>
            <div className="mt-2 flex flex-wrap gap-3 text-[11px] text-muted">
              <span className="flex items-center gap-1"><svg width="10" height="10"><circle cx="5" cy="5" r="4" fill="var(--accent)" /></svg>truly positive</span>
              <span className="flex items-center gap-1"><svg width="10" height="10"><circle cx="5" cy="5" r="3.5" fill="var(--surface)" stroke="var(--violet)" strokeWidth="1.5" /></svg>truly negative</span>
              <span>drag the threshold handle (or use the slider) to move the operating point</span>
            </div>
          </Panel>
          <Panel title="What's happening in this step">
            <AnimatePresence mode="wait" initial={false}>
              <motion.div key={animKey} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: 0.2 }}>
                <StepCard sim={sim} phase={phase} k={k} />
              </motion.div>
            </AnimatePresence>
          </Panel>
          <PlaybackControls pb={pb} stepTitles={sim.steps.map((s) => s.title)} />
          <Panel title="Parameters" actions={<Button size="sm" variant="ghost" onClick={() => setCfg(DEFAULT_CLS)}><RotateCcw size={12} /> Reset</Button>}>
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              <Slider label="Threshold" value={Math.min(cfg.threshold, 1)} min={0} max={1} step={0.01} format={(v) => (cfg.threshold > 1 ? "> every score" : v.toFixed(2))} onChange={(v) => set({ threshold: v })} />
              <Slider label="Class separation (model quality)" value={cfg.separation} min={0} max={5} step={0.1} format={(v) => v.toFixed(1)} onChange={(v) => set({ separation: v })} hint="0 = scores carry no information." />
              <Slider label="Share of positives" value={cfg.prevalence} min={0.04} max={0.9} step={0.02} format={(v) => `${Math.round(v * 100)}%`} onChange={(v) => set({ prevalence: v })} />
              <Slider label="Examples" value={cfg.n} min={10} max={80} step={2} onChange={(v) => set({ n: v })} />
              <Slider label="Random seed" value={cfg.seed} min={1} max={40} onChange={(v) => set({ seed: v })} hint="Same seed → same scores." />
            </div>
            <div className="mt-3 flex flex-wrap gap-1.5">
              {PRESETS.map((p) => <Button key={p.label} size="sm" onClick={() => set(p.cfg)}>{p.label}</Button>)}
            </div>
          </Panel>
        </>
      }
      inspector={
        <>
          <StepExplainer index={pb.index} title={step.title} what={step.description} why={WHY[phase]} />
          <Panel title="Calculation"><Calculation sim={sim} phase={phase} /></Panel>
          <Panel title="At this threshold">
            <KV rows={[
              ["threshold", sim.t > 1 ? "> every score" : sim.t.toFixed(2)],
              ["TP / FP / FN / TN", `${sim.m.tp} / ${sim.m.fp} / ${sim.m.fn} / ${sim.m.tn}`],
              ["precision", pct(sim.m.precision)],
              ["recall (TPR)", pct(sim.m.recall)],
              ["F1", pct(sim.m.f1)],
              ["accuracy", pct(sim.m.accuracy)],
              ["specificity (TNR)", pct(sim.m.specificity)],
              ["ROC AUC", sim.auc.toFixed(3)],
              ["average precision", sim.ap.toFixed(3)],
            ]} />
          </Panel>
          <Panel title="Events"><EventLog events={step.events} /></Panel>
        </>
      }
    />
  );
}

/* ---------- score strip ---------- */
const W = 680, H = 250, X0 = 40, X1 = 650, TOP = 70, BOT = 165, MID = 118, AXIS = 225, R = 5;
const sx = (s: number) => X0 + clamp(s, 0, 1) * (X1 - X0);

/** Beeswarm-lite: each dot takes the nearest free vertical slot in its lane so close scores do not overlap. */
function layout(ex: readonly Example[], laneOf: (e: Example) => number) {
  const placed: { x: number; lane: number; level: number }[] = [];
  const out = new Map<number, number>();
  for (const e of [...ex].sort((a, b) => a.score - b.score || a.id - b.id)) {
    const x = sx(e.score), lane = laneOf(e);
    let level = 0;
    for (const cand of [0, 1, -1, 2, -2, 3, -3, 4, -4]) {
      level = cand;
      if (!placed.some((p) => p.lane === lane && p.level === cand && Math.abs(p.x - x) < 2 * R + 1)) break;
    }
    placed.push({ x, lane, level });
    out.set(e.id, lane + level * (2 * R + 1));
  }
  return out;
}

function ScoreStrip({ sim, phase, k, animKey, replay, onThreshold }: { sim: ClsSim; phase: ClsPhase; k: number; animKey: string; replay: number; onThreshold: (t: number) => void }) {
  const ref = useRef<SVGSVGElement>(null);
  const split = phase !== "scores";
  const tx = sx(Math.min(sim.t, 1));
  const ys = useMemo(() => layout(sim.ex, (e) => (!split ? MID : e.score >= sim.t ? TOP : BOT)), [sim, split]);
  const rank = useMemo(() => new Map([...sim.ex].sort((a, b) => b.score - a.score).map((e, i) => [e.id, i])), [sim.ex]);
  const drag = (e: RPointerEvent<SVGElement>) => {
    const box = ref.current?.getBoundingClientRect();
    if (!box) return;
    const x = ((e.clientX - box.left) / box.width) * W;
    onThreshold(Math.round(clamp((x - X0) / (X1 - X0), 0, 1) * 100) / 100);
  };
  const dim = (e: Example) => {
    const predPos = e.score >= sim.t;
    if (phase === "precision") return predPos ? 1 : 0.15;
    if (phase === "recall") return e.y ? 1 : 0.15;
    return 1;
  };
  const groupKey = phase === "scores" ? `s-${replay}` : `t-${phase === "threshold" ? replay : 0}`;
  const sweepX = sim.roc.map((p) => sx(Math.min(1, p.t)));
  return (
    <svg ref={ref} viewBox={`0 0 ${W} ${H}`} className="h-auto w-full select-none" role="img" aria-label={`${sim.ex.length} examples placed by model score with a threshold at ${sim.t.toFixed(2)}`}>
      {split && (
        <motion.g initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.4 * k }}>
          <motion.rect y={TOP - 46} height={92} fill="var(--accent)" fillOpacity={0.06} initial={false} animate={{ x: tx, width: Math.max(0, X1 + 10 - tx) }} transition={{ type: "spring", stiffness: 260, damping: 30 }} />
          <motion.rect y={BOT - 46} height={92} x={X0 - 10} fill="var(--violet)" fillOpacity={0.05} initial={false} animate={{ width: Math.max(0, tx - X0 + 10) }} transition={{ type: "spring", stiffness: 260, damping: 30 }} />
          <text x={X1 + 8} y={TOP - 34} textAnchor="end" fontSize={11} fill="var(--accent)" fontWeight={600}>▲ predicted positive</text>
          <text x={X0 - 8} y={BOT + 52} fontSize={11} fill="var(--violet)" fontWeight={600}>▼ predicted negative</text>
        </motion.g>
      )}
      {/* score axis */}
      <line x1={X0} x2={X1} y1={AXIS} y2={AXIS} stroke="var(--line)" />
      {[0, 0.25, 0.5, 0.75, 1].map((v) => (
        <g key={v}>
          <line x1={sx(v)} x2={sx(v)} y1={AXIS} y2={AXIS + 4} stroke="var(--muted)" />
          <text x={sx(v)} y={AXIS + 16} textAnchor="middle" fontSize={10} fill="var(--muted)" fontFamily="var(--font-mono)">{v}</text>
        </g>
      ))}
      <text x={X1} y={AXIS - 6} textAnchor="end" fontSize={10} fill="var(--muted)">model score →</text>

      {/* dots */}
      <g key={groupKey}>
        {sim.ex.map((e, i) => {
          const y = ys.get(e.id) ?? MID;
          const isFN = phase === "recall" && e.y === 1 && e.score < sim.t;
          const initial = phase === "scores" ? { x: sx(e.score), y: -10, opacity: 0 } : { x: sx(e.score), y: phase === "threshold" ? MID : y, opacity: 1 };
          return (
            <motion.g key={e.id} initial={initial} animate={{ x: sx(e.score), y, opacity: dim(e) }}
              transition={phase === "scores"
                ? { delay: DROP(i, sim.ex.length) * k, duration: 0.6 * k, ease }
                : { y: { type: "spring", stiffness: 170, damping: 17, delay: phase === "threshold" ? (SPLIT + (rank.get(e.id) ?? 0) * 0.02) * k : 0 }, opacity: { duration: 0.4 * k } }}>
              {e.y ? <circle r={R} fill="var(--accent)" /> : <circle r={R - 0.75} fill="var(--surface)" stroke="var(--violet)" strokeWidth={1.5} />}
              {isFN && <circle r={R + 3.5} fill="none" stroke="var(--err)" strokeWidth={1.5} />}
            </motion.g>
          );
        })}
      </g>

      {/* outcome tallies */}
      {(phase === "count" || phase === "summary") && (
        <motion.g key={`tally-${animKey}`} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 * k }} fontSize={11} fontFamily="var(--font-mono)">
          <text x={Math.min(tx + 6, X1 - 110)} y={TOP + 44} fill="var(--fg)">TP <AnimatedNumber svg value={sim.m.tp} from={0} digits={0} delay={0.2 * k} duration={0.6 * k} /> · FP <AnimatedNumber svg value={sim.m.fp} from={0} digits={0} delay={0.6 * k} duration={0.6 * k} /></text>
          <text x={Math.max(tx - 6, X0 + 110)} y={BOT - 36} textAnchor="end" fill="var(--fg)">FN <AnimatedNumber svg value={sim.m.fn} from={0} digits={0} delay={1 * k} duration={0.6 * k} /> · TN <AnimatedNumber svg value={sim.m.tn} from={0} digits={0} delay={1.4 * k} duration={0.6 * k} /></text>
        </motion.g>
      )}

      {/* ghost threshold sweeping during the ROC/PR steps */}
      {(phase === "roc" || phase === "pr") && sweepX.length > 1 && (
        <motion.line key={`ghost-${animKey}`} y1={TOP - 46} y2={AXIS} stroke="var(--warn)" strokeWidth={2} strokeDasharray="3 3"
          initial={{ x1: sweepX[0], x2: sweepX[0], opacity: 1 }} animate={{ x1: sweepX, x2: sweepX, opacity: [1, 1, 0] }}
          transition={{ duration: SWEEP * k, delay: 0.3 * k, ease: "linear", opacity: { duration: (SWEEP + 0.8) * k, delay: 0.3 * k, times: [0, 0.85, 1] } }} />
      )}

      {/* the threshold the learner controls */}
      {split && (
        <motion.g initial={phase === "threshold" ? { opacity: 0 } : false} animate={{ opacity: 1 }} transition={{ duration: 0.5 * k }}>
          <motion.g initial={false} animate={{ x: tx }} transition={{ type: "spring", stiffness: 300, damping: 30 }}>
            <line y1={TOP - 46} y2={AXIS} stroke="var(--fg)" strokeWidth={2} />
            <rect x={-34} y={2} width={68} height={18} rx={9} fill="var(--fg)" />
            <text y={15} textAnchor="middle" fontSize={11} fill="var(--surface)" fontFamily="var(--font-mono)">t = {sim.t > 1 ? ">max" : sim.t.toFixed(2)}</text>
          </motion.g>
          <rect x={tx - 16} y={0} width={32} height={AXIS} fill="transparent" className="cursor-ew-resize" style={{ touchAction: "none" }}
            role="slider" aria-label="Threshold" aria-valuemin={0} aria-valuemax={1} aria-valuenow={Math.min(1, sim.t)} tabIndex={0}
            onPointerDown={(e) => { e.currentTarget.setPointerCapture(e.pointerId); drag(e); }}
            onPointerMove={(e) => e.currentTarget.hasPointerCapture(e.pointerId) && drag(e)}
            onKeyDown={(e) => {
              if (e.key === "ArrowLeft" || e.key === "ArrowRight") { e.preventDefault(); onThreshold(Math.round(clamp(Math.min(sim.t, 1) + (e.key === "ArrowRight" ? 0.01 : -0.01), 0, 1) * 100) / 100); }
            }} />
        </motion.g>
      )}
    </svg>
  );
}

/* ---------- ROC and PR charts ---------- */
const CW = 280, CH = 240, CL = 40, CR = 268, CT = 14, CB = 206;
const px = (v: number) => CL + v * (CR - CL);
const py = (v: number) => CB - v * (CB - CT);

function Axes({ x, y }: { x: string; y: string }) {
  return (
    <>
      <rect x={CL} y={CT} width={CR - CL} height={CB - CT} fill="none" stroke="var(--line)" />
      {[0, 0.5, 1].map((v) => (
        <g key={v} fontSize={9} fill="var(--muted)" fontFamily="var(--font-mono)">
          <text x={px(v)} y={CB + 12} textAnchor="middle">{v}</text>
          <text x={CL - 5} y={py(v) + 3} textAnchor="end">{v}</text>
        </g>
      ))}
      <text x={(CL + CR) / 2} y={CH - 6} textAnchor="middle" fontSize={10} fill="var(--muted)">{x}</text>
      <text x={11} y={(CT + CB) / 2} textAnchor="middle" fontSize={10} fill="var(--muted)" transform={`rotate(-90 11 ${(CT + CB) / 2})`}>{y}</text>
    </>
  );
}

function RocChart({ sim, phase, k, animKey }: { sim: ClsSim; phase: ClsPhase; k: number; animKey: string }) {
  const pts = sim.roc.map((p) => [px(p.fpr), py(p.tpr)] as const);
  const d = pts.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(1)} ${y.toFixed(1)}`).join(" ");
  const tracing = phase === "roc";
  const times = pts.map((_, i) => i / Math.max(1, pts.length - 1));
  const op = { x: px(Number.isFinite(sim.m.fpr) ? sim.m.fpr : 0), y: py(Number.isFinite(sim.m.recall) ? sim.m.recall : 0) };
  return (
    <figure className={cx("rounded-md border border-line p-1.5", tracing && "border-accent/50")}>
      <figcaption className="flex items-baseline justify-between px-1 text-xs"><span className="font-semibold">ROC curve</span><span className="font-mono text-muted">AUC {tracing ? <AnimatedNumber key={animKey} value={sim.auc} from={0} digits={3} delay={(SWEEP + 0.3) * k} duration={0.8 * k} /> : sim.auc.toFixed(3)}</span></figcaption>
      <svg viewBox={`0 0 ${CW} ${CH}`} className="h-auto w-full" role="img" aria-label={`ROC curve with AUC ${sim.auc.toFixed(3)}`}>
        <Axes x="false-positive rate (FPR)" y="true-positive rate (recall)" />
        <line x1={px(0)} y1={py(0)} x2={px(1)} y2={py(1)} stroke="var(--muted)" strokeDasharray="4 4" />
        {tracing && <motion.path key={`area-${animKey}`} d={`${d} L${px(1)} ${py(0)} Z`} fill="var(--accent)" initial={{ fillOpacity: 0 }} animate={{ fillOpacity: 0.15 }} transition={{ delay: (SWEEP + 0.3) * k, duration: 0.6 * k }} />}
        <path d={d} fill="none" stroke="var(--accent)" strokeOpacity={tracing ? 0.15 : phase === "pr" ? 0.35 : 0.6} strokeWidth={2} />
        {tracing && <motion.path key={`trace-${animKey}`} d={d} fill="none" stroke="var(--accent)" strokeWidth={2.5} initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ delay: 0.3 * k, duration: SWEEP * k, ease: "linear" }} />}
        {tracing && pts.length > 1 && (
          <motion.circle key={`tr-${animKey}`} r={5} fill="var(--warn)" initial={{ cx: pts[0][0], cy: pts[0][1] }} animate={{ cx: pts.map((p) => p[0]), cy: pts.map((p) => p[1]) }}
            transition={{ delay: 0.3 * k, duration: SWEEP * k, ease: "linear", times }} />
        )}
        <motion.g initial={false} animate={{ x: op.x, y: op.y }} transition={{ type: "spring", stiffness: 220, damping: 24 }}>
          <circle r={6} fill="var(--fg)" stroke="var(--surface)" strokeWidth={2} />
          <text x={op.x > CR - 60 ? -9 : 9} y={op.y < CT + 20 ? 14 : -8} textAnchor={op.x > CR - 60 ? "end" : "start"} fontSize={10} fill="var(--fg)" stroke="var(--surface)" strokeWidth={3} paintOrder="stroke" fontFamily="var(--font-mono)">you (t={sim.t > 1 ? ">max" : sim.t.toFixed(2)})</text>
        </motion.g>
      </svg>
    </figure>
  );
}

function PrChart({ sim, phase, k, animKey }: { sim: ClsSim; phase: ClsPhase; k: number; animKey: string }) {
  const pts = sim.pr.map((p) => [px(p.recall), py(Number.isFinite(p.precision) ? p.precision : 1)] as const);
  const d = pts.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(1)} ${y.toFixed(1)}`).join(" ");
  const tracing = phase === "pr";
  const times = pts.map((_, i) => i / Math.max(1, pts.length - 1));
  const has = Number.isFinite(sim.m.precision);
  const op = { x: px(Number.isFinite(sim.m.recall) ? sim.m.recall : 0), y: py(has ? sim.m.precision : 1) };
  return (
    <figure className={cx("rounded-md border border-line p-1.5", tracing && "border-accent/50")}>
      <figcaption className="flex items-baseline justify-between px-1 text-xs"><span className="font-semibold">Precision-recall curve</span><span className="font-mono text-muted">AP {tracing ? <AnimatedNumber key={animKey} value={sim.ap} from={0} digits={3} delay={(SWEEP + 0.3) * k} duration={0.8 * k} /> : sim.ap.toFixed(3)}</span></figcaption>
      <svg viewBox={`0 0 ${CW} ${CH}`} className="h-auto w-full" role="img" aria-label={`Precision-recall curve with average precision ${sim.ap.toFixed(3)}`}>
        <Axes x="recall" y="precision" />
        <line x1={px(0)} x2={px(1)} y1={py(sim.prevalence)} y2={py(sim.prevalence)} stroke="var(--muted)" strokeDasharray="4 4" />
        <text x={px(1) - 3} y={py(sim.prevalence) - 4} textAnchor="end" fontSize={9} fill="var(--muted)">random = prevalence {sim.prevalence.toFixed(2)}</text>
        <path d={d} fill="none" stroke="var(--violet)" strokeOpacity={tracing ? 0.15 : 0.6} strokeWidth={2} />
        {tracing && <motion.path key={`trace-${animKey}`} d={d} fill="none" stroke="var(--violet)" strokeWidth={2.5} initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ delay: 0.3 * k, duration: SWEEP * k, ease: "linear" }} />}
        {tracing && pts.length > 1 && (
          <motion.circle key={`tr-${animKey}`} r={5} fill="var(--warn)" initial={{ cx: pts[0][0], cy: pts[0][1] }} animate={{ cx: pts.map((p) => p[0]), cy: pts.map((p) => p[1]) }}
            transition={{ delay: 0.3 * k, duration: SWEEP * k, ease: "linear", times }} />
        )}
        <motion.g initial={false} animate={{ x: op.x, y: op.y, opacity: has ? 1 : 0.3 }} transition={{ type: "spring", stiffness: 220, damping: 24 }}>
          <circle r={6} fill="var(--fg)" stroke="var(--surface)" strokeWidth={2} />
          <text x={op.x > CR - 60 ? -9 : 9} y={op.y < CT + 20 ? 14 : -8} textAnchor={op.x > CR - 60 ? "end" : "start"} fontSize={10} fill="var(--fg)" stroke="var(--surface)" strokeWidth={3} paintOrder="stroke" fontFamily="var(--font-mono)">{has ? "you" : "you (precision undefined)"}</text>
        </motion.g>
      </svg>
    </figure>
  );
}

/* ---------- step card ---------- */
function Blocks({ a, b, aLabel, bLabel, k }: { a: number; b: number; aLabel: string; bLabel: string; k: number }) {
  const total = a + b;
  const gap = Math.min(0.1, 1.6 / Math.max(1, total));
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap gap-1">
        {Array.from({ length: total }, (_, i) => (
          <motion.span key={i} className={cx("h-4 w-4 rounded-sm", i < a ? "bg-accent" : "border-2 border-violet bg-surface")}
            initial={{ scale: 0, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ delay: (0.2 + i * gap) * k, duration: 0.3 * k, ease }} />
        ))}
        {total === 0 && <span className="text-xs text-muted">No examples in this group.</span>}
      </div>
      <div className="flex flex-wrap gap-3 text-[11px] text-muted">
        <span className="flex items-center gap-1"><span className="h-3 w-3 rounded-sm bg-accent" />{aLabel}: {a}</span>
        <span className="flex items-center gap-1"><span className="h-3 w-3 rounded-sm border-2 border-violet" />{bLabel}: {b}</span>
      </div>
    </div>
  );
}

function StepCard({ sim, phase, k }: { sim: ClsSim; phase: ClsPhase; k: number }) {
  const { m } = sim;
  const end = (n: number) => 0.2 + n * Math.min(0.1, 1.6 / Math.max(1, n));
  if (phase === "scores") {
    const pos = m.tp + m.fn;
    return (
      <div className="space-y-2 text-sm">
        <MetricBar label="positives" value={sim.prevalence} k={k} delay={0.3} note={`${pos} of ${sim.ex.length}`} />
        <MetricBar label="negatives" value={1 - sim.prevalence} k={k} delay={0.6} tone="violet" note={`${sim.ex.length - pos} of ${sim.ex.length}`} />
        <p className="text-xs text-muted">A score is the model's confidence that an example is positive. A perfect model would put every filled dot to the right of every hollow one.</p>
      </div>
    );
  }
  if (phase === "threshold") {
    const pp = (m.tp + m.fp) / m.n;
    return (
      <div className="space-y-2">
        <MetricBar label="predicted +" value={pp} k={k} delay={SPLIT} note={`${m.tp + m.fp} examples`} />
        <MetricBar label="predicted −" value={1 - pp} k={k} delay={SPLIT + 0.3} tone="violet" note={`${m.tn + m.fn} examples`} />
        <p className="text-xs text-muted">Rule: <span className="font-mono">score ≥ {sim.t > 1 ? "max" : sim.t.toFixed(2)} → positive</span>. Moving the threshold changes which side each dot falls on, and therefore every metric that follows.</p>
      </div>
    );
  }
  if (phase === "count") {
    const cells = [
      { l: "TP", v: m.tp, d: "filled, top lane", tone: "bg-ok-soft" },
      { l: "FN", v: m.fn, d: "filled, bottom lane", tone: "bg-err-soft" },
      { l: "FP", v: m.fp, d: "hollow, top lane", tone: "bg-warn-soft" },
      { l: "TN", v: m.tn, d: "hollow, bottom lane", tone: "bg-surface-2" },
    ];
    const delays = [0.2, 1, 0.6, 1.4];
    return (
      <div className="grid gap-3 sm:grid-cols-[auto_minmax(0,1fr)]">
        <div className="grid grid-cols-[64px_repeat(2,92px)] gap-1 text-center text-xs">
          <span />
          <span className="text-muted">predicted +</span>
          <span className="text-muted">predicted −</span>
          {[0, 1].map((row) => (
            <div key={row} className="contents">
              <span className="self-center text-right text-muted">{row ? "actual −" : "actual +"}</span>
              {[row ? 2 : 0, row ? 3 : 1].map((ci) => (
                <motion.div key={ci} className={cx("rounded-md border border-line p-2", cells[ci].tone)} initial={{ scale: 0.8, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ delay: delays[ci] * k, duration: 0.35 * k, ease }}>
                  <div className="font-semibold">{cells[ci].l}</div>
                  <div className="font-mono text-lg"><AnimatedNumber value={cells[ci].v} from={0} digits={0} delay={delays[ci] * k} duration={0.6 * k} /></div>
                </motion.div>
              ))}
            </div>
          ))}
        </div>
        <ul className="space-y-1 text-xs text-muted">
          {cells.map((c) => <li key={c.l}><strong className="text-fg">{c.l}</strong>: {c.d}</li>)}
        </ul>
      </div>
    );
  }
  if (phase === "precision" || phase === "recall") {
    const isP = phase === "precision";
    const a = m.tp, b = isP ? m.fp : m.fn;
    const v = isP ? m.precision : m.recall;
    return (
      <div className="space-y-3">
        <Blocks a={a} b={b} aLabel="TP" bLabel={isP ? "FP (false alarm)" : "FN (missed)"} k={k} />
        <div className="flex items-baseline gap-2 font-mono text-sm">
          <span>{isP ? "precision" : "recall"} = {a} / ({a} + {b}) =</span>
          <span className="text-lg font-semibold">{Number.isFinite(v) ? <AnimatedNumber value={v} from={0} digits={3} delay={end(a + b) * k} duration={0.6 * k} /> : "undefined (0/0)"}</span>
        </div>
        {isP && !Number.isFinite(v) && <Callout tone="warn">Nothing is predicted positive, so precision is 0/0. Libraries often report 0 here with a warning.</Callout>}
      </div>
    );
  }
  if (phase === "summary") {
    return (
      <div className="space-y-1.5">
        <MetricBar label="accuracy" value={m.accuracy} k={k} delay={0.2} />
        <MetricBar label="precision" value={m.precision} k={k} delay={0.55} tone="ok" />
        <MetricBar label="recall" value={m.recall} k={k} delay={0.9} tone="ok" />
        <MetricBar label="F1" value={m.f1} k={k} delay={1.25} tone="violet" />
        <MetricBar label="specificity" value={m.specificity} k={k} delay={1.6} tone="warn" />
        <p className="pt-1 text-xs text-muted">Accuracy counts TN; F1 does not. With many easy negatives, accuracy can be high while F1 is poor.</p>
      </div>
    );
  }
  if (phase === "roc" || phase === "pr") {
    const roc = phase === "roc";
    const v = roc ? sim.auc : sim.ap;
    const base = roc ? 0.5 : sim.prevalence;
    return (
      <div className="space-y-2">
        <MetricBar label={roc ? "ROC AUC" : "avg precision"} value={v} k={k} delay={SWEEP + 0.3} tone={roc ? "accent" : "violet"} />
        <MetricBar label="random model" value={base} k={k} delay={SWEEP + 0.6} tone="warn" />
        <p className="text-xs text-muted">
          {roc ? `Each of the ${sim.roc.length} thresholds gives one (FPR, TPR) point; the orange dot visits them in order while the dashed line sweeps the scores above. ` : `Precision is undefined until something is predicted positive, so the sweep starts at the highest score. `}
          The black dot is your current threshold; drag it and it slides along the curve.
        </p>
      </div>
    );
  }
  return (
    <div className="space-y-2">
      <MetricBar label="your accuracy" value={m.accuracy} k={k} delay={0.2} />
      <MetricBar label="always-negative" value={sim.allNeg.accuracy} k={k} delay={0.55} tone="warn" />
      <MetricBar label="your recall" value={m.recall} k={k} delay={0.9} tone="ok" />
      <MetricBar label="always-neg recall" value={0} k={k} delay={1.25} tone="warn" />
      {sim.prevalence < 0.2 && <Callout tone="warn" title="Class imbalance">Only {pct(sim.prevalence, 0)} of examples are positive, so accuracy is dominated by negatives. Judge this model by recall, precision and the PR curve.</Callout>}
    </div>
  );
}

function Calculation({ sim, phase }: { sim: ClsSim; phase: ClsPhase }) {
  const { m } = sim;
  const f = (v: number) => (Number.isFinite(v) ? v.toFixed(3) : "\\text{undefined}");
  const tex: Record<ClsPhase, string> = {
    scores: `\\text{prevalence} = \\frac{\\#\\text{positives}}{n} = \\frac{${m.tp + m.fn}}{${m.n}} = ${sim.prevalence.toFixed(3)}`,
    threshold: `\\hat{y}_i = \\begin{cases} 1 & s_i \\ge ${sim.t > 1 ? "\\max s" : sim.t.toFixed(2)} \\\\ 0 & \\text{otherwise} \\end{cases}`,
    count: `\\begin{aligned} TP &= ${m.tp} & FP &= ${m.fp} \\\\ FN &= ${m.fn} & TN &= ${m.tn} \\end{aligned}`,
    precision: `\\text{precision} = \\frac{TP}{TP + FP} = \\frac{${m.tp}}{${m.tp} + ${m.fp}} = ${f(m.precision)}`,
    recall: `\\text{recall} = \\frac{TP}{TP + FN} = \\frac{${m.tp}}{${m.tp} + ${m.fn}} = ${f(m.recall)}`,
    summary: `\\begin{aligned} F_1 &= \\frac{2\\,TP}{2\\,TP + FP + FN} = \\frac{${2 * m.tp}}{${2 * m.tp + m.fp + m.fn}} = ${f(m.f1)} \\\\ \\text{acc} &= \\frac{TP + TN}{n} = \\frac{${m.tp + m.tn}}{${m.n}} = ${f(m.accuracy)} \\\\ \\text{spec} &= \\frac{TN}{TN + FP} = \\frac{${m.tn}}{${m.tn + m.fp}} = ${f(m.specificity)} \\end{aligned}`,
    roc: `\\begin{aligned} FPR &= \\frac{FP}{FP + TN},\\; TPR = \\frac{TP}{TP + FN} \\\\ \\text{AUC} &= \\sum_k (FPR_k - FPR_{k-1})\\frac{TPR_k + TPR_{k-1}}{2} = ${sim.auc.toFixed(3)} \\end{aligned}`,
    pr: `AP = \\sum_k (R_k - R_{k-1})\\, P_k = ${sim.ap.toFixed(3)} \\quad (\\text{random} \\approx ${sim.prevalence.toFixed(3)})`,
    baseline: `\\text{acc}_{\\text{all neg}} = \\frac{TN + FP}{n} = \\frac{${m.tn + m.fp}}{${m.n}} = ${f(sim.allNeg.accuracy)}`,
  };
  return (
    <div className="space-y-2">
      <TexBlock>{tex[phase]}</TexBlock>
      {phase === "summary" && <p className="text-[11px] text-muted">F1 is the harmonic mean <Tex>{"\\tfrac{2PR}{P+R}"}</Tex>, which simplifies to the count form above.</p>}
    </div>
  );
}
