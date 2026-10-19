import { useMemo, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { RotateCcw } from "lucide-react";
import { usePlayback } from "../../hooks/usePlayback";
import { PlaybackControls } from "../../components/visualizer/PlaybackControls";
import { EventLog } from "../../components/visualizer/EventLog";
import { StepExplainer, Workspace } from "../../components/visualizer/Workspace";
import { AnimatedNumber } from "../../components/visualizer/AnimatedNumber";
import { Button, Callout, KV, Panel, Select, SimulatedLabel, Slider, Toggle } from "../../components/ui";
import { TexBlock } from "../../components/Tex";
import type { VisualizerTab } from "../types";
import { MetricBar, StepCaption, ease, pct } from "../classification-metrics/kit";
import { DEFAULT_LEAK, LEAK_FEATURE, PHASES, simulate, type LeakConfig, type LeakSim, type ModelKind, type Phase, type Tracked } from "./simulation";

/* ---------- one timing schedule (seconds at 1×) ---------- */
const DROP = (i: number) => 0.1 + i * 0.025;
const FLY = (i: number) => 0.2 + i * 0.025;
const LINKS = 1.9;
const FLOW = (i: number) => 0.2 + i * 0.02;
const MARK = (i: number) => 0.3 + i * 0.08;
const DURATIONS: Record<Phase, number> = { load: 2600, features: 3400, split: 3800, scale: 3400, train: 3200, evaluate: 3600, production: 3800 };

const PRESETS: { label: string; cfg: Partial<LeakConfig> }[] = [
  { label: "Clean pipeline", cfg: { leakTarget: false, scaleBeforeSplit: false, duplicates: false } },
  { label: "Target leak", cfg: { leakTarget: true, scaleBeforeSplit: false, duplicates: false } },
  { label: "Duplicates + 1-NN", cfg: { leakTarget: false, scaleBeforeSplit: false, duplicates: true, model: "knn" } },
  { label: "Break it: all three leaks", cfg: { leakTarget: true, scaleBeforeSplit: true, duplicates: true } },
];
const WHY: Record<Phase, string> = {
  load: "Leakage is any path by which information unavailable at prediction time reaches training.",
  features: "Ask of every column: would I know this value at the moment I make the prediction?",
  split: "Split first, by entity (customer, patient, session) or by time, before anything learns from the data.",
  scale: "Every fitted transform (scaler, imputer, encoder, feature selection) must be fit on training rows only.",
  train: "A model happily uses whatever predicts the label best in the data you hand it, legitimate or not.",
  evaluate: "The test score is only an honest estimate if the test rows never influenced training.",
  production: "Production is the only test that cannot leak. A big offline/production gap is the classic symptom.",
};

/* ---------- canvas geometry ---------- */
const W = 720, H = 350, CHIP_Y = 22;
const chipX = (i: number) => 56 + i * 101;
const pool = (i: number) => ({ x: 120 + (i % 15) * 34, y: 150 + Math.floor(i / 15) * 34 });
const trainPos = (i: number) => ({ x: 40 + (i % 10) * 34, y: 112 + Math.floor(i / 10) * 30 });
const testPos = (i: number) => ({ x: 420 + (i % 5) * 30, y: 112 + Math.floor(i / 5) * 30 });
const prodPos = (i: number) => ({ x: 600 + (i % 4) * 28, y: 112 + Math.floor(i / 4) * 30 });
const SCALER = { x: 210, y: 262 }, MODEL = { x: 210, y: 262 };
const COLS_Y = 72;

function Dot({ y, r = 6 }: { y: number; r?: number }) {
  return y ? <circle r={r} fill="var(--accent)" /> : <circle r={r - 0.75} fill="var(--surface)" stroke="var(--violet)" strokeWidth={1.5} />;
}

function Canvas({ sim, phase, k, animKey }: { sim: LeakSim; phase: Phase; k: number; animKey: string }) {
  const pi = PHASES.indexOf(phase);
  const split = pi >= 2;
  // A readable sample: originals 0..44 plus any duplicate copies of them.
  const sample = useMemo(() => sim.rows.filter((r) => (r.twinOf ?? r.id) < 45).sort((a, b) => a.id - b.id), [sim.rows]);
  const pos = useMemo(() => {
    const m = new Map<number, { x: number; y: number }>();
    let a = 0, b = 0;
    sample.forEach((r, i) => m.set(r.id, !split ? pool(i) : r.split === "train" ? (a < 40 ? trainPos(a++) : { x: -40, y: 0 }) : (b < 20 ? testPos(b++) : { x: -40, y: 0 })));
    return m;
  }, [sample, split]);
  const testIndex = useMemo(() => new Map(sim.rows.filter((r) => r.split === "test").map((r, i) => [r.id, i])), [sim.rows]);
  const twinLinks = useMemo(() => {
    if (!sim.cfg.duplicates) return [];
    const fam = (r: Tracked) => r.twinOf ?? r.id;
    return sample.filter((r) => r.split === "test").flatMap((t) => sample.filter((o) => o.split === "train" && fam(o) === fam(t)).map((o) => [t, o] as const));
  }, [sample, sim.cfg.duplicates]);
  const prod = sim.prodRows.slice(0, 16);
  const fitters = sample.filter((r) => r.split === "train" || sim.cfg.scaleBeforeSplit);
  const cols = [...sim.features, "y (label)"];
  const leakCol = sim.features.indexOf(LEAK_FEATURE);
  const colX = (j: number) => 90 + j * 120;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full select-none" role="img" aria-label={`Pipeline at stage ${phase}`}>
      {/* stage chips with a travelling pulse */}
      <line x1={chipX(0)} x2={chipX(6)} y1={CHIP_Y} y2={CHIP_Y} stroke="var(--line)" strokeWidth={2} />
      {PHASES.map((p, i) => (
        <g key={p} transform={`translate(${chipX(i)} ${CHIP_Y})`}>
          <rect x={-44} y={-12} width={88} height={24} rx={12} fill={i === pi ? "var(--accent)" : "var(--surface-2)"} stroke={i < pi ? "var(--ok)" : "var(--line)"} />
          <text y={4} textAnchor="middle" fontSize={11} fill={i === pi ? "var(--surface)" : "var(--fg)"} fontWeight={i === pi ? 600 : 400}>{i < pi ? "✓ " : ""}{p}</text>
        </g>
      ))}
      <motion.circle key={`pulse-${animKey}`} r={5} cy={CHIP_Y + 16} fill="var(--accent)" initial={{ cx: chipX(Math.max(0, pi - 1)) }} animate={{ cx: chipX(pi) }} transition={{ duration: 0.6 * k, ease }} />

      {/* columns + target-leak arrow */}
      {pi === 1 && (
        <g key={`cols-${animKey}`}>
          {cols.map((c, j) => (
            <motion.g key={c} initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: j * 0.15 * k, duration: 0.4 * k }}>
              <rect x={colX(j) - 56} y={COLS_Y - 13} width={112} height={24} rx={5} fill={j === leakCol ? "var(--err)" : "var(--surface-2)"} fillOpacity={j === leakCol ? 0.18 : 1} stroke={j === leakCol ? "var(--err)" : j === cols.length - 1 ? "var(--accent)" : "var(--line)"} />
              <text x={colX(j)} y={COLS_Y + 3} textAnchor="middle" fontSize={10} fontFamily="var(--font-mono)" fill="var(--fg)">{j === leakCol ? "⚠ " : ""}{c.replace("_months", "").replace("_logged", "")}</text>
            </motion.g>
          ))}
          {leakCol >= 0 && (() => {
            const d = `M${colX(cols.length - 1)} ${COLS_Y + 12} C ${colX(cols.length - 1) - 10} ${COLS_Y + 44}, ${colX(leakCol) + 10} ${COLS_Y + 44}, ${colX(leakCol)} ${COLS_Y + 13}`;
            return (
              <g>
                <motion.path d={d} fill="none" stroke="var(--err)" strokeWidth={2.5} markerEnd="url(#dl-arrow)" initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ delay: 0.9 * k, duration: 0.9 * k, ease }} />
                <motion.text x={(colX(leakCol) + colX(cols.length - 1)) / 2} y={COLS_Y + 52} textAnchor="middle" fontSize={11} fill="var(--err)" fontWeight={600} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1.6 * k }}>label leaks into a feature (written after cancel)</motion.text>
              </g>
            );
          })()}
        </g>
      )}
      <defs><marker id="dl-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="var(--err)" /></marker></defs>

      {/* region frames */}
      {split && (
        <motion.g initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.4 * k }} fontSize={11} fontWeight={600}>
          <rect x={22} y={86} width={362} height={140} rx={8} fill="var(--ok)" fillOpacity={0.05} stroke="var(--line)" />
          <text x={30} y={80} fill="var(--fg)">TRAIN · {sim.nTrain} rows</text>
          <rect x={404} y={86} width={152} height={140} rx={8} fill="var(--warn)" fillOpacity={0.06} stroke="var(--line)" />
          <text x={410} y={80} fill="var(--fg)">TEST · {sim.nTest}</text>
          {pi >= 6 && <><rect x={584} y={86} width={126} height={140} rx={8} fill="var(--violet)" fillOpacity={0.06} stroke="var(--line)" /><text x={590} y={80} fill="var(--fg)">PRODUCTION</text></>}
        </motion.g>
      )}

      {/* dots */}
      <g key={`dots-${pi < 2 ? (pi === 0 ? animKey : "pool") : pi === 2 ? animKey : "split"}`}>
        {sample.map((r, i) => {
          const p = pos.get(r.id) ?? { x: -40, y: 0 };
          const from = pi === 0 ? { x: p.x, y: -20, opacity: 0 } : pi === 2 ? { ...pool(i), opacity: 1 } : false;
          const ti = testIndex.get(r.id);
          const judged = pi >= 5 && ti !== undefined;
          const ok = ti !== undefined && sim.offline.correct[ti];
          return (
            <motion.g key={r.id} initial={from} animate={{ x: p.x, y: p.y, opacity: pi === 4 && r.split === "train" ? 0.35 : 1 }} transition={{ delay: (pi === 0 ? DROP(i) : pi === 2 ? FLY(i) : 0) * k, duration: 0.6 * k, ease }}>
              <Dot y={r.y} />
              {r.twinOf !== undefined && <text y={-8} textAnchor="middle" fontSize={8} fill="var(--err)">copy</text>}
              {judged && (
                <motion.g key={`j-${animKey}`} initial={{ scale: 0, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ delay: (pi === 5 ? MARK(ti % 20) : 0) * k, duration: 0.3 * k }}>
                  <circle r={10} fill="none" stroke={ok ? "var(--ok)" : "var(--err)"} strokeWidth={2} />
                  {!ok && <text x={9} y={-6} fontSize={10} fill="var(--err)" fontWeight={700}>✗</text>}
                </motion.g>
              )}
            </motion.g>
          );
        })}
      </g>

      {/* duplicate twins straddling the split */}
      {split && twinLinks.map(([t, o], i) => {
        const a = pos.get(t.id), b = pos.get(o.id);
        if (!a || !b || a.x < 0 || b.x < 0) return null;
        return (
          <g key={`tw-${t.id}-${o.id}-${pi === 2 ? animKey : "s"}`}>
            <motion.line x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke="var(--err)" strokeWidth={1.5} strokeDasharray="4 3" initial={{ pathLength: 0, opacity: 0 }} animate={{ pathLength: 1, opacity: 0.8 }} transition={{ delay: (pi === 2 ? LINKS + i * 0.08 : 0) * k, duration: 0.5 * k }} />
            {pi === 2 && <motion.circle r={3.5} fill="var(--err)" initial={{ cx: a.x, cy: a.y, opacity: 0 }} animate={{ cx: b.x, cy: b.y, opacity: [0, 1, 0] }} transition={{ delay: (LINKS + 0.4 + i * 0.08) * k, duration: 0.9 * k }} />}
          </g>
        );
      })}
      {split && twinLinks.length > 0 && <text x={404} y={246} fontSize={11} fill="var(--err)" fontWeight={600}>⚠ {sim.leakedTwins} test rows have a twin in train</text>}

      {/* scaler: statistics flow from rows into μ, σ */}
      {pi === 3 && (
        <g key={`scale-${animKey}`}>
          {fitters.map((r, i) => {
            const p = pos.get(r.id);
            if (!p || p.x < 0) return null;
            const leak = r.split === "test";
            return <motion.circle key={r.id} r={3} fill={leak ? "var(--err)" : "var(--ok)"} initial={{ cx: p.x, cy: p.y, opacity: 0 }} animate={{ cx: SCALER.x, cy: SCALER.y, opacity: [0, 1, 1, 0] }} transition={{ delay: FLOW(i) * k, duration: 0.8 * k, ease }} />;
          })}
          {sim.cfg.scaleBeforeSplit && <motion.path d={`M480 228 Q 440 290 ${SCALER.x + 97} ${SCALER.y}`} fill="none" stroke="var(--err)" strokeWidth={2.5} markerEnd="url(#dl-arrow)" initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ delay: 0.4 * k, duration: 0.8 * k }} />}
          <rect x={SCALER.x - 95} y={SCALER.y - 16} width={190} height={32} rx={6} fill="var(--surface)" stroke={sim.cfg.scaleBeforeSplit ? "var(--err)" : "var(--ok)"} strokeWidth={2} />
          <text x={SCALER.x} y={SCALER.y + 4} textAnchor="middle" fontSize={11} fontFamily="var(--font-mono)" fill="var(--fg)">scaler μ, σ ← {sim.scaler.fitOn}</text>
        </g>
      )}

      {/* training: rows pour into the model */}
      {pi === 4 && (
        <g key={`train-${animKey}`}>
          {sample.filter((r) => r.split === "train").slice(0, 40).map((r, i) => {
            const p = pos.get(r.id) ?? MODEL;
            return <motion.circle key={r.id} r={4} fill={r.y ? "var(--accent)" : "var(--violet)"} initial={{ cx: p.x, cy: p.y, opacity: 1 }} animate={{ cx: MODEL.x, cy: MODEL.y, opacity: 0 }} transition={{ delay: FLOW(i) * k, duration: 0.9 * k, ease }} />;
          })}
        </g>
      )}
      {pi >= 4 && (
        <g>
          <rect x={MODEL.x - 70} y={MODEL.y - 18} width={140} height={36} rx={8} fill="var(--accent)" fillOpacity={0.12} stroke="var(--accent)" strokeWidth={2} />
          <text x={MODEL.x} y={MODEL.y + 4} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--fg)">{sim.cfg.model === "logistic" ? "logistic model" : "1-NN (memory)"}</text>
        </g>
      )}

      {/* production: fresh customers arrive from the right */}
      {pi === 6 && prod.map((r, i) => {
        const p = prodPos(i), ok = sim.production.correct[i];
        return (
          <motion.g key={`p-${i}-${animKey}`} initial={{ x: W + 20, y: p.y }} animate={{ x: p.x, y: p.y }} transition={{ delay: (0.2 + i * 0.06) * k, duration: 0.6 * k, ease }}>
            <Dot y={r.y} />
            <motion.g initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ delay: (1.4 + i * 0.07) * k, duration: 0.3 * k }}>
              <circle r={10} fill="none" stroke={ok ? "var(--ok)" : "var(--err)"} strokeWidth={2} />
              {!ok && <text x={9} y={-6} fontSize={10} fill="var(--err)" fontWeight={700}>✗</text>}
            </motion.g>
          </motion.g>
        );
      })}

      {/* score counters */}
      {pi >= 5 && (
        <text x={404} y={306} fontSize={12} fontFamily="var(--font-mono)" fill="var(--fg)">offline acc <AnimatedNumber key={`o-${pi === 5 ? animKey : "x"}`} svg value={sim.offline.accuracy * 100} from={pi === 5 ? 0 : undefined} digits={1} delay={(pi === 5 ? 1.8 : 0) * k} duration={0.8 * k} />%</text>
      )}
      {pi === 6 && (
        <text x={404} y={328} fontSize={12} fontFamily="var(--font-mono)" fill="var(--violet)" fontWeight={600}>production acc <AnimatedNumber key={`p-${animKey}`} svg value={sim.production.accuracy * 100} from={0} digits={1} delay={2.4 * k} duration={0.8 * k} />%</text>
      )}
    </svg>
  );
}

function StepCard({ sim, phase, k }: { sim: LeakSim; phase: Phase; k: number }) {
  const s = sim;
  if (phase === "load") return <div className="space-y-2"><MetricBar label="churn rate" value={s.rows.filter((r) => r.twinOf === undefined && r.y).length / s.cfg.n} k={k} /><p className="text-xs text-muted">{s.cfg.n} customers. Filled dots churned, hollow dots stayed. The canvas shows a sample of {s.rows.filter((r) => (r.twinOf ?? r.id) < 45).length} rows.</p></div>;
  if (phase === "features") return <div className="space-y-1.5"><p className="text-xs text-muted">Correlation of each feature with the label on the training rows:</p>{s.corr.map((c, j) => <MetricBar key={c.name} label={c.name.replace("_months", "").replace("_logged", "")} value={Math.abs(c.r)} delay={0.2 + j * 0.2} k={k} tone={c.name === LEAK_FEATURE ? "err" : "accent"} note={c.name === LEAK_FEATURE ? "too good to be true" : undefined} />)}</div>;
  if (phase === "split") return <div className="space-y-1.5"><MetricBar label="train share" value={s.nTrain / s.rows.length} k={k} tone="ok" /><MetricBar label="test share" value={s.nTest / s.rows.length} delay={0.3} k={k} tone="warn" /><MetricBar label="test rows with twin in train" value={s.leakedTwins / s.nTest} delay={0.6} k={k} tone="err" note={`${s.leakedTwins}/${s.nTest}`} /></div>;
  if (phase === "scale") return (
    <div className="overflow-x-auto"><table className="w-full min-w-[420px] text-xs"><thead><tr className="text-left text-muted"><th>feature</th><th>μ train-only</th><th>μ used</th><th>σ train-only</th><th>σ used</th></tr></thead>
      <tbody>{s.features.map((f, j) => <motion.tr key={f} initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: j * 0.25 * k }} className="font-mono"><td className="font-sans">{f.replace("_months", "")}</td><td>{s.scaler.trainOnlyMean[j].toFixed(3)}</td><td className={s.cfg.scaleBeforeSplit ? "text-err" : ""}>{s.scaler.mean[j].toFixed(3)}</td><td>{s.scaler.trainOnlyStd[j].toFixed(3)}</td><td className={s.cfg.scaleBeforeSplit ? "text-err" : ""}>{s.scaler.std[j].toFixed(3)}</td></motion.tr>)}</tbody></table></div>
  );
  if (phase === "train") {
    if (!s.weights.length) return <p className="text-xs">1-NN keeps all {s.nTrain} training rows. A test row that is an exact copy of a training row gets that row's label back: guaranteed correct, zero generalization.</p>;
    const max = Math.max(...s.weights.map((w) => Math.abs(w.w)));
    return <div className="space-y-1.5"><p className="text-xs text-muted">|weight| relative to the largest (standardized features):</p>{s.weights.map((w, j) => <MetricBar key={w.name} label={w.name.replace("_months", "").replace("_logged", "")} value={Math.abs(w.w) / max} delay={0.2 + j * 0.2} k={k} tone={w.name === LEAK_FEATURE ? "err" : "accent"} note={`w = ${w.w.toFixed(2)}`} />)}</div>;
  }
  const bars = (
    <div className="space-y-1.5">
      <MetricBar label="offline acc" value={s.offline.accuracy} delay={0.3} k={k} tone="warn" />
      <MetricBar label="offline recall" value={s.offline.recall} delay={0.5} k={k} tone="warn" />
      {phase === "production" && <><MetricBar label="prod acc" value={s.production.accuracy} delay={1.2} k={k} tone="violet" /><MetricBar label="prod recall" value={s.production.recall} delay={1.4} k={k} tone="violet" /></>}
    </div>
  );
  const gap = s.offline.accuracy - s.production.accuracy;
  return <div className="space-y-2">{bars}{phase === "production" && <Callout tone={gap > 0.05 ? "err" : "ok"} title={gap > 0.05 ? `✗ ${(gap * 100).toFixed(1)}-point gap: the offline score was inflated` : "✓ Offline estimate holds up in production"}>{s.cfg.scaleBeforeSplit && !s.cfg.leakTarget && !s.cfg.duplicates ? "The scaler leak is real but tiny here: a mean and a std computed on 30% more rows barely change. It becomes large when the fitted step uses labels (target encoding, feature selection) or future data." : gap > 0.05 ? "Information from the test labels reached training. Fix the pipeline, not the model." : "No information crossed from test to train, so the held-out score was an honest estimate."}</Callout>}</div>;
}

function Calc({ sim, phase }: { sim: LeakSim; phase: Phase }) {
  const s = sim, nTestCorrect = s.offline.correct.filter(Boolean).length, nProdCorrect = s.production.correct.filter(Boolean).length;
  const tex: Record<Phase, string> = {
    load: `\\text{churn rate} = \\frac{${s.rows.filter((r) => r.twinOf === undefined && r.y).length}}{${s.cfg.n}}`,
    features: `r = \\frac{\\sum (x_i-\\bar x)(y_i-\\bar y)}{\\sqrt{\\sum (x_i-\\bar x)^2 \\sum (y_i-\\bar y)^2}} \\Rightarrow r_{\\text{${s.corr[s.corr.length - 1].name.replace(/_/g, "\\_")}}} = ${s.corr[s.corr.length - 1].r.toFixed(2)}`,
    split: `n_{\\text{train}} = ${s.nTrain},\\; n_{\\text{test}} = ${s.nTest},\\; \\text{twins} = ${s.leakedTwins}`,
    scale: `z = \\frac{x - \\mu}{\\sigma},\\quad \\mu_{\\text{tenure}} = ${s.scaler.mean[0].toFixed(2)},\\; \\sigma = ${s.scaler.std[0].toFixed(2)}`,
    train: s.cfg.model === "logistic" ? `\\hat p = \\sigma\\!\\left(b + \\textstyle\\sum_j w_j z_j\\right)` : `\\hat y = y_{\\arg\\min_i \\lVert z - z_i \\rVert}`,
    evaluate: `\\text{acc}_{\\text{offline}} = \\frac{${nTestCorrect}}{${s.nTest}} = ${(s.offline.accuracy * 100).toFixed(1)}\\%`,
    production: `\\text{acc}_{\\text{prod}} = \\frac{${nProdCorrect}}{${s.production.n}} = ${(s.production.accuracy * 100).toFixed(1)}\\%,\\; \\Delta = ${((s.offline.accuracy - s.production.accuracy) * 100).toFixed(1)}\\text{ pts}`,
  };
  return <TexBlock>{tex[phase]}</TexBlock>;
}

export default function DataLeakageVisualizer({ tab }: { tab: VisualizerTab }) {
  const [cfg, setCfg] = useState(DEFAULT_LEAK);
  const [replay, setReplay] = useState(0);
  const set = (p: Partial<LeakConfig>) => setCfg((c) => ({ ...c, ...p }));
  const sim = useMemo(() => simulate(cfg), [cfg]);
  const pb = usePlayback(sim.steps.length, (i) => DURATIONS[sim.steps[i]?.state.phase ?? "load"]);
  const step = sim.steps[pb.index];
  const phase = step.state.phase;
  const k = 1 / pb.speed;
  const animKey = `${step.id}-${replay}`;
  if (tab === "metrics") return null;
  const leaks = [cfg.leakTarget && "target feature", cfg.scaleBeforeSplit && "scaler before split", cfg.duplicates && "duplicates"].filter(Boolean).join(", ") || "none";
  const gap = sim.offline.accuracy - sim.production.accuracy;
  return (
    <Workspace
      main={
        <>
          <Panel title="Pipeline and the rows flowing through it" actions={<SimulatedLabel>Synthetic churn data</SimulatedLabel>}>
            <StepCaption index={pb.index} total={sim.steps.length} stepId={step.id} headline={step.title} sub={step.description} onReplay={() => setReplay((r) => r + 1)} />
            <div className="overflow-x-auto"><div className="min-w-[600px]"><Canvas sim={sim} phase={phase} k={k} animKey={animKey} /></div></div>
            <div className="mt-1 flex flex-wrap gap-3 text-[11px] text-muted">
              <span>● churned · ○ stayed</span><span className="text-ok">◯ correct</span><span className="text-err">◯✗ wrong</span><span className="text-err">red = information crossing from test/label into training</span>
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
          <Panel title="Parameters" actions={<Button size="sm" variant="ghost" onClick={() => setCfg(DEFAULT_LEAK)}><RotateCcw size={12} /> Reset</Button>}>
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              <Toggle label="Leak: keep cancel_ticket_logged" checked={cfg.leakTarget} onChange={(v) => set({ leakTarget: v })} hint="Written by billing after a customer cancels." />
              <Toggle label="Leak: fit scaler before split" checked={cfg.scaleBeforeSplit} onChange={(v) => set({ scaleBeforeSplit: v })} hint="μ and σ computed on train + test." />
              <Toggle label="Leak: duplicate rows before split" checked={cfg.duplicates} onChange={(v) => set({ duplicates: v })} hint="30% of rows copied, twins land on both sides." />
              <Select<ModelKind> label="Model" value={cfg.model} onChange={(v) => set({ model: v })} options={[{ value: "logistic", label: "Logistic regression" }, { value: "knn", label: "1-nearest-neighbour (memorizes)" }]} />
              <Slider label="Customers" value={cfg.n} min={80} max={480} step={20} onChange={(v) => set({ n: v })} />
              <Slider label="Random seed" value={cfg.seed} min={1} max={40} onChange={(v) => set({ seed: v })} />
            </div>
            <div className="mt-3 flex flex-wrap gap-1.5">{PRESETS.map((p) => <Button key={p.label} size="sm" onClick={() => set(p.cfg)}>{p.label}</Button>)}</div>
          </Panel>
        </>
      }
      inspector={
        <>
          <StepExplainer index={pb.index} title={step.title} what={step.description} why={WHY[phase]} />
          <Panel title="Calculation"><Calc sim={sim} phase={phase} /></Panel>
          <Panel title="Offline vs production">
            <KV rows={[
              ["active leaks", leaks],
              ["model", cfg.model === "logistic" ? "logistic regression" : "1-NN"],
              ["offline accuracy", pct(sim.offline.accuracy)],
              ["production accuracy", pct(sim.production.accuracy)],
              ["gap", `${(gap * 100).toFixed(1)} pts ${gap > 0.05 ? "✗ inflated" : "✓ honest"}`],
              ["offline / prod recall", `${pct(sim.offline.recall)} / ${pct(sim.production.recall)}`],
              ["test rows with train twin", `${sim.leakedTwins} / ${sim.nTest}`],
            ]} />
          </Panel>
          <Panel title="Events"><EventLog events={step.events} /></Panel>
        </>
      }
    />
  );
}
