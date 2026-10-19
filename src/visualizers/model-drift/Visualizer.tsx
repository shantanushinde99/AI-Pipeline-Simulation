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
import { StepCaption, ease, pct } from "../classification-metrics/kit";
import { DEFAULT_DRIFT, EDGES, T, simulate, type Batch, type DriftConfig, type DriftSim, type DriftType } from "./simulation";

/* ---------- one timing schedule (seconds at 1×) ---------- */
const HIST = { at: 0.2, dur: 1.0 };
const ECDF = { at: 0.3, dur: 1.1 }, GAP = 1.5;
const BAR = { at: 1.7, dur: 0.6 };
const LABEL = { at: 2.4, dur: 0.8 }, POP = 3.2, RETRAIN = 3.4;
const ROW = (i: number) => 0.2 + i * 0.12;
const duration = (t: number) => (t < 0 || t >= T ? 3000 : 4300);

const KINDS: { value: DriftType; label: string }[] = [
  { value: "none", label: "No drift" }, { value: "covariate", label: "Covariate (inputs move)" },
  { value: "concept", label: "Concept (input→label rule moves)" }, { value: "gradual", label: "Gradual (both creep)" },
];
const PRESETS: { label: string; cfg: Partial<DriftConfig> }[] = [
  { label: "Covariate: alarm, no harm", cfg: { kind: "covariate", retrain: false } },
  { label: "Concept: silent failure", cfg: { kind: "concept", retrain: false, labelDelay: 2 } },
  { label: "Gradual creep", cfg: { kind: "gradual", retrain: false } },
  { label: "Concept + retrain", cfg: { kind: "concept", retrain: true } },
  { label: "Break it: 4-week label delay", cfg: { kind: "concept", labelDelay: 4, retrain: false } },
];

/* ---------- geometry ---------- */
const W = 720, H = 450;
const HX0 = 40, HX1 = 390, HY0 = 40, HY1 = 180, XMIN = -3, XMAX = 3.5, SMAX = 0.26;
const hx = (v: number) => HX0 + ((Math.min(XMAX, Math.max(XMIN, v)) - XMIN) / (XMAX - XMIN)) * (HX1 - HX0);
const hy = (share: number) => HY1 - (Math.min(share, SMAX) / SMAX) * (HY1 - HY0);
const BINS = Array.from({ length: EDGES.length + 1 }, (_, i) => ({ lo: i === 0 ? XMIN + 0.5 : EDGES[i - 1], hi: i === EDGES.length ? XMAX - 0.5 : EDGES[i] }));
const EX0 = 450, EX1 = 700;
const ex = (v: number) => EX0 + ((Math.min(XMAX, Math.max(XMIN, v)) - XMIN) / (XMAX - XMIN)) * (EX1 - EX0);
const ey = (f: number) => HY1 - f * (HY1 - HY0);
const TX = (t: number) => 102 + t * 52;
const PSI = { y0: 225, y1: 280 }, KS = { y0: 300, y1: 350 }, ACC = { y0: 372, y1: 432 };
const accY = (a: number) => ACC.y1 - ((Math.max(0.5, a) - 0.5) / 0.5) * (ACC.y1 - ACC.y0);

function ecdfPath(xs: number[]) {
  const s = [...xs].sort((a, b) => a - b);
  const pts = Array.from({ length: 66 }, (_, i) => XMIN + (i * (XMAX - XMIN)) / 65).map((v) => {
    let lo = 0, hi = s.length;
    while (lo < hi) { const m = (lo + hi) >> 1; if (s[m] <= v) lo = m + 1; else hi = m; }
    return [ex(v), ey(lo / s.length)] as const;
  });
  return pts.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(1)} ${y.toFixed(1)}`).join(" ");
}
const cdfAt = (xs: number[], v: number) => xs.filter((u) => u <= v).length / xs.length;

function Canvas({ sim, t, k, animKey }: { sim: DriftSim; t: number; k: number; animKey: string }) {
  const b: Batch | undefined = sim.batches[t];
  const ref = sim.refs[b?.version ?? (t >= T ? sim.retrains.length : 0)];
  const refShares = b?.refShares ?? sim.batches[0].refShares;
  const modelB = sim.boundaries[b?.version ?? (t >= T ? sim.retrains.length : 0)];
  const trueB = b?.b ?? (t >= T ? sim.batches[T - 1].b : 0.5);
  const lab = t >= 0 && t < T ? sim.batches[t - sim.cfg.labelDelay] : undefined;
  const rt = sim.retrains.find((r) => r.at === t);
  const known = (x: Batch) => x.labelsAt <= Math.min(t, T - 1) && t >= 0;
  const psiMax = Math.max(0.6, ...sim.batches.map((x) => x.psi));
  const ksMax = Math.max(0.35, ...sim.batches.map((x) => x.ks));
  const accPts = sim.batches.filter((x) => known(x) && !(lab && x.t === lab.t));
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full select-none" role="img" aria-label={b ? `Week ${t + 1}: PSI ${b.psi.toFixed(2)}, KS ${b.ks.toFixed(2)}` : "Reference histogram and monitoring timeline"}>
      {/* histogram: reference outline vs this week's bars */}
      <text x={HX0} y={24} fontSize={12} fontWeight={600} fill="var(--fg)">Input histogram: {b ? `week ${t + 1} vs reference` : "training reference"}</text>
      <line x1={HX0} x2={HX1} y1={HY1} y2={HY1} stroke="var(--line)" />
      {[-2, -1, 0, 1, 2, 3].map((v) => <text key={v} x={hx(v)} y={HY1 + 13} textAnchor="middle" fontSize={9} fill="var(--muted)" fontFamily="var(--font-mono)">{v}</text>)}
      {BINS.map((bin, i) => {
        const x = hx(bin.lo) + 1, w = hx(bin.hi) - hx(bin.lo) - 2;
        const target = b ? b.shares[i] : refShares[i];
        return (
          <g key={i}>
            <motion.rect key={`h-${animKey}-${i}`} x={x} width={w} fill="var(--accent)" fillOpacity={b ? 0.55 : 0.25}
              initial={{ y: b ? hy(refShares[i]) : HY1, height: b ? HY1 - hy(refShares[i]) : 0 }} animate={{ y: hy(target), height: HY1 - hy(target) }}
              transition={{ delay: (HIST.at + (b ? 0 : i * 0.06)) * k, duration: HIST.dur * k, ease }} />
            <rect x={x} y={hy(refShares[i])} width={w} height={HY1 - hy(refShares[i])} fill="none" stroke="var(--fg)" strokeOpacity={0.55} strokeDasharray="3 2" />
          </g>
        );
      })}
      <motion.g initial={false} animate={{ x: hx(trueB) }} transition={{ type: "spring", stiffness: 120, damping: 18, delay: HIST.at * k }}>
        <line y1={HY0 - 6} y2={HY1} stroke="var(--err)" strokeWidth={2} strokeDasharray="5 3" />
        <text x={4} y={HY0 + 4} fontSize={10} fill="var(--err)" fontWeight={600} stroke="var(--surface)" strokeWidth={3} paintOrder="stroke">true</text>
      </motion.g>
      <motion.g initial={false} animate={{ x: hx(modelB) }} transition={{ type: "spring", stiffness: 120, damping: 18 }}>
        <line y1={HY0 - 6} y2={HY1} stroke="var(--fg)" strokeWidth={2} />
        <text x={-4} y={HY0 + 18} textAnchor="end" fontSize={10} fill="var(--fg)" fontWeight={600} stroke="var(--surface)" strokeWidth={3} paintOrder="stroke">model</text>
      </motion.g>
      <text x={HX1} y={HY1 + 26} textAnchor="end" fontSize={9} fill="var(--muted)">feature x → (boundary: predict 1 right of it)</text>

      {/* KS: two empirical CDFs and their largest gap */}
      <text x={EX0} y={24} fontSize={12} fontWeight={600} fill="var(--fg)">KS: largest gap between CDFs</text>
      <rect x={EX0} y={HY0} width={EX1 - EX0} height={HY1 - HY0} fill="none" stroke="var(--line)" />
      <path d={ecdfPath(ref)} fill="none" stroke="var(--fg)" strokeOpacity={0.5} strokeWidth={1.5} strokeDasharray="3 2" />
      {b && (
        <g key={`ks-${animKey}`}>
          <motion.path d={ecdfPath(b.xs)} fill="none" stroke="var(--accent)" strokeWidth={2} initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ delay: ECDF.at * k, duration: ECDF.dur * k, ease: "linear" }} />
          <motion.g initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: GAP * k, duration: 0.3 * k }}>
            <line x1={ex(b.ksAt)} x2={ex(b.ksAt)} y1={ey(cdfAt(ref, b.ksAt))} y2={ey(cdfAt(b.xs, b.ksAt))} stroke={b.ks > b.ksCrit ? "var(--err)" : "var(--warn)"} strokeWidth={3} />
            <text x={ex(b.ksAt) + (b.ksAt > 1.5 ? -6 : 6)} y={(ey(cdfAt(ref, b.ksAt)) + ey(cdfAt(b.xs, b.ksAt))) / 2 + 4} textAnchor={b.ksAt > 1.5 ? "end" : "start"} fontSize={11} fontFamily="var(--font-mono)" fill="var(--fg)" stroke="var(--surface)" strokeWidth={3} paintOrder="stroke">D = {b.ks.toFixed(3)}</text>
          </motion.g>
        </g>
      )}
      <text x={EX1} y={HY1 + 13} textAnchor="end" fontSize={9} fill="var(--muted)">dashed = reference · solid = this week</text>

      {/* timeline strips */}
      {[["PSI", PSI], ["KS D", KS], ["accuracy", ACC]].map(([name, s]) => {
        const st = s as typeof PSI;
        return <g key={name as string}><rect x={44} y={st.y0 - 6} width={W - 52} height={st.y1 - st.y0 + 12} rx={4} fill="var(--surface-2)" fillOpacity={0.5} /><text x={48} y={st.y0 + 5} fontSize={10} fontWeight={600} fill="var(--muted)">{name as string}</text></g>;
      })}
      {sim.batches.map((x) => <text key={x.t} x={TX(x.t)} y={H - 4} textAnchor="middle" fontSize={9} fill={x.t === t ? "var(--accent)" : "var(--muted)"} fontWeight={x.t === t ? 700 : 400}>w{x.t + 1}</text>)}
      {t >= 0 && t < T && <motion.rect initial={false} animate={{ x: TX(t) - 22 }} y={PSI.y0 - 6} width={44} height={ACC.y1 - PSI.y0 + 12} rx={5} fill="none" stroke="var(--accent)" strokeWidth={1.5} transition={{ duration: 0.4 * k, ease }} />}
      <line x1={50} x2={W - 12} y1={PSI.y1 - (sim.cfg.psiThreshold / psiMax) * (PSI.y1 - PSI.y0)} y2={PSI.y1 - (sim.cfg.psiThreshold / psiMax) * (PSI.y1 - PSI.y0)} stroke="var(--warn)" strokeDasharray="4 3" />
      <line x1={50} x2={W - 12} y1={KS.y1 - (sim.batches[0].ksCrit / ksMax) * (KS.y1 - KS.y0)} y2={KS.y1 - (sim.batches[0].ksCrit / ksMax) * (KS.y1 - KS.y0)} stroke="var(--warn)" strokeDasharray="4 3" />
      <line x1={50} x2={W - 12} y1={accY(sim.cfg.accThreshold)} y2={accY(sim.cfg.accThreshold)} stroke="var(--err)" strokeDasharray="4 3" />
      {sim.batches.filter((x) => x.t <= t).map((x) => {
        const cur = x.t === t;
        const hP = (Math.min(x.psi, psiMax) / psiMax) * (PSI.y1 - PSI.y0), hK = (Math.min(x.ks, ksMax) / ksMax) * (KS.y1 - KS.y0);
        const tr = { delay: cur ? BAR.at * k : 0, duration: (cur ? BAR.dur : 0) * k, ease };
        return (
          <g key={`${x.t}-${cur ? animKey : "old"}`}>
            <motion.rect x={TX(x.t) - 9} width={18} fill={x.psi > sim.cfg.psiThreshold ? "var(--warn)" : "var(--accent)"} initial={{ y: PSI.y1, height: 0 }} animate={{ y: PSI.y1 - hP, height: hP }} transition={tr} />
            <motion.rect x={TX(x.t) - 9} width={18} fill={x.ks > x.ksCrit ? "var(--warn)" : "var(--accent)"} initial={{ y: KS.y1, height: 0 }} animate={{ y: KS.y1 - hK, height: hK }} transition={tr} />
            {x.inputAlert && <motion.text x={TX(x.t)} y={PSI.y1 - hP - 3} textAnchor="middle" fontSize={10} fill="var(--warn)" fontWeight={700} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: (cur ? BAR.at + BAR.dur : 0) * k }}>!</motion.text>}
            {!known(x) && <text x={TX(x.t)} y={(ACC.y0 + ACC.y1) / 2 + 4} textAnchor="middle" fontSize={11} fill="var(--muted)">?</text>}
          </g>
        );
      })}
      {accPts.length > 1 && <path d={accPts.map((x, i) => `${i ? "L" : "M"}${TX(x.t)} ${accY(x.acc)}`).join(" ")} fill="none" stroke="var(--violet)" strokeWidth={1.5} />}
      {accPts.map((x) => <circle key={x.t} cx={TX(x.t)} cy={accY(x.acc)} r={4} fill={x.accAlert ? "var(--err)" : "var(--violet)"} />)}
      {lab && (
        <g key={`lab-${animKey}`}>
          <motion.circle r={4} fill="var(--violet)" initial={{ cx: TX(t), cy: ACC.y0 - 4, opacity: 0 }} animate={{ cx: TX(lab.t), cy: accY(lab.acc), opacity: [0, 1, 1] }} transition={{ delay: LABEL.at * k, duration: LABEL.dur * k, ease }} />
          <motion.g initial={{ scale: 0, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ delay: POP * k, duration: 0.3 * k }} style={{ originX: `${TX(lab.t)}px`, originY: `${accY(lab.acc)}px` }}>
            <circle cx={TX(lab.t)} cy={accY(lab.acc)} r={5.5} fill={lab.accAlert ? "var(--err)" : "var(--violet)"} />
            <text x={TX(lab.t)} y={accY(lab.acc) - 8} textAnchor="middle" fontSize={10} fontFamily="var(--font-mono)" fill={lab.accAlert ? "var(--err)" : "var(--fg)"} stroke="var(--surface)" strokeWidth={3} paintOrder="stroke">{lab.accAlert ? "✗ " : ""}{(lab.acc * 100).toFixed(0)}%</text>
          </motion.g>
        </g>
      )}
      {sim.retrains.filter((r) => r.at <= t).map((r) => (
        <motion.g key={`rt-${r.at}-${r.at === t ? animKey : "o"}`} initial={{ opacity: r.at === t ? 0 : 1 }} animate={{ opacity: 1 }} transition={{ delay: (r.at === t ? RETRAIN : 0) * k }}>
          <line x1={TX(r.at) + 27} x2={TX(r.at) + 27} y1={PSI.y0 - 10} y2={ACC.y1 + 4} stroke="var(--ok)" strokeWidth={2} strokeDasharray="5 3" />
          <text x={TX(r.at) + 30} y={PSI.y0 - 12} fontSize={10} fill="var(--ok)" fontWeight={600}>↻ retrain v{r.version}</text>
        </motion.g>
      ))}
      {rt && <motion.text x={HX0} y={HY0 - 4} fontSize={10} fill="var(--ok)" fontWeight={600} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: RETRAIN * k }}>↻ model retrained on week {rt.from.map((v) => v + 1).join(", ")}: boundary {sim.boundaries[rt.version - 1].toFixed(2)} → {rt.boundary.toFixed(2)} (from next week)</motion.text>}
    </svg>
  );
}

const GRID = { gridTemplateColumns: "72px 48px 48px minmax(0,1fr) 52px" };

function StepCard({ sim, t, k }: { sim: DriftSim; t: number; k: number }) {
  if (t < 0) return <p className="text-xs">The model is trained on {sim.trainXs.length} labelled rows ({pct(sim.trainAcc)} accuracy). The dashed outline is the <b>reference histogram</b>: every week's inputs will be compared to it bin by bin (PSI) and by their cumulative distributions (KS). Labels arrive {sim.cfg.labelDelay} week(s) after predictions.</p>;
  if (t >= T) {
    const k2 = sim.cfg.kind;
    return (
      <div className="space-y-2 text-xs">
        <KV rows={[["first input alert (PSI/KS)", sim.firstInputAlert !== undefined ? `week ${sim.firstInputAlert + 1}` : "never"], ["first accuracy alert known", sim.firstAccAlertKnown !== undefined ? `week ${sim.firstAccAlertKnown + 1}` : "never"], ["retrains", String(sim.retrains.length)], ["final-week accuracy", pct(sim.batches[T - 1].acc)]]} />
        <Callout tone={k2 === "concept" ? "err" : k2 === "none" ? "ok" : "warn"} title={k2 === "concept" ? "✗ Input monitors were blind" : k2 === "covariate" ? "⚠ Alarm without harm" : k2 === "gradual" ? "⚠ Slow creep" : "✓ Stable"}>
          {k2 === "concept" ? "The inputs never moved, so PSI and KS stayed quiet. Only labels, arriving late, revealed the drop." : k2 === "covariate" ? "PSI and KS fired, but the input→label rule held, so accuracy barely changed. Investigate before retraining." : k2 === "gradual" ? "Each week moves a little; the cumulative shift crosses thresholds late. Compare against a fixed reference, not last week." : "Week-to-week noise stays under the thresholds."}
        </Callout>
      </div>
    );
  }
  const b = sim.batches[t];
  const max = Math.max(sim.cfg.psiThreshold, ...b.contrib);
  return (
    <div className="space-y-1">
      <div className="grid gap-1 text-[10px] text-muted" style={GRID}><span>bin</span><span>ref e</span><span>week a</span><span>(a−e)·ln(a/e)</span><span className="text-right">term</span></div>
      {b.contrib.map((c, i) => (
        <motion.div key={i} className="grid items-center gap-1 font-mono text-[10px]" style={GRID} initial={{ opacity: 0, x: -6 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: ROW(i) * k, duration: 0.25 * k }}>
          <span>{i === 0 ? `< ${EDGES[0]}` : i === EDGES.length ? `≥ ${EDGES[i - 1]}` : `${EDGES[i - 1]}…${EDGES[i]}`}</span>
          <span>{(b.refShares[i] * 100).toFixed(1)}%</span><span>{(b.shares[i] * 100).toFixed(1)}%</span>
          <div className="h-2.5 overflow-hidden rounded bg-surface-2"><motion.div className="h-full rounded bg-warn" initial={{ width: 0 }} animate={{ width: `${(c / max) * 100}%` }} transition={{ delay: (ROW(i) + 0.1) * k, duration: 0.4 * k, ease }} /></div>
          <span className="text-right">{c.toFixed(3)}</span>
        </motion.div>
      ))}
      <p className="pt-1 text-xs">Σ = PSI <b className="font-mono"><AnimatedNumber key={`${t}`} value={b.psi} from={0} digits={3} delay={ROW(b.contrib.length) * k} duration={0.6 * k} /></b> vs threshold {sim.cfg.psiThreshold} → {b.psi > sim.cfg.psiThreshold ? "⚠ alert" : "✓ ok"} · KS D {b.ks.toFixed(3)} vs {b.ksCrit.toFixed(3)} → {b.ks > b.ksCrit ? "⚠ alert" : "✓ ok"}</p>
    </div>
  );
}

function Calc({ sim, t }: { sim: DriftSim; t: number }) {
  if (t < 0 || t >= T) return <TexBlock>{`\\text{PSI} = \\sum_i (a_i - e_i)\\ln\\frac{a_i}{e_i},\\quad D = \\max_x |F_{\\text{ref}}(x) - F_{\\text{week}}(x)|`}</TexBlock>;
  const b = sim.batches[t];
  const i = b.contrib.indexOf(Math.max(...b.contrib));
  const e = Math.max(b.refShares[i], 1e-4), a = Math.max(b.shares[i], 1e-4);
  return (
    <div className="space-y-2">
      <TexBlock>{`(${a.toFixed(3)} - ${e.toFixed(3)})\\ln\\frac{${a.toFixed(3)}}{${e.toFixed(3)}} = ${b.contrib[i].toFixed(4)}`}</TexBlock>
      <TexBlock>{`\\text{PSI} = ${b.psi.toFixed(3)} \\;${b.psi > sim.cfg.psiThreshold ? ">" : "\\le"}\\; ${sim.cfg.psiThreshold}`}</TexBlock>
      <TexBlock>{`D = ${b.ks.toFixed(3)},\\; D_{\\alpha=0.05} = 1.36\\sqrt{\\tfrac{n+m}{nm}} = ${b.ksCrit.toFixed(3)}`}</TexBlock>
    </div>
  );
}

export default function ModelDriftVisualizer({ tab }: { tab: VisualizerTab }) {
  const [cfg, setCfg] = useState(DEFAULT_DRIFT);
  const [replay, setReplay] = useState(0);
  const set = (p: Partial<DriftConfig>) => setCfg((c) => ({ ...c, ...p }));
  const sim = useMemo(() => simulate(cfg), [cfg]);
  const pb = usePlayback(sim.steps.length, (i) => duration(sim.steps[i]?.state.t ?? 0));
  const step = sim.steps[pb.index];
  const t = step.state.t;
  const k = 1 / pb.speed;
  const animKey = `${step.id}-${replay}`;
  if (tab === "metrics") return null;
  const b = sim.batches[Math.min(Math.max(t, 0), T - 1)];
  return (
    <Workspace
      main={
        <>
          <Panel title="Production weeks vs the training reference" actions={<SimulatedLabel>Synthetic 1-feature data</SimulatedLabel>}>
            <StepCaption index={pb.index} total={sim.steps.length} stepId={step.id} headline={step.title} sub={step.description} onReplay={() => setReplay((r) => r + 1)} />
            <div className="overflow-x-auto"><div className="min-w-[600px]"><Canvas sim={sim} t={t} k={k} animKey={animKey} /></div></div>
            <div className="mt-1 flex flex-wrap gap-3 text-[11px] text-muted">
              <span>bars: PSI / KS per week (orange + ! = over threshold)</span><span className="text-violet">● accuracy appears only when labels arrive</span><span>? = labels pending</span><span className="text-err">dashed red = threshold / true boundary</span>
            </div>
          </Panel>
          <Panel title="What's happening in this step">
            <AnimatePresence mode="wait" initial={false}>
              <motion.div key={animKey} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: 0.2 }}>
                <StepCard sim={sim} t={t} k={k} />
              </motion.div>
            </AnimatePresence>
          </Panel>
          <PlaybackControls pb={pb} stepTitles={sim.steps.map((s) => s.title)} />
          <Panel title="Parameters" actions={<Button size="sm" variant="ghost" onClick={() => setCfg(DEFAULT_DRIFT)}><RotateCcw size={12} /> Reset</Button>}>
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              <Select<DriftType> label="Drift type" value={cfg.kind} options={KINDS} onChange={(v) => set({ kind: v })} />
              <Slider label="Drift magnitude" value={cfg.magnitude} min={0} max={2} step={0.1} format={(v) => v.toFixed(1)} onChange={(v) => set({ magnitude: v })} />
              <Slider label="Drift starts in week" value={cfg.start + 1} min={1} max={T} onChange={(v) => set({ start: v - 1 })} />
              <Slider label="Label delay (weeks)" value={cfg.labelDelay} min={0} max={4} onChange={(v) => set({ labelDelay: v })} hint="How long until you learn if predictions were right." />
              <Slider label="PSI alert threshold" value={cfg.psiThreshold} min={0.05} max={0.5} step={0.05} format={(v) => v.toFixed(2)} onChange={(v) => set({ psiThreshold: v })} hint="0.1 watch · 0.2 act (rule of thumb)" />
              <Slider label="Accuracy alert threshold" value={cfg.accThreshold} min={0.6} max={0.95} step={0.01} format={(v) => pct(v, 0)} onChange={(v) => set({ accThreshold: v })} />
              <Toggle label="Retrain on recent data" checked={cfg.retrain} onChange={(v) => set({ retrain: v })} hint="When labels confirm an accuracy alert, retrain on the latest labelled week." />
              <Slider label="Random seed" value={cfg.seed} min={1} max={40} onChange={(v) => set({ seed: v })} />
            </div>
            <div className="mt-3 flex flex-wrap gap-1.5">{PRESETS.map((p) => <Button key={p.label} size="sm" onClick={() => set(p.cfg)}>{p.label}</Button>)}</div>
          </Panel>
        </>
      }
      inspector={
        <>
          <StepExplainer index={pb.index} title={step.title} what={step.description} why={t < 0 ? "Drift is always measured against something. The training snapshot is the usual reference." : t >= T ? "Input monitors are early but blind to concept drift; label monitors see everything but late." : b.inputAlert ? "An input alert says the population changed. It does not say the model got worse." : "Concept drift changes P(y|x) without moving P(x): only labelled outcomes reveal it."} />
          <Panel title="Calculation"><Calc sim={sim} t={t} /></Panel>
          <Panel title="Monitors">
            <KV rows={[
              ["drift type", KINDS.find((x) => x.value === cfg.kind)?.label ?? cfg.kind],
              ["this week PSI / KS", t >= 0 && t < T ? `${b.psi.toFixed(3)} / ${b.ks.toFixed(3)}` : "–"],
              ["input alert", t >= 0 && t < T ? (b.inputAlert ? "⚠ yes" : "✓ no") : "–"],
              ["true accuracy this week", t >= 0 && t < T ? `${pct(b.acc)} (hidden until w${b.labelsAt + 1})` : "–"],
              ["model boundary / true", t >= 0 && t < T ? `${sim.boundaries[b.version].toFixed(2)} / ${b.b.toFixed(2)}` : `${sim.boundaries[0].toFixed(2)} / 0.50`],
              ["retrains so far", String(sim.retrains.filter((r) => r.at <= t).length)],
            ]} />
          </Panel>
          <Panel title="Events"><EventLog events={step.events} /></Panel>
        </>
      }
    />
  );
}
