import { useMemo, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { CheckCircle2, RotateCcw, XCircle } from "lucide-react";
import { usePlayback } from "../../hooks/usePlayback";
import { PlaybackControls } from "../../components/visualizer/PlaybackControls";
import { EventLog } from "../../components/visualizer/EventLog";
import { StepExplainer, Workspace } from "../../components/visualizer/Workspace";
import { AnimatedNumber } from "../../components/visualizer/AnimatedNumber";
import { Badge, Button, Callout, KV, NumberField, Panel, Select, SimulatedLabel, Slider, cx } from "../../components/ui";
import { TexBlock } from "../../components/Tex";
import type { VisualizerTab } from "../types";
import { StepCaption, ease, pct, pts } from "./kit";
import { DEFAULT_OFFLINE, HIST_FRACTIONS, SCENARIOS, SLICES, simulate, type OfflineParams, type OfflineSim, type Phase, type Scenario } from "./simulation";

const PHASES: Phase[] = ["dataset", "runA", "runB", "aggregate", "resample", "bootstrap", "slices", "reweight"];
const order = (p: Phase) => PHASES.indexOf(p);

/** One schedule (seconds at 1×) for canvas, card and counters. */
const TOK = { start: 0.3, gap: 0.03, travel: 1.0, grade: 0.7 };
const tokenStart = (i: number) => TOK.start + i * TOK.gap;
const gradedAt = (i: number) => tokenStart(i) + TOK.travel * TOK.grade;
const PICK = { start: 0.3, gap: 0.028 };
const HIST = { start: 0.3, grow: 2.6, ci: 3.1 };
const SLICE_AT = (i: number) => 0.3 + i * 0.55;
const DURATIONS: Record<Phase, number> = { dataset: 2600, runA: 4300, runB: 4300, aggregate: 2600, resample: 3400, bootstrap: 4400, slices: 3600, reweight: 3200 };

const NARRATION: Record<Phase, (s: OfflineSim) => { headline: string; sub: string }> = {
  dataset: () => ({ headline: "A fixed, labelled test set, split into slices", sub: "Every example has a known right answer. Slices let us check each segment, not just the average." }),
  runA: (s) => ({ headline: "Model A answers every example; a grader marks each one", sub: `Each token travels: dataset → model → grader → correct or wrong bin. A gets ${s.nA}/${s.data.length}.` }),
  runB: (s) => ({ headline: "Model B answers the exact same examples", sub: `Same inputs, same order, same grader, so the comparison is paired. B gets ${s.nB}/${s.data.length}.` }),
  aggregate: (s) => ({ headline: `Aggregate: B ${pts(s.accB - s.accA)} over A`, sub: "One number from one sample of 80 examples. It hides both uncertainty and where the change came from." }),
  resample: (s) => ({ headline: "Bootstrap: redraw 80 examples with replacement", sub: `Each dot is one draw landing on an example. ${s.neverPicked} examples were never drawn this time.` }),
  bootstrap: (s) => ({ headline: `${s.params.resamples} resamples build a distribution of Δ`, sub: `The middle 95% is the confidence interval: ${pts(s.boot.ci[0])} to ${pts(s.boot.ci[1])}.` }),
  slices: (s) => s.worst.regress
    ? { headline: `The trap: ${s.worst.name} collapses (${pts(s.worst.diff, 0)})`, sub: "The aggregate improved because the big slices improved. The small slice's regression is drowned out." }
    : { headline: "Per-slice view: no slice clearly regresses", sub: "Each slice's CI is shown. Small slices have wide intervals, so look at both size and direction." },
  reweight: (s) => ({ headline: `Weighted to real traffic: A ${pct(s.prodA)} vs B ${pct(s.prodB)}`, sub: s.prodB < s.prodA ? "With production's slice mix, the 'better' model is actually worse. The eval mix decided the verdict." : "With production's slice mix the verdict holds. Always check that the eval mix matches traffic." }),
};

export default function OfflineEvaluationVisualizer({ tab }: { tab: VisualizerTab }) {
  const [params, setParams] = useState<OfflineParams>(DEFAULT_OFFLINE);
  const [replay, setReplay] = useState(0);
  const sim = useMemo(() => simulate(params), [params]);
  const pb = usePlayback(sim.steps.length, (i) => DURATIONS[PHASES[i]] ?? 3000);
  const step = sim.steps[pb.index];
  const phase = step.state.phase;
  const k = 1 / pb.speed;
  const animKey = `${step.id}-${replay}`;
  const set = (p: Partial<OfflineParams>) => setParams((c) => ({ ...c, ...p }));
  if (tab === "metrics") return null;
  const story = NARRATION[phase](sim);

  return (
    <Workspace
      main={
        <>
          <Panel title="Offline evaluation" actions={<SimulatedLabel>Simulated model outputs</SimulatedLabel>}>
            <StepCaption index={pb.index} total={sim.steps.length} stepKey={step.id} headline={story.headline} sub={story.sub} onReplay={() => setReplay((r) => r + 1)} />
            <div className="overflow-x-auto">
              <div className="min-w-[560px]">
                <Canvas sim={sim} phase={phase} k={k} animKey={animKey} />
              </div>
            </div>
            <div className="mt-2 flex flex-wrap gap-3 text-[11px] text-muted">
              <span className="flex items-center gap-1"><CheckCircle2 size={12} className="text-ok" /> correct (filled)</span>
              <span className="flex items-center gap-1"><XCircle size={12} className="text-err" /> wrong (outlined ×)</span>
              <span>one cell = one labelled example</span>
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
          <Panel title="Parameters" actions={<Button size="sm" variant="ghost" onClick={() => { setParams(DEFAULT_OFFLINE); pb.restart(); }}><RotateCcw size={12} /> Reset</Button>}>
            <div className="grid gap-3 sm:grid-cols-2">
              <Select label="Scenario" value={params.scenario} options={(Object.keys(SCENARIOS) as Scenario[]).map((v) => ({ value: v, label: SCENARIOS[v].label }))}
                onChange={(v) => set({ scenario: v, bNonEn: SCENARIOS[v].bNonEn })} />
              <Slider label="Break it: model B correct on Non-English (of 10)" value={params.bNonEn} min={0} max={10} onChange={(v) => set({ bNonEn: v })} hint="Model A gets 8/10 on this slice." />
              <Slider label="Bootstrap resamples" value={params.resamples} min={100} max={2000} step={100} onChange={(v) => set({ resamples: v })} />
              <Slider label="Non-English share of production traffic" value={params.prodShareNonEn} min={0} max={0.8} step={0.05} format={(v) => pct(v, 0)} onChange={(v) => set({ prodShareNonEn: v })} hint={`Eval set: ${pct(sim.evalWeights[3])}`} />
              <NumberField label="Seed (which examples each model misses, which resamples)" value={params.seed} step={1} min={1} max={9999} onChange={(v) => set({ seed: Math.round(v) })} />
            </div>
          </Panel>
        </>
      }
      inspector={
        <>
          <StepExplainer index={pb.index} title={step.title} what={step.description}
            why={phase === "slices" || phase === "reweight" ? "An aggregate is a weighted average over slices. Change the weights (the mix) and the winner can flip." : "Offline evals are cheap and repeatable, so they gate every model change before users see it."} />
          <Panel title="Calculation"><Calculation sim={sim} phase={phase} /></Panel>
          <Panel title="Key values">
            <KV rows={[
              ["examples", sim.data.length],
              ["accuracy A", order(phase) >= order("runA") ? pct(sim.accA) : "not run yet"],
              ["accuracy B", order(phase) >= order("runB") ? pct(sim.accB) : "not run yet"],
              ["Δ (B − A)", order(phase) >= order("aggregate") ? pts(sim.accB - sim.accA) : "not computed yet"],
              ["95% bootstrap CI", order(phase) >= order("bootstrap") ? `${pts(sim.boot.ci[0])} … ${pts(sim.boot.ci[1])}` : "not computed yet"],
              ["worst slice", order(phase) >= order("slices") ? `${sim.worst.name} ${pts(sim.worst.diff, 0)}` : "not computed yet"],
            ]} />
          </Panel>
          <Panel title="Events"><EventLog events={step.events} /></Panel>
        </>
      }
    />
  );
}

/* ---------- canvas ---------- */
const W = 680, H = 430, PITCH = 14, CELL = 12, GA = 128, GB = 292, TOP = 34;
const sliceRows = SLICES.map((s) => Math.ceil(s.n / 10));
const sliceY = SLICES.map((_, i) => TOP + sliceRows.slice(0, i).reduce((a, b) => a + b, 0) * PITCH + i * 8);
const sliceStart = SLICES.map((_, i) => SLICES.slice(0, i).reduce((a, s) => a + s.n, 0));
function cellXY(id: number, gx: number): [number, number] {
  const si = sliceStart.findLastIndex((st) => id >= st);
  const j = id - sliceStart[si];
  return [gx + (j % 10) * PITCH, sliceY[si] + Math.floor(j / 10) * PITCH];
}

function Cell({ x, y, ok, show, delay, k, flag }: { x: number; y: number; ok: boolean; show: "none" | "static" | "animate"; delay: number; k: number; flag?: boolean }) {
  const inner = ok
    ? <><rect width={CELL} height={CELL} rx={2} fill="var(--ok)" fillOpacity={0.85} /><path d="M3 6.5 L5.2 8.6 L9 3.8" stroke="var(--surface)" strokeWidth={1.6} fill="none" /></>
    : <><rect x={0.75} y={0.75} width={CELL - 1.5} height={CELL - 1.5} rx={2} fill="var(--err-soft)" stroke="var(--err)" strokeWidth={1.5} /><path d="M3.5 3.5 L8.5 8.5 M8.5 3.5 L3.5 8.5" stroke="var(--err)" strokeWidth={1.4} /></>;
  return (
    <g transform={`translate(${x} ${y})`}>
      <rect width={CELL} height={CELL} rx={2} fill="var(--surface-2)" stroke={flag ? "var(--err)" : "var(--line)"} strokeWidth={flag ? 1.5 : 0.75} />
      {show === "static" && inner}
      {show === "animate" && (
        <motion.g initial={{ opacity: 0, scale: 0.2 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: delay * k, duration: 0.3 * k, ease }}>{inner}</motion.g>
      )}
    </g>
  );
}

function Canvas({ sim, phase, k, animKey }: { sim: OfflineSim; phase: Phase; k: number; animKey: string }) {
  const o = order(phase);
  const showA = o > order("runA") ? "static" : o === order("runA") ? "animate" : "none";
  const showB = o > order("runB") ? "static" : o === order("runB") ? "animate" : "none";
  const flagSlice = o >= order("slices") && sim.worst.regress ? sim.worst.id : null;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label={`Results grid for models A and B over ${sim.data.length} examples; current step: ${phase}`}>
      <text x={GA} y={20} fontSize={11} fontWeight={600} fill="var(--fg)">Model A (baseline)</text>
      <text x={GB} y={20} fontSize={11} fontWeight={600} fill="var(--fg)">Model B (candidate)</text>
      {SLICES.map((s, i) => (
        <g key={s.id}>
          <text x={0} y={sliceY[i] + 10} fontSize={11} fill={flagSlice === s.id ? "var(--err)" : "var(--fg)"} fontWeight={flagSlice === s.id ? 700 : 400}>{s.name}</text>
          <text x={0} y={sliceY[i] + 23} fontSize={10} fill="var(--muted)">n = {s.n}{flagSlice === s.id ? "  ⚠ regressed" : ""}</text>
        </g>
      ))}
      <g key={`grid-${animKey}`}>
        {sim.data.map((e) => {
          const [ax, ay] = cellXY(e.id, GA);
          const [bx, by] = cellXY(e.id, GB);
          const flag = flagSlice === e.slice;
          return (
            <g key={e.id}>
              {phase === "dataset" ? (
                <motion.g initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: (0.2 + e.id * 0.02) * k, duration: 0.3 * k }}>
                  <Cell x={ax} y={ay} ok={e.a} show="none" delay={0} k={k} />
                  <Cell x={bx} y={by} ok={e.b} show="none" delay={0} k={k} />
                </motion.g>
              ) : (
                <>
                  <Cell x={ax} y={ay} ok={e.a} show={showA} delay={gradedAt(e.id)} k={k} />
                  <Cell x={bx} y={by} ok={e.b} show={showB} delay={gradedAt(e.id)} k={k} flag={flag} />
                </>
              )}
            </g>
          );
        })}
      </g>
      <Aggregates sim={sim} phase={phase} k={k} animKey={animKey} />
      <line x1={0} x2={W} y1={196} y2={196} stroke="var(--line)" />
      <g key={`bottom-${animKey}`}>
        {phase === "dataset" && <Composition sim={sim} k={k} />}
        {(phase === "runA" || phase === "runB") && <TokenFlow sim={sim} model={phase === "runA" ? "a" : "b"} k={k} />}
        {phase === "aggregate" && <DiffAxis sim={sim} k={k} />}
        {phase === "resample" && <PickStrip sim={sim} k={k} />}
        {phase === "bootstrap" && <Histogram sim={sim} k={k} />}
        {phase === "slices" && <Forest sim={sim} k={k} />}
        {phase === "reweight" && <Reweight sim={sim} k={k} />}
      </g>
    </svg>
  );
}

function Aggregates({ sim, phase, k, animKey }: { sim: OfflineSim; phase: Phase; k: number; animKey: string }) {
  const o = order(phase);
  const X = 470, BW = 190;
  const last = gradedAt(sim.data.length - 1);
  const rows = [
    { label: "A accuracy", v: sim.accA, from: o === order("runA") ? 0 : sim.accA, visible: o >= order("runA") },
    { label: "B accuracy", v: sim.accB, from: o === order("runB") ? 0 : sim.accB, visible: o >= order("runB") },
  ];
  return (
    <g>
      <text x={X} y={20} fontSize={11} fontWeight={600} fill="var(--fg)">Aggregate (all {sim.data.length})</text>
      {rows.map((r, i) => r.visible && (
        <g key={`${r.label}-${animKey}`} transform={`translate(${X} ${40 + i * 52})`}>
          <text y={10} fontSize={11} fill="var(--muted)">{r.label}</text>
          <rect y={16} width={BW} height={16} rx={3} fill="var(--surface-2)" />
          <motion.rect y={16} height={16} rx={3} fill={i ? "var(--violet)" : "var(--accent)"} initial={{ width: r.from * BW }} animate={{ width: r.v * BW }}
            transition={{ delay: TOK.start * k, duration: (last - TOK.start) * k, ease: "linear" }} />
          <text x={BW} y={10} textAnchor="end" fontSize={12} fontWeight={600} fontFamily="var(--font-mono)" fill="var(--fg)">
            <AnimatedNumber svg value={r.v * 100} from={r.from * 100} digits={1} delay={TOK.start * k} duration={(last - TOK.start) * k} />%
          </text>
        </g>
      ))}
      {o >= order("aggregate") && (
        <motion.g key={`d-${animKey}`} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: (o === order("aggregate") ? 0.5 : 0) * k, duration: 0.4 * k }}>
          <text x={X} y={160} fontSize={13} fontWeight={700} fontFamily="var(--font-mono)" fill={sim.accB >= sim.accA ? "var(--ok)" : "var(--err)"}>Δ = {pts(sim.accB - sim.accA)}</text>
          {o >= order("bootstrap") && <text x={X} y={178} fontSize={10} fontFamily="var(--font-mono)" fill="var(--muted)">95% CI [{pts(sim.boot.ci[0])}, {pts(sim.boot.ci[1])}]</text>}
        </motion.g>
      )}
    </g>
  );
}

const BY = 214; // top of the bottom region
/** Start offset of each segment when laid end to end. */
const offsets = (ws: number[]) => ws.map((_, i) => ws.slice(0, i).reduce((a, b) => a + b, 0));

function Composition({ sim, k }: { sim: OfflineSim; k: number }) {
  const total = sim.data.length;
  const colors = ["var(--accent)", "var(--violet)", "var(--warn)", "var(--muted)"];
  const starts = offsets(SLICES.map((s) => (s.n / total) * 640));
  return (
    <g transform={`translate(20 ${BY + 40})`}>
      <text y={-14} fontSize={11} fontWeight={600} fill="var(--fg)">Eval-set composition by slice</text>
      {SLICES.map((s, i) => {
        const w = (s.n / total) * 640, x0 = starts[i];
        return (
          <g key={s.id}>
            <motion.rect x={x0} y={0} height={34} fill={colors[i]} fillOpacity={0.75} initial={{ width: 0 }} animate={{ width: w - 2 }} transition={{ delay: (0.3 + i * 0.4) * k, duration: 0.5 * k, ease }} />
            <motion.text x={x0 + 4} y={52} fontSize={10.5} fill="var(--fg)" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: (0.6 + i * 0.4) * k }}>{s.name}</motion.text>
            <motion.text x={x0 + 4} y={66} fontSize={10} fontFamily="var(--font-mono)" fill="var(--muted)" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: (0.6 + i * 0.4) * k }}>{s.n} · {pct(s.n / total, 1)}</motion.text>
          </g>
        );
      })}
      <text y={110} fontSize={10.5} fill="var(--muted)">The labels were written before any model ran, so neither model can influence what counts as correct.</text>
    </g>
  );
}

function TokenFlow({ sim, model, k }: { sim: OfflineSim; model: "a" | "b"; k: number }) {
  const y = BY + 70;
  const pts3 = { src: 40, model: 250, grader: 420, bin: 590 };
  const okN = model === "a" ? sim.nA : sim.nB;
  const last = gradedAt(sim.data.length - 1);
  const box = (x: number, label: string, sub: string) => (
    <g transform={`translate(${x - 58} ${y - 26})`}>
      <rect width={116} height={52} rx={8} fill="var(--surface)" stroke="var(--line)" />
      <text x={58} y={22} textAnchor="middle" fontSize={11} fontWeight={600} fill="var(--fg)">{label}</text>
      <text x={58} y={38} textAnchor="middle" fontSize={9.5} fill="var(--muted)">{sub}</text>
    </g>
  );
  return (
    <g>
      <line x1={pts3.src} x2={pts3.grader} y1={y} y2={y} stroke="var(--line)" strokeDasharray="3 4" />
      {box(pts3.src + 20, "Dataset", `${sim.data.length} examples`)}
      {box(pts3.model, model === "a" ? "Model A" : "Model B", "generates an answer")}
      {box(pts3.grader, "Grader", "answer vs label")}
      {[{ ok: true, yy: y - 50, n: okN }, { ok: false, yy: y + 50, n: sim.data.length - okN }].map((b) => (
        <g key={String(b.ok)} transform={`translate(${pts3.bin - 40} ${b.yy - 20})`}>
          <rect width={92} height={40} rx={8} fill={b.ok ? "var(--ok-soft)" : "var(--err-soft)"} stroke={b.ok ? "var(--ok)" : "var(--err)"} />
          <text x={46} y={16} textAnchor="middle" fontSize={10} fill="var(--fg)">{b.ok ? "✓ correct" : "✗ wrong"}</text>
          <text x={46} y={32} textAnchor="middle" fontSize={13} fontWeight={700} fontFamily="var(--font-mono)" fill="var(--fg)">
            <AnimatedNumber svg value={b.n} from={0} digits={0} delay={gradedAt(0) * k} duration={(last - gradedAt(0)) * k} />
          </text>
        </g>
      ))}
      {sim.data.map((e) => {
        const ok = e[model];
        const binY = ok ? y - 50 : y + 50;
        return (
          <motion.g key={e.id} initial={{ x: pts3.src, y, opacity: 0 }}
            animate={{ x: [pts3.src, pts3.model, pts3.grader, pts3.bin - 46], y: [y, y, y, binY], opacity: [0, 1, 1, 0] }}
            transition={{ delay: tokenStart(e.id) * k, duration: TOK.travel * k, ease: "easeInOut", times: [0, 0.4, TOK.grade, 1] }}>
            <circle r={5} fill="var(--muted)" />
            <motion.circle r={5} fill={ok ? "var(--ok)" : "var(--err)"} initial={{ opacity: 0 }} animate={{ opacity: [0, 0, 1, 1] }}
              transition={{ delay: tokenStart(e.id) * k, duration: TOK.travel * k, times: [0, TOK.grade - 0.05, TOK.grade, 1] }} />
          </motion.g>
        );
      })}
    </g>
  );
}

/** x-axis helper for Δ in [lo, hi] over [x0, x1]. */
const axis = (lo: number, hi: number, x0: number, x1: number) => (v: number) => x0 + ((v - lo) / (hi - lo)) * (x1 - x0);

function DiffAxis({ sim, k }: { sim: OfflineSim; k: number }) {
  const d = sim.accB - sim.accA;
  const sx = axis(-0.3, 0.3, 60, 620);
  const y = BY + 90;
  return (
    <g>
      <text x={60} y={BY + 20} fontSize={11} fontWeight={600} fill="var(--fg)">Where the aggregate difference lands (percentage points)</text>
      <line x1={60} x2={620} y1={y} y2={y} stroke="var(--line)" strokeWidth={2} />
      {[-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3].map((t) => (
        <g key={t}><line x1={sx(t)} x2={sx(t)} y1={y - 5} y2={y + 5} stroke={t === 0 ? "var(--fg)" : "var(--line)"} strokeWidth={t === 0 ? 2 : 1} />
          <text x={sx(t)} y={y + 20} textAnchor="middle" fontSize={10} fontFamily="var(--font-mono)" fill="var(--muted)">{t > 0 ? "+" : ""}{Math.round(t * 100)}</text></g>
      ))}
      <text x={sx(-0.15)} y={y - 30} textAnchor="middle" fontSize={10} fill="var(--muted)">◀ A better</text>
      <text x={sx(0.15)} y={y - 30} textAnchor="middle" fontSize={10} fill="var(--muted)">B better ▶</text>
      <motion.g initial={{ x: sx(0), opacity: 0 }} animate={{ x: sx(Math.max(-0.3, Math.min(0.3, d))), opacity: 1 }} transition={{ delay: 0.3 * k, duration: 1 * k, ease }}>
        <circle cy={y} r={8} fill={d >= 0 ? "var(--ok)" : "var(--err)"} stroke="var(--surface)" strokeWidth={2} />
        <text y={y - 14} textAnchor="middle" fontSize={12} fontWeight={700} fontFamily="var(--font-mono)" fill="var(--fg)">{pts(d)}</text>
      </motion.g>
      <motion.text x={60} y={BY + 160} fontSize={10.5} fill="var(--muted)" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1.5 * k }}>
        A point estimate only. No error bar yet: we don't know how much this would move on a different 80 examples.
      </motion.text>
    </g>
  );
}

function PickStrip({ sim, k }: { sim: OfflineSim; k: number }) {
  const x0 = 20, colW = 640 / sim.data.length, base = BY + 150;
  // Draw order: replay the counts as a sequence of draws (deterministic order: example id, then repeat index).
  const draws: { id: number; level: number }[] = [];
  sim.picks.forEach((c, id) => { for (let l = 0; l < c; l++) draws.push({ id, level: l }); });
  const seq = draws.map((d, i) => ({ ...d, order: (i * 37) % draws.length }));
  return (
    <g>
      <text x={x0} y={BY + 18} fontSize={11} fontWeight={600} fill="var(--fg)">Resample #1: how many times each example was drawn (stacked dots)</text>
      <line x1={x0} x2={x0 + 640} y1={base} y2={base} stroke="var(--line)" />
      {sim.picks.map((c, id) => c === 0 && (
        <motion.text key={id} x={x0 + id * colW + colW / 2} y={base + 12} textAnchor="middle" fontSize={8} fill="var(--err)" initial={{ opacity: 0 }} animate={{ opacity: 1 }}
          transition={{ delay: (PICK.start + draws.length * PICK.gap) * k }}>×</motion.text>
      ))}
      {seq.map((d) => (
        <motion.circle key={`${d.id}-${d.level}`} cx={x0 + d.id * colW + colW / 2} r={3} fill="var(--accent)"
          initial={{ cy: BY + 30, opacity: 0 }} animate={{ cy: base - 5 - d.level * 7, opacity: 1 }}
          transition={{ delay: (PICK.start + d.order * PICK.gap) * k, duration: 0.35 * k, ease: "easeIn" }} />
      ))}
      <text x={x0} y={base + 30} fontSize={10} fill="var(--muted)">examples 1 → {sim.data.length} (slices in order) · × = never drawn ({sim.neverPicked}) · Δ* on this resample = {pts(sim.firstDiff)}</text>
    </g>
  );
}

function Histogram({ sim, k }: { sim: OfflineSim; k: number }) {
  const { hist, boot } = sim;
  const lo = Math.min(hist.x0, -0.02), hi = Math.max(hist.x1, 0.02);
  const sx = axis(lo, hi, 40, 640);
  const base = BY + 165, hMax = 120;
  const final = hist.counts[hist.counts.length - 1];
  const max = Math.max(1, ...final);
  const bw = (hist.x1 - hist.x0) / final.length;
  const times = HIST_FRACTIONS;
  return (
    <g>
      <text x={40} y={BY + 18} fontSize={11} fontWeight={600} fill="var(--fg)">Bootstrap distribution of Δ* (B − A)</text>
      <text x={640} y={BY + 18} textAnchor="end" fontSize={11} fontFamily="var(--font-mono)" fill="var(--muted)">
        resamples: <AnimatedNumber svg value={sim.params.resamples} from={0} digits={0} delay={HIST.start * k} duration={HIST.grow * k} />
      </text>
      <line x1={40} x2={640} y1={base} y2={base} stroke="var(--line)" />
      {final.map((_, b) => {
        const hs = hist.counts.map((c) => (c[b] / max) * hMax);
        const x = sx(hist.x0 + b * bw);
        return (
          <motion.rect key={b} x={x + 0.5} width={Math.max(1, sx(hist.x0 + (b + 1) * bw) - x - 1)} fill="var(--accent)" fillOpacity={0.7}
            initial={{ height: 0, y: base }} animate={{ height: hs, y: hs.map((h) => base - h) }} transition={{ delay: HIST.start * k, duration: HIST.grow * k, times, ease: "linear" }} />
        );
      })}
      <line x1={sx(0)} x2={sx(0)} y1={base - hMax - 10} y2={base + 4} stroke="var(--fg)" strokeWidth={1.5} />
      <text x={sx(0)} y={base + 16} textAnchor="middle" fontSize={10} fontFamily="var(--font-mono)" fill="var(--fg)">0</text>
      {boot.ci.map((c, i) => (
        <motion.g key={i} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: HIST.ci * k, duration: 0.4 * k }}>
          <line x1={sx(c)} x2={sx(c)} y1={base - hMax - 6} y2={base} stroke="var(--violet)" strokeWidth={2} strokeDasharray="4 3" />
          <text x={sx(c)} y={base + 16} textAnchor="middle" fontSize={10} fontFamily="var(--font-mono)" fill="var(--violet)">{pts(c)}</text>
        </motion.g>
      ))}
      <motion.text x={340} y={base + 34} textAnchor="middle" fontSize={10.5} fill="var(--fg)" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: (HIST.ci + 0.3) * k }}>
        {boot.ci[0] > 0 ? "95% CI excludes 0: the aggregate gain is unlikely to be sampling noise." : "95% CI includes 0: this difference could be noise."}
      </motion.text>
    </g>
  );
}

function Forest({ sim, k }: { sim: OfflineSim; k: number }) {
  const rows = [...sim.slices.map((s) => ({ name: s.name, n: s.n, d: s.diff, ci: s.ci, bad: s.regress })), { name: "All (aggregate)", n: sim.data.length, d: sim.accB - sim.accA, ci: sim.boot.ci, bad: false }];
  const lo = Math.min(-0.2, ...rows.map((r) => r.ci[0])) - 0.05, hi = Math.max(0.2, ...rows.map((r) => r.ci[1])) + 0.05;
  const sx = axis(lo, hi, 180, 590);
  return (
    <g>
      <text x={0} y={BY + 16} fontSize={11} fontWeight={600} fill="var(--fg)">Δ per slice with 95% bootstrap CI</text>
      <line x1={sx(0)} x2={sx(0)} y1={BY + 24} y2={BY + 190} stroke="var(--fg)" strokeWidth={1.2} />
      <text x={sx(0)} y={BY + 204} textAnchor="middle" fontSize={10} fontFamily="var(--font-mono)" fill="var(--muted)">0</text>
      {rows.map((r, i) => {
        const y = BY + 42 + i * 34;
        return (
          <motion.g key={r.name} initial={{ opacity: 0, x: -12 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: SLICE_AT(i) * k, duration: 0.4 * k, ease }}>
            <text x={0} y={y + 4} fontSize={11} fill={r.bad ? "var(--err)" : "var(--fg)"} fontWeight={r.bad || i === rows.length - 1 ? 700 : 400}>{r.bad ? "✗ " : ""}{r.name} (n={r.n})</text>
            <motion.line y1={y} y2={y} stroke={r.bad ? "var(--err)" : "var(--accent)"} strokeWidth={3} strokeLinecap="round"
              initial={{ x1: sx(r.d), x2: sx(r.d) }} animate={{ x1: sx(r.ci[0]), x2: sx(r.ci[1]) }} transition={{ delay: (SLICE_AT(i) + 0.2) * k, duration: 0.5 * k, ease }} />
            <rect x={sx(r.d) - 5} y={y - 5} width={10} height={10} fill={r.bad ? "var(--err)" : i === rows.length - 1 ? "var(--fg)" : "var(--accent)"} />
            <text x={680} y={y + 4} textAnchor="end" fontSize={10.5} fontWeight={600} fontFamily="var(--font-mono)" fill={r.bad ? "var(--err)" : "var(--fg)"}>{pts(r.d, 0)}</text>
          </motion.g>
        );
      })}
    </g>
  );
}

function Reweight({ sim, k }: { sim: OfflineSim; k: number }) {
  const colors = ["var(--accent)", "var(--violet)", "var(--warn)", "var(--muted)"];
  const mix = (ws: number[], y: number, label: string, delay: number) => {
    const starts = offsets(ws.map((w) => w * 400)).map((v) => v + 130);
    return (
      <g>
        <text x={0} y={y + 14} fontSize={11} fill="var(--fg)">{label}</text>
        {ws.map((w, i) => (
          <motion.rect key={i} x={starts[i]} y={y} height={20} fill={colors[i]} fillOpacity={0.75} initial={{ width: 0 }} animate={{ width: Math.max(0, w * 400 - 1) }} transition={{ delay: (delay + i * 0.15) * k, duration: 0.4 * k }} />
        ))}
        <text x={540} y={y + 14} fontSize={10} fontFamily="var(--font-mono)" fill="var(--muted)">Non-Eng {pct(ws[3], 0)}</text>
      </g>
    );
  };
  const res = [{ l: "A, production mix", v: sim.prodA, c: "var(--accent)" }, { l: "B, production mix", v: sim.prodB, c: "var(--violet)" }];
  return (
    <g transform={`translate(0 ${BY + 10})`}>
      <text x={0} y={6} fontSize={11} fontWeight={600} fill="var(--fg)">Slice mix: eval set vs production traffic</text>
      {mix(sim.evalWeights, 18, "Eval set", 0.2)}
      {mix(sim.prodWeights, 48, "Production", 0.8)}
      {res.map((r, i) => (
        <g key={r.l} transform={`translate(0 ${100 + i * 34})`}>
          <text x={0} y={14} fontSize={11} fill="var(--fg)">{r.l}</text>
          <rect x={130} y={2} width={400} height={18} rx={3} fill="var(--surface-2)" />
          <motion.rect x={130} y={2} height={18} rx={3} fill={r.c} initial={{ width: 0 }} animate={{ width: r.v * 400 }} transition={{ delay: 1.6 * k, duration: 0.9 * k, ease }} />
          <text x={540} y={15} fontSize={12} fontWeight={700} fontFamily="var(--font-mono)" fill="var(--fg)"><AnimatedNumber svg value={r.v * 100} from={0} digits={1} delay={1.6 * k} duration={0.9 * k} />%</text>
        </g>
      ))}
      <motion.text x={0} y={180} fontSize={10.5} fill={sim.prodB < sim.prodA ? "var(--err)" : "var(--ok)"} fontWeight={600} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 2.6 * k }}>
        {sim.prodB < sim.prodA ? "✗ On real traffic B is worse: do not ship on the aggregate alone." : "✓ The win survives the production mix."}
      </motion.text>
    </g>
  );
}

/* ---------- card under the canvas ---------- */
function StepCard({ sim, phase, k }: { sim: OfflineSim; phase: Phase; k: number }) {
  if (phase === "dataset" || phase === "runA" || phase === "runB") {
    const cols = phase === "dataset" ? 0 : phase === "runA" ? 1 : 2;
    return (
      <div className="overflow-x-auto">
        <table className="w-full min-w-[320px] text-xs">
          <thead className="text-left text-muted"><tr><th className="py-1">slice</th><th>n</th>{cols >= 1 && <th>A correct</th>}{cols >= 2 && <th>B correct</th>}{cols >= 2 && <th>Δ</th>}</tr></thead>
          <tbody className="font-mono">
            {sim.slices.map((s, i) => {
              const end = gradedAt(sliceStart[i] + s.n - 1), start = gradedAt(sliceStart[i]);
              return (
                <motion.tr key={s.id} className="border-t border-line" initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: (phase === "dataset" ? 0.2 + i * 0.3 : 0) * k }}>
                  <td className="py-1 font-sans">{s.name}</td><td>{s.n}</td>
                  {cols >= 1 && <td>{cols === 1 ? <AnimatedNumber value={s.accA * 100} from={0} digits={0} delay={start * k} duration={(end - start) * k} /> : (s.accA * 100).toFixed(0)}%</td>}
                  {cols >= 2 && <td><AnimatedNumber value={s.accB * 100} from={0} digits={0} delay={start * k} duration={(end - start) * k} />%</td>}
                  {cols >= 2 && <motion.td className={s.diff < 0 ? "text-err" : "text-ok"} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: end * k }}>{s.diff < 0 ? "▼" : "▲"} {pts(s.diff, 0)}</motion.td>}
                </motion.tr>
              );
            })}
          </tbody>
        </table>
        <p className="mt-2 text-xs text-muted">{phase === "dataset" ? "Slices are chosen up front (language, length, domain). Small slices are where regressions hide." : "Per-slice accuracy fills in as the grader reaches each slice's examples."}</p>
      </div>
    );
  }
  if (phase === "aggregate") {
    return (
      <div className="grid gap-3 sm:grid-cols-3">
        {[["Model A", sim.accA, sim.nA], ["Model B", sim.accB, sim.nB]].map(([l, v, n]) => (
          <div key={String(l)} className="rounded-md bg-surface-2 p-2"><div className="text-xs text-muted">{l}</div>
            <div className="font-mono text-xl"><AnimatedNumber value={Number(v) * 100} from={0} digits={1} delay={0.2 * k} duration={0.8 * k} />%</div>
            <div className="text-[11px] text-muted">{n}/{sim.data.length} correct</div></div>
        ))}
        <div className="rounded-md bg-surface-2 p-2"><div className="text-xs text-muted">Δ (B − A)</div>
          <div className={cx("font-mono text-xl", sim.accB >= sim.accA ? "text-ok" : "text-err")}>{pts(sim.accB - sim.accA)}</div>
          <div className="text-[11px] text-muted">= {sim.nB - sim.nA} more examples right</div></div>
      </div>
    );
  }
  if (phase === "resample") {
    const buckets = [0, 1, 2, 3].map((c) => sim.picks.filter((p) => (c === 3 ? p >= 3 : p === c)).length);
    return (
      <div>
        <p className="mb-2 text-xs">How often each of the {sim.data.length} examples was drawn in resample #1:</p>
        <div className="space-y-1">
          {buckets.map((b, i) => (
            <div key={i} className="grid grid-cols-[80px_minmax(0,1fr)_40px] items-center gap-2 text-xs">
              <span className="font-mono">{i === 3 ? "3+ times" : `${i} time${i === 1 ? "" : "s"}`}</span>
              <div className="h-4 rounded bg-surface-2"><motion.div className={cx("h-4 rounded", i === 0 ? "bg-err/60" : "bg-accent/60")} initial={{ width: 0 }} animate={{ width: `${(b / sim.data.length) * 100}%` }} transition={{ delay: (0.4 + i * 0.3) * k, duration: 0.5 * k, ease }} /></div>
              <span className="text-right font-mono">{b}</span>
            </div>
          ))}
        </div>
        <p className="mt-2 text-xs text-muted">With replacement, about 1 − (1 − 1/n)ⁿ ≈ 63% of examples appear at least once, so ≈37% are left out. That variation is what simulates "a different test set".</p>
      </div>
    );
  }
  if (phase === "bootstrap") {
    const excl = sim.boot.ci[0] > 0 || sim.boot.ci[1] < 0;
    return (
      <div className="space-y-2 text-xs">
        <div className="relative h-8 rounded bg-surface-2">
          <motion.div className="absolute inset-y-1 rounded bg-violet/40" style={{ left: `${Math.max(0, Math.min(100, (sim.boot.ci[0] + 0.5) * 100))}%` }} initial={{ width: 0 }}
            animate={{ width: `${(sim.boot.ci[1] - sim.boot.ci[0]) * 100}%` }} transition={{ delay: HIST.ci * k, duration: 0.5 * k }} />
          <div className="absolute inset-y-0 left-1/2 w-0.5 bg-fg/60" />
          <span className="absolute top-1/2 left-1/2 ml-1 -translate-y-1/2 text-[10px] text-muted">0</span>
        </div>
        <p>Sorted Δ* values: the 2.5th percentile is <span className="font-mono">{pts(sim.boot.ci[0])}</span> and the 97.5th is <span className="font-mono">{pts(sim.boot.ci[1])}</span>.</p>
        <Callout tone={excl ? "ok" : "warn"}>{excl ? "The interval excludes 0, so the aggregate difference is statistically solid. That still says nothing about individual slices." : "The interval includes 0: the aggregate difference may be noise. More examples would narrow it."}</Callout>
      </div>
    );
  }
  if (phase === "slices") {
    return (
      <div className="overflow-x-auto">
        <table className="w-full min-w-[380px] text-xs">
          <thead className="text-left text-muted"><tr><th className="py-1">slice</th><th>A</th><th>B</th><th>Δ</th><th>95% CI</th><th>verdict</th></tr></thead>
          <tbody className="font-mono">
            {sim.slices.map((s, i) => (
              <motion.tr key={s.id} className={cx("border-t border-line", s.regress && "bg-err-soft")} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: SLICE_AT(i) * k }}>
                <td className="py-1 font-sans">{s.name}</td><td>{pct(s.accA, 0)}</td><td>{pct(s.accB, 0)}</td><td>{pts(s.diff, 0)}</td>
                <td>[{pts(s.ci[0], 0)}, {pts(s.ci[1], 0)}]</td>
                <td>{s.regress ? <Badge tone="err"><XCircle size={11} /> regressed</Badge> : s.ci[0] > 0 ? <Badge tone="ok"><CheckCircle2 size={11} /> improved</Badge> : <Badge>unclear</Badge>}</td>
              </motion.tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[380px] text-xs">
        <thead className="text-left text-muted"><tr><th className="py-1">slice</th><th>eval weight</th><th>prod weight</th><th>A</th><th>B</th></tr></thead>
        <tbody className="font-mono">
          {sim.slices.map((s, i) => (
            <motion.tr key={s.id} className="border-t border-line" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: (0.3 + i * 0.2) * k }}>
              <td className="py-1 font-sans">{s.name}</td><td>{pct(sim.evalWeights[i])}</td><td>{pct(sim.prodWeights[i])}</td><td>{pct(s.accA, 0)}</td><td>{pct(s.accB, 0)}</td>
            </motion.tr>
          ))}
          <tr className="border-t border-line font-semibold"><td className="py-1 font-sans">weighted total</td><td>{pct(sim.accA)} / {pct(sim.accB)}</td><td>{pct(sim.prodA)} / {pct(sim.prodB)}</td><td /><td /></tr>
        </tbody>
      </table>
    </div>
  );
}

function Calculation({ sim, phase }: { sim: OfflineSim; phase: Phase }) {
  const f = (v: number) => v.toFixed(4);
  switch (phase) {
    case "dataset": return <TexBlock>{`N = ${SLICES.map((s) => s.n).join(" + ")} = ${sim.data.length}`}</TexBlock>;
    case "runA": return <TexBlock>{`\\text{acc}_A = \\frac{1}{N}\\sum_{i} [\\hat{y}^A_i = y_i] = \\frac{${sim.nA}}{${sim.data.length}} = ${f(sim.accA)}`}</TexBlock>;
    case "runB": return <TexBlock>{`\\text{acc}_B = \\frac{${sim.nB}}{${sim.data.length}} = ${f(sim.accB)}`}</TexBlock>;
    case "aggregate": return <TexBlock>{`\\Delta = \\text{acc}_B - \\text{acc}_A = ${f(sim.accB)} - ${f(sim.accA)} = ${f(sim.accB - sim.accA)}`}</TexBlock>;
    case "resample": return <TexBlock>{`\\begin{aligned} i^{*}_1,\\dots,i^{*}_{N} &\\sim \\text{Uniform}\\{1..N\\} \\\\ \\Delta^{*} &= \\tfrac{1}{N}\\textstyle\\sum_j (b_{i^*_j} - a_{i^*_j}) = ${f(sim.firstDiff)} \\\\ P(\\text{never drawn}) &= (1-\\tfrac{1}{N})^{N} \\approx ${Math.pow(1 - 1 / sim.data.length, sim.data.length).toFixed(3)} \\end{aligned}`}</TexBlock>;
    case "bootstrap": return <TexBlock>{`\\begin{aligned} \\text{CI}_{95} &= \\big[q_{0.025}(\\Delta^{*}),\\, q_{0.975}(\\Delta^{*})\\big] \\\\ &= [${f(sim.boot.ci[0])},\\, ${f(sim.boot.ci[1])}] \\end{aligned}`}</TexBlock>;
    case "slices": {
      const s = sim.worst;
      return <TexBlock>{`\\begin{aligned} \\Delta_{\\text{${s.name}}} &= ${f(s.accB)} - ${f(s.accA)} = ${f(s.diff)} \\\\ \\text{CI}_{95} &= [${f(s.ci[0])},\\, ${f(s.ci[1])}] \\\\ \\Delta_{\\text{all}} &= \\textstyle\\sum_s \\tfrac{n_s}{N} \\Delta_s = ${f(sim.accB - sim.accA)} \\end{aligned}`}</TexBlock>;
    }
    case "reweight": {
      const v = (xs: number[]) => `(${xs.map((x) => x.toFixed(2)).join(",\\,")})`;
      return <TexBlock>{`\\begin{aligned} \\text{acc}^{\\text{prod}} &= \\textstyle\\sum_s w^{\\text{prod}}_s\\, \\text{acc}_s \\\\ w^{\\text{prod}} &= ${v(sim.prodWeights)} \\\\ A &= w \\cdot ${v(sim.slices.map((s) => s.accA))} = ${f(sim.prodA)} \\\\ B &= w \\cdot ${v(sim.slices.map((s) => s.accB))} = ${f(sim.prodB)} \\end{aligned}`}</TexBlock>;
    }
  }
}
