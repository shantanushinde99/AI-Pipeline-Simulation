import { useMemo, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { RotateCcw } from "lucide-react";
import { usePlayback } from "../../hooks/usePlayback";
import { PlaybackControls } from "../../components/visualizer/PlaybackControls";
import { EventLog } from "../../components/visualizer/EventLog";
import { StepExplainer, Workspace } from "../../components/visualizer/Workspace";
import { AnimatedNumber } from "../../components/visualizer/AnimatedNumber";
import { Button, Callout, KV, Panel, Select, SimulatedLabel, Slider, Toggle, cx } from "../../components/ui";
import { TexBlock } from "../../components/Tex";
import type { VisualizerTab } from "../types";
import { MetricBar, StepCaption, ease, pct, useTicks } from "../classification-metrics/kit";
import { DEFAULT_CM, FIRST, simulate, type CmConfig, type CmMode, type CmPhase, type CmSim } from "./simulation";

/* ---------- one timing schedule (seconds at 1×) ---------- */
const FLIGHT = 0.8;
const landAt = (phase: CmPhase, j: number, n: number) => (phase === "first" ? 0.5 + j * 1.4 : 0.3 + j * Math.min(0.2, 3.2 / Math.max(1, n - FIRST)));
const sweepAt = (c: number) => 0.3 + c * 1.1;
const DURATIONS: Record<CmPhase, number> = { list: 3000, first: 5800, rest: 5200, diagonal: 3600, precision: 4600, recall: 4600, normalize: 3800, cost: 4200 };
type Display = "counts" | "row";

const SHORT: Record<string, string> = { urgent: "URG", "not urgent": "NOT", routine: "RTN", "self-care": "SELF" };

const NARRATE: Record<CmPhase, (s: CmSim) => { headline: string; sub: string }> = {
  list: (s) => ({ headline: `${s.n} labelled messages wait to be sorted`, sub: "Each chip shows its id, its true class and the model's prediction (actual → predicted)." }),
  first: () => ({ headline: `Watch the first ${FIRST} fly into the grid`, sub: "Row = what it really was. Column = what the model said. Same row and column = correct." }),
  rest: (s) => ({ headline: `The other ${s.n - FIRST} examples land; counters tick up`, sub: "Every example ends up in exactly one cell, so the cells always add up to the total." }),
  diagonal: (s) => ({ headline: `${s.correct} of ${s.n} landed on the diagonal: accuracy ${pct(s.accuracy)}`, sub: "Off-diagonal cells are mistakes, and their position tells you which mistake." }),
  precision: () => ({ headline: "Precision: read down each column", sub: "Of everything the model put in this column, how much truly belongs there?" }),
  recall: () => ({ headline: "Recall: read across each row", sub: "Of everything that truly is this class, how much did the model find?" }),
  normalize: () => ({ headline: "Divide each row by its total", sub: "Rare classes become comparable with common ones. The diagonal becomes per-class recall." }),
  cost: (s) => ({ headline: `Weighted by cost, these mistakes cost ${s.total}`, sub: "A missed urgent message is far worse than a false alarm. Accuracy treats them the same." }),
};
const WHY: Record<CmPhase, string> = {
  list: "A confusion matrix is just a tally of (actual, predicted) pairs. Nothing more.",
  first: "Agreeing on 'row = actual, column = predicted' avoids the most common misreading.",
  rest: "Counts are the raw evidence; every metric is a ratio over some of these cells.",
  diagonal: "Accuracy hides which classes are confused with which.",
  precision: "Low precision for a class means its alerts are often wrong.",
  recall: "Low recall for a critical class means real cases are slipping through.",
  normalize: "Raw counts are dominated by big classes; normalized rows show per-class behaviour.",
  cost: "Choosing a model or threshold should minimize real cost, not maximize accuracy.",
};

export default function ConfusionMatricesVisualizer({ tab }: { tab: VisualizerTab }) {
  const [cfg, setCfg] = useState(DEFAULT_CM);
  const [display, setDisplay] = useState<Display>("counts");
  const [replay, setReplay] = useState(0);
  const set = (p: Partial<CmConfig>) => setCfg((c) => ({ ...c, ...p }));
  const sim = useMemo(() => simulate(cfg), [cfg]);
  const pb = usePlayback(sim.steps.length, (i) => DURATIONS[sim.steps[i]?.state.phase ?? "list"]);
  const step = sim.steps[pb.index];
  const phase = step.state.phase;
  const k = 1 / pb.speed;
  const animKey = `${step.id}-${replay}-${cfg.mode}`;
  if (tab === "metrics") return null;
  const story = NARRATE[phase](sim);
  const binary = cfg.mode === "binary";

  return (
    <Workspace
      main={
        <>
          <Panel title="Examples and matrix" actions={<SimulatedLabel>Synthetic predictions</SimulatedLabel>}>
            <StepCaption index={pb.index} total={sim.steps.length} stepId={step.id} headline={story.headline} sub={story.sub} onReplay={() => setReplay((r) => r + 1)} />
            <div className="overflow-x-auto">
              <div className="min-w-[600px]">
                <MatrixCanvas sim={sim} phase={phase} k={k} animKey={animKey} display={display} landedBefore={step.state.landedBefore} landedAfter={step.state.landedAfter} binary={binary} />
              </div>
            </div>
            <div className="mt-1 flex flex-wrap gap-3 text-[11px] text-muted">
              <span className="flex items-center gap-1"><svg width="10" height="10"><circle cx="5" cy="5" r="4" fill="var(--ok)" /></svg>landed on the diagonal (correct)</span>
              <span className="flex items-center gap-1"><svg width="10" height="10"><rect x="1" y="1" width="8" height="8" fill="var(--err)" /></svg>landed off the diagonal (error)</span>
              <span>rows = actual class, columns = predicted class</span>
            </div>
          </Panel>
          <Panel title="What's happening in this step">
            <AnimatePresence mode="wait" initial={false}>
              <motion.div key={animKey} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: 0.2 }}>
                <StepCard sim={sim} phase={phase} k={k} animKey={animKey} cfg={cfg} />
              </motion.div>
            </AnimatePresence>
          </Panel>
          <PlaybackControls pb={pb} stepTitles={sim.steps.map((s) => s.title)} />
          <Panel title="Parameters" actions={<Button size="sm" variant="ghost" onClick={() => { setCfg(DEFAULT_CM); setDisplay("counts"); }}><RotateCcw size={12} /> Reset</Button>}>
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              <Select label="Classes" value={cfg.mode} options={[{ value: "binary" as CmMode, label: "2 classes (urgent / not)" }, { value: "multi" as CmMode, label: "3 classes (urgent / routine / self-care)" }]} onChange={(v) => set({ mode: v, n: v === "binary" ? 20 : 24 })} />
              <Select label="Cell values" value={display} options={[{ value: "counts" as Display, label: "Counts" }, { value: "row" as Display, label: "Row-normalized (share of actual class)" }]} onChange={setDisplay} />
              <Slider label="Examples" value={cfg.n} min={8} max={36} step={1} onChange={(v) => set({ n: v })} />
              <Slider label="Model quality (chance a prediction is right)" value={cfg.quality} min={0} max={1} step={0.05} format={(v) => `${Math.round(v * 100)}%`} onChange={(v) => set({ quality: v })} />
              <Slider label="Cost of a missed urgent message" value={cfg.costMiss} min={1} max={50} onChange={(v) => set({ costMiss: v })} />
              <Slider label="Cost of any other error" value={cfg.costOther} min={0} max={10} onChange={(v) => set({ costOther: v })} />
              <Slider label="Random seed" value={cfg.seed} min={1} max={40} onChange={(v) => set({ seed: v })} />
              <div className="space-y-2 pt-1">
                <Toggle label="Imbalanced classes" checked={cfg.imbalanced} onChange={(v) => set({ imbalanced: v })} hint="Urgent becomes rare (≈12–15%)." />
                <Toggle label="Break it: model always predicts the majority class" checked={cfg.majorityFault} onChange={(v) => set({ majorityFault: v })} />
              </div>
            </div>
          </Panel>
        </>
      }
      inspector={
        <>
          <StepExplainer index={pb.index} title={step.title} what={step.description} why={WHY[phase]} />
          <Panel title="Calculation"><Calculation sim={sim} phase={phase} /></Panel>
          <Panel title="Summary">
            <KV rows={[
              ["accuracy", pct(sim.accuracy)],
              ["macro F1", pct(sim.macroF1)],
              ...sim.names.map((nm, c) => [`${nm}: P / R`, `${pct(sim.pc[c].precision, 0)} / ${pct(sim.pc[c].recall, 0)}`] as [string, string]),
              ["total cost", String(sim.total)],
            ]} />
          </Panel>
          <Panel title="Events"><EventLog events={step.events} /></Panel>
        </>
      }
    />
  );
}

/* ---------- canvas ---------- */
const W = 700, MY0 = 56, MX0 = 262, GRID = 360, CHIP_W = 104, CHIP_H = 17, PITCH = 20, LIST_Y = 44;

function MatrixCanvas({ sim, phase, k, animKey, display, landedBefore, landedAfter, binary }: {
  sim: CmSim; phase: CmPhase; k: number; animKey: string; display: Display; landedBefore: number; landedAfter: number; binary: boolean;
}) {
  const { K, n, names, m } = sim;
  const CS = GRID / K;
  const cols = n > 18 ? 2 : 1;
  const perCol = Math.ceil(n / cols);
  const H = Math.max(MY0 + GRID + 36, LIST_Y + perCol * PITCH + 10);
  const listPos = (i: number) => ({ x: 10 + Math.floor(i / perCol) * (CHIP_W + 8), y: LIST_Y + (i % perCol) * PITCH });
  // Each example's slot inside its cell, in arrival order.
  const slots = useMemo(() => {
    const seen = new Map<string, number>();
    return sim.ex.map((e) => {
      const key = `${e.actual}-${e.predicted}`;
      const j = seen.get(key) ?? 0;
      seen.set(key, j + 1);
      const per = Math.floor((CS - 24) / 12);
      return { x: MX0 + e.predicted * CS + 14 + (j % per) * 12, y: MY0 + e.actual * CS + 52 + Math.floor(j / per) * 12 };
    });
  }, [sim.ex, CS]);
  const filling = phase === "list" || phase === "first" || phase === "rest";
  const showRow = phase === "normalize" || (display === "row" && (phase === "diagonal" || phase === "precision" || phase === "recall"));
  const rowSum = m.map((r) => r.reduce((a, b) => a + b, 0));
  const colSum = m[0].map((_, c) => m.reduce((s, r) => s + r[c], 0));
  const maxCount = Math.max(1, ...m.flat());
  const outcome = (a: number, p: number) => (!binary ? null : a === 0 ? (p === 0 ? "TP" : "FN") : p === 0 ? "FP" : "TN");
  const offDiag = m.flatMap((row, a) => row.map((_, p) => [a, p] as const)).filter(([a, p]) => a !== p);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label={`${K} by ${K} confusion matrix filled from ${n} examples`}>
      <text x={10} y={26} fontSize={11} fontWeight={600} fill="var(--muted)">EXAMPLES (actual → predicted)</text>
      <text x={MX0 + GRID / 2} y={18} textAnchor="middle" fontSize={11} fontWeight={600} fill="var(--muted)">PREDICTED →</text>
      <text x={MX0 - 26} y={MY0 + GRID / 2} textAnchor="middle" fontSize={11} fontWeight={600} fill="var(--muted)" transform={`rotate(-90 ${MX0 - 26} ${MY0 + GRID / 2})`}>ACTUAL →</text>
      {names.map((nm, c) => (
        <g key={nm} fontSize={11} fill="var(--fg)">
          <text x={MX0 + c * CS + CS / 2} y={MY0 - 10} textAnchor="middle">{nm}</text>
          <text x={MX0 - 8} y={MY0 + c * CS + CS / 2} textAnchor="end" transform={`rotate(-90 ${MX0 - 8} ${MY0 + c * CS + CS / 2})`} dominantBaseline="auto">{nm}</text>
        </g>
      ))}

      {/* cells */}
      {m.map((row, a) => row.map((v, p) => {
        const diag = a === p;
        const val = showRow ? sim.norm[a][p] : v;
        const shade = showRow ? (Number.isFinite(val) ? val : 0) : v / maxCount;
        const x = MX0 + p * CS, y = MY0 + a * CS;
        const hl = phase === "diagonal" && diag;
        return (
          <g key={`${a}-${p}`}>
            <rect x={x + 2} y={y + 2} width={CS - 4} height={CS - 4} rx={8} fill="var(--surface-2)" stroke="var(--line)" />
            <motion.rect x={x + 2} y={y + 2} width={CS - 4} height={CS - 4} rx={8} fill={diag ? "var(--ok)" : "var(--err)"} initial={false}
              animate={{ fillOpacity: filling ? 0.04 : 0.06 + 0.3 * shade }} transition={{ duration: 0.6 * k, delay: phase === "normalize" ? 1.4 * k : 0 }} />
            {hl && (
              <motion.rect key={`hl-${animKey}`} x={x + 2} y={y + 2} width={CS - 4} height={CS - 4} rx={8} fill="none" stroke="var(--ok)" strokeWidth={3}
                initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ delay: (0.3 + a * 0.4) * k, duration: 0.5 * k }} />
            )}
            {outcome(a, p) && <text x={x + CS - 10} y={y + 18} textAnchor="end" fontSize={11} fontWeight={600} fill="var(--muted)">{outcome(a, p)}</text>}
            {phase === "normalize" ? (
              <NormalizeText key={animKey} x={x + 12} y={y + 34} count={v} total={rowSum[a]} frac={sim.norm[a][p]} k={k} />
            ) : (
              <text x={x + 12} y={y + 34} fontSize={22} fontWeight={700} fontFamily="var(--font-mono)" fill="var(--fg)">
                {filling ? (
                  <CellCount key={animKey} before={sim.ex.slice(0, landedBefore).filter((e) => e.actual === a && e.predicted === p).length}
                    times={sim.ex.slice(landedBefore, landedAfter).map((e, j) => (e.actual === a && e.predicted === p ? (landAt(phase, j, n) + FLIGHT) * k : -1)).filter((t) => t >= 0)} resetKey={animKey} />
                ) : showRow ? (Number.isFinite(val) ? val.toFixed(2) : "–") : v}
              </text>
            )}
            {phase === "cost" && !diag && (
              <motion.g key={`c-${animKey}`} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: (0.3 + offDiag.findIndex(([aa, pp]) => aa === a && pp === p) * 0.3) * k, duration: 0.4 * k }}>
                <rect x={x + 10} y={y + CS - 34} width={CS - 20} height={22} rx={11} fill={sim.cost[a][p] > 1 ? "var(--err)" : "var(--fg)"} />
                <text x={x + CS / 2} y={y + CS - 19} textAnchor="middle" fontSize={11} fill="var(--surface)" fontFamily="var(--font-mono)">{v} × {sim.cost[a][p]} = {v * sim.cost[a][p]}</text>
              </motion.g>
            )}
          </g>
        );
      }))}

      {/* precision column / recall row sweep */}
      {(phase === "precision" || phase === "recall") && (() => {
        const isP = phase === "precision";
        const pos = names.map((_, c) => (isP ? MX0 + c * CS : MY0 + c * CS));
        const kf = pos.flatMap((p) => [p, p]);
        const times = names.flatMap((_, c) => [c / K, (c + 0.75) / K]);
        return (
          <g key={`sweep-${animKey}`}>
            <motion.rect {...(isP ? { y: MY0 - 2, width: CS, height: GRID + 4 } : { x: MX0 - 2, width: GRID + 4, height: CS })} rx={10} fill="none" stroke="var(--accent)" strokeWidth={3}
              initial={isP ? { x: pos[0], opacity: 0 } : { y: pos[0], opacity: 0 }} animate={isP ? { x: kf, opacity: 1 } : { y: kf, opacity: 1 }}
              transition={{ delay: 0.3 * k, duration: K * 1.1 * k, times, ease: "easeInOut", opacity: { delay: 0.3 * k, duration: 0.2 * k } }} />
            {names.map((_, c) => {
              const v = isP ? sim.pc[c].precision : sim.pc[c].recall;
              const frac = isP ? `${sim.pc[c].tp}/${colSum[c]}` : `${sim.pc[c].tp}/${rowSum[c]}`;
              const val = Number.isFinite(v) ? v.toFixed(2) : "undef.";
              return (
                <motion.text key={c} x={isP ? MX0 + c * CS + CS / 2 : MX0 + GRID + 8} y={isP ? MY0 + GRID + 22 : MY0 + c * CS + CS / 2 - 6} textAnchor={isP ? "middle" : "start"}
                  fontSize={11} fontWeight={600} fontFamily="var(--font-mono)" fill="var(--accent)" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: (sweepAt(c) + 0.4) * k, duration: 0.3 * k }}>
                  {isP ? `P = ${frac} = ${val}` : <><tspan>R = {frac}</tspan><tspan x={MX0 + GRID + 8} dy={15}>= {val}</tspan></>}
                </motion.text>
              );
            })}
          </g>
        );
      })()}
      {phase === "normalize" && names.map((_, a) => (
        <motion.text key={`rs-${a}-${animKey}`} x={MX0 + GRID + 8} y={MY0 + a * CS + CS / 2} fontSize={11} fontFamily="var(--font-mono)" fill="var(--accent)"
          initial={{ opacity: 0, x: 24 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: (0.3 + a * 0.25) * k, duration: 0.4 * k }}>
          ÷{rowSum[a]}
        </motion.text>
      ))}

      {/* examples: list chips, flights, landed dots */}
      {sim.ex.map((e, i) => {
        const L = listPos(i);
        const s = slots[i];
        const ok = e.actual === e.predicted;
        const landed = i < landedBefore || (!filling && true);
        const flying = filling && i >= landedBefore && i < landedAfter;
        const label = `#${String(e.id).padStart(2, "0")} ${SHORT[names[e.actual]]}→${SHORT[names[e.predicted]]}`;
        const chip = (
          <>
            <rect x={0} y={-CHIP_H / 2} width={CHIP_W} height={CHIP_H} rx={4} fill="var(--surface)" stroke="var(--line)" />
            <text x={6} y={4} fontSize={10.5} fontFamily="var(--font-mono)" fill="var(--fg)">{label}</text>
          </>
        );
        const dot = ok ? <circle r={4.5} fill="var(--ok)" /> : <rect x={-4} y={-4} width={8} height={8} fill="var(--err)" />;
        if (flying) {
          const j = i - landedBefore;
          const at = landAt(phase, j, n);
          return (
            <g key={`${e.id}-${animKey}`}>
              <g transform={`translate(${L.x} ${L.y})`} opacity={0.25}>{chip}</g>
              <motion.g initial={{ x: L.x, y: L.y }} animate={{ x: s.x, y: s.y }} transition={{ delay: at * k, duration: FLIGHT * k, ease }}>
                <motion.g initial={{ opacity: 1, scale: 1 }} animate={{ opacity: 0, scale: 0.4 }} transition={{ delay: (at + FLIGHT * 0.6) * k, duration: FLIGHT * 0.4 * k }}>
                  <rect x={-2} y={-CHIP_H / 2 - 2} width={CHIP_W + 4} height={CHIP_H + 4} rx={5} fill="none" stroke="var(--accent)" strokeWidth={phase === "first" ? 2 : 0} />
                  {chip}
                </motion.g>
                <motion.g initial={{ opacity: 0, scale: 0 }} animate={{ opacity: 1, scale: [0, 1.8, 1] }} transition={{ delay: (at + FLIGHT * 0.8) * k, duration: 0.4 * k }}>{dot}</motion.g>
              </motion.g>
            </g>
          );
        }
        return (
          <g key={e.id}>
            <g transform={`translate(${L.x} ${L.y})`} opacity={landed ? 0.25 : 1}>{chip}</g>
            {landed && <g transform={`translate(${s.x} ${s.y})`}>{dot}</g>}
          </g>
        );
      })}
    </svg>
  );
}

function CellCount({ before, times, resetKey }: { before: number; times: number[]; resetKey: string }) {
  const n = useTicks(times, resetKey);
  return <motion.tspan key={n} initial={{ opacity: 0.3 }} animate={{ opacity: 1 }}>{before + n}</motion.tspan>;
}

function NormalizeText({ x, y, count, total, frac, k }: { x: number; y: number; count: number; total: number; frac: number; k: number }) {
  const common = { x, y, fontWeight: 700, fontFamily: "var(--font-mono)", fill: "var(--fg)" } as const;
  return (
    <g>
      <motion.text {...common} fontSize={22} initial={{ opacity: 1 }} animate={{ opacity: [1, 1, 0], y: [0, 0, -10] }} transition={{ duration: 1.4 * k, times: [0, 0.75, 1] }}>
        {count}<tspan fontSize={14} fill="var(--accent)"> / {total}</tspan>
      </motion.text>
      <motion.text {...common} fontSize={22} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 1.4 * k, duration: 0.4 * k }}>
        {Number.isFinite(frac) ? frac.toFixed(2) : "– (empty row)"}
      </motion.text>
    </g>
  );
}

/* ---------- step card ---------- */
function StepCard({ sim, phase, k, animKey, cfg }: { sim: CmSim; phase: CmPhase; k: number; animKey: string; cfg: CmConfig }) {
  const { names, m, pc, n } = sim;
  if (phase === "list") {
    return (
      <div className="space-y-1.5">
        {names.map((nm, c) => {
          const cnt = m[c].reduce((a, b) => a + b, 0);
          return <MetricBar key={nm} label={`actual ${nm}`} value={cnt / n} k={k} delay={0.3 + c * 0.3} tone={c === 0 ? "err" : "accent"} note={`${cnt} of ${n}`} />;
        })}
        <p className="pt-1 text-xs text-muted">A model-agnostic tally: any classifier, any number of classes. The rows will hold these class totals.</p>
      </div>
    );
  }
  if (phase === "first") {
    return (
      <ol className="space-y-1.5 text-sm">
        {sim.ex.slice(0, FIRST).map((e, j) => {
          const ok = e.actual === e.predicted;
          return (
            <motion.li key={e.id} className={cx("rounded-md border px-2 py-1.5", ok ? "border-ok/40 bg-ok-soft" : "border-err/40 bg-err-soft")}
              initial={{ opacity: 0, x: -12 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: (landAt("first", j, n) + FLIGHT * 0.8) * k, duration: 0.35 * k }}>
              <span className="font-mono">#{e.id}</span>: actually <strong>{names[e.actual]}</strong>, model said <strong>{names[e.predicted]}</strong> → row “{names[e.actual]}”, column “{names[e.predicted]}”
              {ok ? " ✓ on the diagonal" : " ✗ off the diagonal"}
            </motion.li>
          );
        })}
      </ol>
    );
  }
  if (phase === "rest") return <RestCard sim={sim} k={k} animKey={animKey} />;
  if (phase === "diagonal") {
    return (
      <div className="space-y-2">
        <MetricBar label="accuracy" value={sim.accuracy} k={k} delay={0.3 + sim.K * 0.4} tone="ok" note={`${sim.correct} / ${n}`} />
        <p className="text-xs text-muted">Diagonal cells: {m.map((r, i) => `${names[i]} ${r[i]}`).join(", ")}. The {n - sim.correct} errors are spread over the {sim.K * sim.K - sim.K} off-diagonal cells.</p>
        {cfg.majorityFault && <Callout tone="warn" title="Majority-class model">Every prediction lands in one column. Accuracy equals the majority share, yet the other classes are never found.</Callout>}
      </div>
    );
  }
  if (phase === "precision" || phase === "recall") {
    const isP = phase === "precision";
    return (
      <div className="space-y-1.5">
        {names.map((nm, c) => {
          const v = isP ? pc[c].precision : pc[c].recall;
          return <MetricBar key={nm} label={nm} value={v} k={k} delay={sweepAt(c) + 0.3} tone={c === 0 ? "err" : "accent"} note={isP ? `${pc[c].tp} / ${pc[c].col} in column` : `${pc[c].tp} / ${pc[c].row} in row`} />;
        })}
        <p className="pt-1 text-xs text-muted">{isP ? "Precision divides the diagonal cell by its column total." : "Recall divides the diagonal cell by its row total."}{" "}
          {!isP && Number.isFinite(pc[0].recall) && pc[0].recall < 0.7 && <strong className="text-err">Only {pct(pc[0].recall, 0)} of urgent messages are caught.</strong>}
        </p>
      </div>
    );
  }
  if (phase === "normalize") {
    return (
      <div className="space-y-1.5">
        {names.map((nm, a) => (
          <div key={nm} className="grid grid-cols-[96px_minmax(0,1fr)] items-center gap-2 text-xs">
            <span className="font-medium">actual {nm}</span>
            <div className="flex h-4 overflow-hidden rounded bg-surface-2">
              {sim.norm[a].map((v, p) => (
                <motion.div key={p} className={cx("h-full", p === a ? "bg-ok" : "bg-err/60", p > 0 && "border-l border-surface")} title={`${names[p]}: ${pct(v, 0)}`}
                  initial={{ width: 0 }} animate={{ width: `${(Number.isFinite(v) ? v : 0) * 100}%` }} transition={{ delay: (1.4 + a * 0.2) * k, duration: 0.6 * k, ease }} />
              ))}
            </div>
          </div>
        ))}
        <p className="pt-1 text-xs text-muted">Each bar is one row of the matrix; the green part is that class's recall, the red parts show where its misses went.</p>
      </div>
    );
  }
  const errsMiss = sim.missedCritical, errsOther = n - sim.correct - sim.missedCritical;
  return (
    <div className="space-y-2">
      <MetricBar label="missed urgent" value={sim.total ? (errsMiss * cfg.costMiss) / sim.total : 0} k={k} delay={0.4} tone="err" note={`${errsMiss} × ${cfg.costMiss} = ${errsMiss * cfg.costMiss}`} />
      <MetricBar label="other errors" value={sim.total ? (errsOther * cfg.costOther) / sim.total : 0} k={k} delay={0.8} tone="warn" note={`${errsOther} × ${cfg.costOther} = ${errsOther * cfg.costOther}`} />
      <p className="font-mono text-sm">total cost = <span className="text-lg font-semibold"><AnimatedNumber value={sim.total} from={0} digits={0} delay={1.2 * k} duration={0.8 * k} /></span> <span className="text-xs text-muted">(share of total shown as bars)</span></p>
      <p className="text-xs text-muted">Two models with the same accuracy can have very different cost. Raise the miss cost and compare the majority-class fault with a real model.</p>
    </div>
  );
}

function RestCard({ sim, k, animKey }: { sim: CmSim; k: number; animKey: string }) {
  const rest = sim.ex.slice(FIRST);
  const times = rest.map((_, j) => (landAt("rest", j, sim.n) + FLIGHT) * k);
  const placed = useTicks(times, animKey);
  const okTimes = rest.map((e, j) => (e.actual === e.predicted ? times[j] : -1)).filter((t) => t >= 0);
  const okN = useTicks(okTimes, `${animKey}-ok`);
  const firstOk = sim.ex.slice(0, FIRST).filter((e) => e.actual === e.predicted).length;
  const total = FIRST + placed;
  return (
    <div className="space-y-2 text-xs">
      <div className="grid grid-cols-[96px_minmax(0,1fr)_56px] items-center gap-2">
        <span className="font-medium">placed</span>
        <div className="h-4 overflow-hidden rounded bg-surface-2"><div className="h-full rounded bg-accent transition-[width] duration-200" style={{ width: `${(total / sim.n) * 100}%` }} /></div>
        <span className="text-right font-mono">{total} / {sim.n}</span>
      </div>
      <div className="grid grid-cols-[96px_minmax(0,1fr)_56px] items-center gap-2">
        <span className="font-medium">on diagonal</span>
        <div className="h-4 overflow-hidden rounded bg-surface-2"><div className="h-full rounded bg-ok transition-[width] duration-200" style={{ width: `${((firstOk + okN) / sim.n) * 100}%` }} /></div>
        <span className="text-right font-mono">{firstOk + okN}</span>
      </div>
      <p className="text-muted">Sum of all cells = number of examples, always. If your matrix doesn't add up, a label or prediction was dropped.</p>
    </div>
  );
}

function Calculation({ sim, phase }: { sim: CmSim; phase: CmPhase }) {
  const { names, m, pc, n } = sim;
  const f = (v: number) => (Number.isFinite(v) ? v.toFixed(3) : "\\text{undef.}");
  const t = (s: string) => `\\text{${s}}`;
  const mat = (M: (number | string)[][]) => `\\begin{bmatrix} ${M.map((r) => r.join(" & ")).join(" \\\\ ")} \\end{bmatrix}`;
  const tex: Record<CmPhase, string> = {
    list: `n = ${n} = ${names.map((_, c) => m[c].reduce((a, b) => a + b, 0)).join(" + ")}`,
    first: `M_{\\,\\text{actual},\\,\\text{pred}} \\mathrel{+}= 1`,
    rest: `M = ${mat(m)}`,
    diagonal: `${t("accuracy")} = \\frac{\\operatorname{tr}(M)}{n} = \\frac{${m.map((r, i) => r[i]).join(" + ")}}{${n}} = ${f(sim.accuracy)}`,
    precision: `\\begin{aligned} ${pc.map((c, k) => `P_{${t(names[k])}} &= \\tfrac{M_{${k + 1}${k + 1}}}{\\sum_i M_{i${k + 1}}} = \\tfrac{${c.tp}}{${c.col}} = ${f(c.precision)}`).join(" \\\\ ")} \\end{aligned}`,
    recall: `\\begin{aligned} ${pc.map((c, k) => `R_{${t(names[k])}} &= \\tfrac{M_{${k + 1}${k + 1}}}{\\sum_j M_{${k + 1}j}} = \\tfrac{${c.tp}}{${c.row}} = ${f(c.recall)}`).join(" \\\\ ")} \\end{aligned}`,
    normalize: `\\tilde{M} = ${mat(sim.norm.map((r) => r.map((v) => (Number.isFinite(v) ? v.toFixed(2) : "-"))))}`,
    cost: `\\text{cost} = \\sum_{i,j} M_{ij} C_{ij}, \\; C = ${mat(sim.cost)} \\Rightarrow ${sim.total}`,
  };
  return <TexBlock>{tex[phase]}</TexBlock>;
}
