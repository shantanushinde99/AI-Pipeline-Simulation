import { useMemo, useState } from "react";
import { motion } from "motion/react";
import { RotateCcw, Shuffle } from "lucide-react";
import { usePlayback } from "../../hooks/usePlayback";
import { PlaybackControls } from "../../components/visualizer/PlaybackControls";
import { EventLog } from "../../components/visualizer/EventLog";
import { StepExplainer, Workspace } from "../../components/visualizer/Workspace";
import { AnimatedNumber } from "../../components/visualizer/AnimatedNumber";
import { Badge, Button, KV, Panel, Select, SimulatedLabel, Slider, StatusBadge, cx } from "../../components/ui";
import { TexBlock } from "../../components/Tex";
import type { VisualizerTab } from "../types";
import { StepCaption, StepCard, ease } from "../model-versioning/kit";
import { DEFAULT_ET, TEST_SIZE, VAL_SIZES, simulateTracking, standardError, type ETConfig, type ETPhase, type ETResult, type ETState, type Mode, type Run } from "./simulation";

/** One timing schedule (seconds at 1×) for canvas, card and counters. */
const T = {
  define: { axes: 0.1, bands: 0.6 },
  run: { drop: 0.15, dropDur: 0.6, line: 0.35, row: 0.2, count: 0.25, best: 1.0 },
  best: { dim: 0.2, ring: 0.5, interval: (i: number) => 0.4 + i * 0.2 },
  compare: { col: (i: number) => 0.2 + i * 0.35, row: (i: number) => 0.6 + i * 0.18 },
  test: { seg: (i: number) => 0.2 + i * 0.09, verdict: 2.4 },
};
const DURATION: Record<ETPhase, number> = { define: 2400, run: 2000, best: 2600, compare: 3000, test: 4000 };

const W = 760;
const SC = { y0: 44, y1: 200, a: [70, 360] as const, b: [450, 740] as const };
const PC = { y0: 268, y1: 398, xs: [90, 300, 510, 720] };
const LR_LO = -4, LR_HI = -0.5;

function narrate(s: ETState, r: ETResult): { headline: string; sub: string } {
  const seeds = s.mode === "seeds";
  switch (s.phase) {
    case "define":
      return seeds ? { headline: "Same config, different random seeds", sub: "If results move a lot here, small differences between configs in a sweep mean little." }
        : { headline: `A random search of ${r.all.length} runs is about to start`, sub: "Every run logs its parameters and metrics to the tracker, so nothing depends on memory." };
    case "run": {
      const c = s.current!;
      const isBest = s.best?.id === c.id;
      return { headline: `Run ${c.id} finishes: validation accuracy ${c.valAcc.toFixed(3)}${isBest ? ", a new best" : ""}`,
        sub: seeds ? `Seed ${c.seed}. Same lr and depth as every other run; only chance differs.` : `lr ${c.lr}, depth ${c.depth}. Its row joins the table and its point drops onto every plot.` };
    }
    case "best":
      return seeds ? { headline: `Same config, a ${((r.seedStats!.max - r.seedStats!.min) * 100).toFixed(1)}-point spread`, sub: "Picking the best seed is picking noise. Report the mean ± spread instead." }
        : { headline: `Run ${s.best!.id} wins on validation accuracy`, sub: "But its lead may be smaller than the measurement noise. The bars show ±2 standard errors." };
    case "compare":
      return { headline: `Run ${s.compare![0].id} vs run ${s.compare![1].id}, side by side`, sub: "A tracker's diff view shows exactly which settings changed and how much the metrics moved." };
    case "test": {
      const drop = s.best!.valAcc - s.best!.testAcc;
      return r.bestByTest.id !== s.best!.id
        ? { headline: `The validation winner drops ${(drop * 100).toFixed(1)} pts on fresh data, and loses`, sub: `Run ${r.bestByTest.id} is better on the held-out test set. Run ${s.best!.id} was best partly by luck.` }
        : { headline: `The winner drops ${(drop * 100).toFixed(1)} pts on fresh data but stays best`, sub: "Taking the max of noisy scores is optimistic. Shrink the validation set or add runs to see the trap." };
    }
  }
}

export default function ExperimentTrackingVisualizer({ tab }: { tab: VisualizerTab }) {
  const [cfg, setCfg] = useState<ETConfig>(DEFAULT_ET);
  const [cmp, setCmp] = useState<[number, number] | undefined>(undefined);
  const [replay, setReplay] = useState(0);
  const result = useMemo(() => simulateTracking(cfg, cmp), [cfg, cmp]);
  const { steps } = result;
  const pb = usePlayback(steps.length, (i) => DURATION[steps[i]?.state.phase ?? "run"]);
  const step = steps[pb.index];
  const s = step.state;
  const k = 1 / pb.speed;
  const animKey = `${step.id}-${replay}`;
  const story = narrate(s, result);
  const set = (p: Partial<ETConfig>) => { setCfg((c) => ({ ...c, ...p })); setCmp(undefined); };
  if (tab === "metrics") return null;
  const cur = s.current ?? s.best;
  const compareIds = cmp ?? [steps.at(-2)!.state.compare![0].id, steps.at(-2)!.state.compare![1].id];

  return (
    <Workspace
      main={
        <>
          <Panel title="Experiment tracker" actions={<SimulatedLabel>Simulated training runs</SimulatedLabel>}>
            <StepCaption index={pb.index} total={steps.length} stepKey={step.id} headline={story.headline} sub={story.sub} onReplay={() => setReplay((r) => r + 1)} />
            <div className="overflow-x-auto"><div className="min-w-[560px]"><TrackerSvg s={s} r={result} k={k} animKey={animKey} /></div></div>
            <div className="mt-2 flex flex-wrap gap-3 text-[11px] text-muted">
              <span className="flex items-center gap-1"><svg width="10" height="10"><circle cx="5" cy="5" r="4" fill="var(--accent)" /></svg>a run (validation accuracy)</span>
              <span className="flex items-center gap-1"><svg width="14" height="14"><circle cx="7" cy="7" r="5" fill="none" stroke="var(--ok)" strokeWidth="2" /></svg>best so far ★</span>
              <span className="flex items-center gap-1"><svg width="10" height="10"><rect x="1" y="1" width="8" height="8" transform="rotate(45 5 5)" fill="var(--violet)" /></svg>test accuracy (last step)</span>
            </div>
          </Panel>
          <Panel title="What's happening in this step">
            <StepCard animKey={animKey}><StepDetail s={s} r={result} k={k} /></StepCard>
          </Panel>
          <PlaybackControls pb={pb} stepTitles={steps.map((x) => x.title)} />
          <Panel title="Sweep settings" actions={<Button size="sm" variant="ghost" onClick={() => { setCfg(DEFAULT_ET); setCmp(undefined); }}><RotateCcw size={12} /> Reset</Button>}>
            <div className="grid gap-4 sm:grid-cols-2">
              <Select<Mode> label="Mode" value={cfg.mode} options={[{ value: "sweep", label: "Hyperparameter sweep (random search)" }, { value: "seeds", label: "Seed variance: best config × N seeds" }]} onChange={(m) => set({ mode: m })} />
              <Slider label="Number of runs" value={cfg.runs} min={4} max={16} onChange={(v) => set({ runs: v })} />
              <Select label="Break it: validation set size" value={String(cfg.valSize)} options={VAL_SIZES.map((v) => ({ value: String(v), label: `${v.toLocaleString("en-US")} examples${v <= 50 ? " (tiny)" : ""}` }))} onChange={(v) => set({ valSize: Number(v) })} />
              <div className="space-y-1">
                <span className="block text-xs font-medium">Sweep seed: {cfg.sweepSeed}</span>
                <Button size="sm" onClick={() => set({ sweepSeed: (cfg.sweepSeed % 9) + 1 })}><Shuffle size={12} /> Sample a new sweep</Button>
              </div>
              <Select label="Compare: run A" value={String(compareIds[0])} options={result.all.map((r) => ({ value: String(r.id), label: `run ${r.id} · val ${r.valAcc.toFixed(3)}` }))} onChange={(v) => setCmp([Number(v), compareIds[1]])} />
              <Select label="Compare: run B" value={String(compareIds[1])} options={result.all.map((r) => ({ value: String(r.id), label: `run ${r.id} · val ${r.valAcc.toFixed(3)}` }))} onChange={(v) => setCmp([compareIds[0], Number(v)])} />
            </div>
            <p className="mt-2 text-xs text-muted">Try 50 validation examples with 16 runs: the winner's validation score gets more optimistic and often loses on the test set.</p>
          </Panel>
        </>
      }
      inspector={
        <>
          <StepExplainer index={pb.index} title={step.title} what={step.description}
            why={s.phase === "test" ? "Choosing the max of many noisy scores selects for luck. A separate test set (or repeated seeds) measures how much." : s.phase === "compare" ? "Without logged params and metrics, you cannot tell what actually changed between two runs." : undefined} />
          <Panel title="Calculation"><Calc s={s} /></Panel>
          <Panel title="Tracker summary">
            <KV rows={[
              ["runs logged", `${s.runs.length} / ${result.all.length}`],
              ["validation n", s.valSize.toLocaleString("en-US")],
              ["best so far", s.best ? `run ${s.best.id} · ${s.best.valAcc.toFixed(3)}` : "—"],
              ["selected run", cur ? `lr ${cur.lr}, depth ${cur.depth}, seed ${cur.seed}` : "—"],
              ...(result.seedStats ? [["seed spread (sd)", result.seedStats.sd.toFixed(4)] as [string, string]] : []),
            ]} />
          </Panel>
          <Panel title="Events"><EventLog events={step.events} /></Panel>
        </>
      }
    />
  );
}

function Calc({ s }: { s: ETState }) {
  const n = s.valSize;
  if (s.phase === "run" && s.current) {
    const c = s.current;
    const se = standardError(c.valAcc, n);
    return <TexBlock>{`\\begin{aligned} \\text{val acc} &= \\tfrac{\\text{correct}}{n} = \\tfrac{${Math.round(c.valAcc * n)}}{${n}} = ${c.valAcc.toFixed(3)} \\\\ \\mathrm{SE} &= \\sqrt{\\tfrac{p(1-p)}{n}} = \\sqrt{\\tfrac{${c.valAcc.toFixed(3)} \\cdot ${(1 - c.valAcc).toFixed(3)}}{${n}}} = ${se.toFixed(4)} \\end{aligned}`}</TexBlock>;
  }
  if (s.phase === "compare" && s.compare) {
    const [a, b] = s.compare;
    const se = Math.sqrt(standardError(a.valAcc, n) ** 2 + standardError(b.valAcc, n) ** 2);
    const d = a.valAcc - b.valAcc;
    return <TexBlock>{`\\begin{aligned} \\Delta &= ${a.valAcc.toFixed(3)} - ${b.valAcc.toFixed(3)} = ${d.toFixed(3)} \\\\ \\mathrm{SE}_\\Delta &= \\sqrt{\\mathrm{SE}_A^2 + \\mathrm{SE}_B^2} = ${se.toFixed(4)} \\\\ |\\Delta| / \\mathrm{SE}_\\Delta &= ${(Math.abs(d) / se).toFixed(2)} \\; ${Math.abs(d) / se < 2 ? "< 2: \\text{not distinguishable}" : "\\ge 2: \\text{likely real}"} \\end{aligned}`}</TexBlock>;
  }
  if (s.phase === "test" && s.best) {
    const b = s.best;
    return <TexBlock>{`\\begin{aligned} \\text{optimism} &= \\text{val} - \\text{test} \\\\ &= ${b.valAcc.toFixed(3)} - ${b.testAcc.toFixed(3)} = ${(b.valAcc - b.testAcc).toFixed(3)} \\\\ \\mathrm{SE}_{\\text{test}} &= \\sqrt{\\tfrac{p(1-p)}{${TEST_SIZE}}} = ${standardError(b.testAcc, TEST_SIZE).toFixed(4)} \\end{aligned}`}</TexBlock>;
  }
  if (s.phase === "best" && s.best) {
    return <TexBlock>{`\\begin{aligned} \\text{best} &= \\arg\\max_i \\text{val}_i = \\text{run } ${s.best.id} \\\\ \\text{val}_{\\text{best}} &= ${s.best.valAcc.toFixed(3)} \\pm ${(2 * standardError(s.best.valAcc, n)).toFixed(3)} \\; (2\\,\\mathrm{SE}) \\end{aligned}`}</TexBlock>;
  }
  return s.mode === "seeds"
    ? <TexBlock>{"\\text{val}_i = \\underbrace{\\text{acc}(\\eta, d)}_{\\text{fixed}} + \\underbrace{\\epsilon_{\\text{seed}}}_{\\text{training noise}} + \\underbrace{\\epsilon_{\\text{val}}}_{\\text{finite } n}"}</TexBlock>
    : <TexBlock>{"\\log_{10}\\eta \\sim U(-4, -0.5), \\quad d \\sim U\\{2, \\dots, 12\\}"}</TexBlock>;
}

/* ───────────────────────────── canvas ───────────────────────────── */

function TrackerSvg({ s, r, k, animKey }: { s: ETState; r: ETResult; k: number; animKey: string }) {
  const all = r.all;
  const vals = all.flatMap((x) => [x.valAcc, x.testAcc]);
  const ylo = Math.floor((Math.min(...vals) - 0.01) * 50) / 50, yhi = Math.ceil((Math.max(...vals) + 0.01) * 50) / 50;
  const sy = (v: number) => SC.y1 - ((v - ylo) / (yhi - ylo)) * (SC.y1 - SC.y0);
  const seeds = s.mode === "seeds";
  const jit = (run: Run) => (seeds ? (run.id - (all.length + 1) / 2) * 4 : 0);
  const sxA = (run: Run) => SC.a[0] + ((Math.log10(run.lr) - LR_LO) / (LR_HI - LR_LO)) * (SC.a[1] - SC.a[0]) + jit(run);
  const sxB = (run: Run) => SC.b[0] + ((run.depth - 2) / 10) * (SC.b[1] - SC.b[0]) + jit(run);
  const yTicks = Array.from({ length: 4 }, (_, i) => ylo + ((yhi - ylo) * i) / 3);
  // parallel coordinates: each axis normalised to its own range
  const trainLo = Math.min(...all.map((x) => x.trainAcc)), trainHi = Math.max(...all.map((x) => x.trainAcc));
  const axes = [
    { name: "log₁₀ lr", f: (x: Run) => (Math.log10(x.lr) - LR_LO) / (LR_HI - LR_LO), lo: "1e-4", hi: "0.3" },
    { name: "depth", f: (x: Run) => (x.depth - 2) / 10, lo: "2", hi: "12" },
    { name: "train acc", f: (x: Run) => (x.trainAcc - trainLo) / Math.max(1e-9, trainHi - trainLo), lo: trainLo.toFixed(2), hi: trainHi.toFixed(2) },
    { name: "val acc", f: (x: Run) => (x.valAcc - ylo) / (yhi - ylo), lo: ylo.toFixed(2), hi: yhi.toFixed(2) },
  ];
  const py = (t: number) => PC.y1 - Math.max(0, Math.min(1, t)) * (PC.y1 - PC.y0);
  const pPath = (x: Run) => axes.map((a, i) => `${i ? "L" : "M"}${PC.xs[i]} ${py(a.f(x)).toFixed(1)}`).join(" ");
  const best = s.best;
  const isNew = (x: Run) => s.phase === "run" && s.current?.id === x.id;
  const dimOthers = s.phase === "best" || s.phase === "compare";
  const highlighted = (x: Run) => (s.phase === "compare" ? s.compare!.some((c) => c.id === x.id) : best?.id === x.id);
  const testOn = s.phase === "test";
  const ranked = [...all].sort((a, b) => b.valAcc - a.valAcc);
  const bestMove = s.phase === "run" && s.previousBest && s.best?.id !== s.previousBest.id;

  const scatter = (sx: (x: Run) => number, x0: number, x1: number, label: string, ticks: [number, string][]) => (
    <g>
      <rect x={x0} y={SC.y0} width={x1 - x0} height={SC.y1 - SC.y0} fill="var(--surface-2)" opacity={0.5} rx={4} />
      {yTicks.map((t) => (
        <g key={t}><line x1={x0} x2={x1} y1={sy(t)} y2={sy(t)} stroke="var(--line)" /><text x={x0 - 5} y={sy(t) + 3} textAnchor="end" fontSize={9} fontFamily="var(--font-mono)" fill="var(--muted)">{t.toFixed(2)}</text></g>
      ))}
      {ticks.map(([v, l]) => <text key={l} x={v} y={SC.y1 + 13} textAnchor="middle" fontSize={9} fontFamily="var(--font-mono)" fill="var(--muted)">{l}</text>)}
      <text x={(x0 + x1) / 2} y={SC.y1 + 27} textAnchor="middle" fontSize={10} fill="var(--muted)">{label}</text>
      {s.phase === "define" && !seeds && (
        <motion.rect key={animKey} x={x0} y={SC.y0} height={SC.y1 - SC.y0} fill="var(--accent)" fillOpacity={0.1} initial={{ width: 0 }} animate={{ width: x1 - x0 }} transition={{ delay: T.define.bands * k, duration: 0.9 * k, ease }} />
      )}
      {testOn && all.map((x, i) => (
        <g key={`t-${x.id}-${animKey}`}>
          <motion.line x1={sx(x)} x2={sx(x)} y1={sy(x.valAcc)} y2={sy(x.testAcc)} stroke={x.id === best?.id ? "var(--err)" : "var(--violet)"} strokeWidth={x.id === best?.id ? 2.5 : 1.2}
            initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ delay: T.test.seg(i) * k, duration: 0.4 * k }} />
          <motion.rect x={sx(x) - 4} y={sy(x.testAcc) - 4} width={8} height={8} fill="var(--violet)" transform={`rotate(45 ${sx(x)} ${sy(x.testAcc)})`}
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: (T.test.seg(i) + 0.35) * k }} />
        </g>
      ))}
      {s.runs.map((x) => (
        <motion.circle key={`${x.id}-${isNew(x) ? animKey : "static"}`} cx={sx(x)} r={isNew(x) ? 6 : 4.5}
          fill={isNew(x) ? "var(--accent)" : "var(--accent)"} fillOpacity={dimOthers && !highlighted(x) ? 0.25 : 0.75} stroke="var(--surface)"
          initial={isNew(x) ? { cy: SC.y0 - 10, opacity: 0 } : false} animate={{ cy: sy(x.valAcc), opacity: 1 }}
          transition={{ delay: T.run.drop * k, duration: T.run.dropDur * k, ease }} />
      ))}
      {best && (
        <motion.g key={`best-${bestMove ? animKey : "s"}`} initial={bestMove && s.previousBest ? { x: sx(s.previousBest), y: sy(s.previousBest.valAcc) } : { x: sx(best), y: sy(best.valAcc), scale: s.phase === "best" ? 0.3 : 1 }}
          animate={{ x: sx(best), y: sy(best.valAcc), scale: 1 }} transition={{ delay: (s.phase === "run" ? T.run.best : T.best.ring) * k, type: "spring", stiffness: 90, damping: 15 }}>
          <circle r={10} fill="none" stroke="var(--ok)" strokeWidth={2.5} />
          <text y={-14} textAnchor="middle" fontSize={11} fill="var(--ok)" fontWeight={700}>★</text>
        </motion.g>
      )}
    </g>
  );

  return (
    <svg viewBox={`0 0 ${W} 420`} className="h-auto w-full" role="img" aria-label={`Experiment tracker plots: ${s.runs.length} of ${all.length} runs logged${best ? `, best run ${best.id}` : ""}`}>
      <text x={SC.a[0]} y={24} fontSize={11} fontWeight={600} fill="var(--fg)">validation accuracy vs learning rate</text>
      <text x={SC.b[0]} y={24} fontSize={11} fontWeight={600} fill="var(--fg)">validation accuracy vs tree depth</text>
      {scatter(sxA, SC.a[0], SC.a[1], "learning rate (log scale)", [[-4, "1e-4"], [-3, "1e-3"], [-2, "0.01"], [-1, "0.1"]].map(([v, l]) => [SC.a[0] + (((v as number) - LR_LO) / (LR_HI - LR_LO)) * (SC.a[1] - SC.a[0]), l as string]))}
      {scatter(sxB, SC.b[0], SC.b[1], "depth", [2, 4, 6, 8, 10, 12].map((d) => [SC.b[0] + ((d - 2) / 10) * (SC.b[1] - SC.b[0]), String(d)]))}
      {seeds && s.phase !== "define" && s.phase !== "run" && r.seedStats && (
        <motion.rect key={`band-${animKey}`} x={SC.a[0]} width={SC.a[1] - SC.a[0]} y={sy(r.seedStats.mean + r.seedStats.sd)} height={sy(r.seedStats.mean - r.seedStats.sd) - sy(r.seedStats.mean + r.seedStats.sd)}
          fill="var(--warn)" fillOpacity={0.15} initial={{ scaleY: 0 }} animate={{ scaleY: 1 }} style={{ transformOrigin: `0px ${sy(r.seedStats.mean)}px` }} transition={{ delay: T.best.dim * k, duration: 0.6 * k }} />
      )}

      {/* parallel coordinates */}
      <text x={20} y={PC.y0 - 22} fontSize={11} fontWeight={600} fill="var(--fg)">parallel coordinates: one line per run</text>
      {axes.map((a, i) => (
        <g key={a.name}>
          <line x1={PC.xs[i]} x2={PC.xs[i]} y1={PC.y0} y2={PC.y1} stroke="var(--muted)" strokeWidth={1.5} />
          <text x={PC.xs[i]} y={PC.y0 - 12} textAnchor="middle" fontSize={10} fontWeight={600} fill="var(--fg)">{a.name}</text>
          <text x={PC.xs[i] + 5} y={PC.y0 + 8} fontSize={9} fontFamily="var(--font-mono)" fill="var(--muted)">{a.hi}</text>
          <text x={PC.xs[i] + 5} y={PC.y1} fontSize={9} fontFamily="var(--font-mono)" fill="var(--muted)">{a.lo}</text>
        </g>
      ))}
      {s.runs.map((x) => {
        const hl = highlighted(x) && s.phase !== "run";
        return (
          <motion.path key={`p-${x.id}-${isNew(x) ? animKey : "static"}`} d={pPath(x)} fill="none" stroke={hl ? "var(--ok)" : isNew(x) ? "var(--accent)" : "var(--muted)"}
            strokeWidth={hl || isNew(x) ? 2.5 : 1.2} strokeOpacity={hl || isNew(x) ? 1 : dimOthers ? 0.2 : 0.45}
            initial={isNew(x) ? { pathLength: 0 } : false} animate={{ pathLength: 1 }} transition={{ delay: T.run.line * k, duration: 0.8 * k, ease }} />
        );
      })}
      {testOn && (
        <motion.text key={`v-${animKey}`} x={W - 20} y={PC.y0 - 22} textAnchor="end" fontSize={10.5} fill="var(--fg)" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: T.test.verdict * k }}>
          ranked by test: {[...all].sort((a, b) => b.testAcc - a.testAcc).slice(0, 3).map((x) => `run ${x.id}`).join(" > ")} · val ranking: {ranked.slice(0, 3).map((x) => `run ${x.id}`).join(" > ")}
        </motion.text>
      )}
    </svg>
  );
}

/* ───────────────────────────── step card ───────────────────────────── */

function Leaderboard({ runs, current, best, k, n = 6 }: { runs: Run[]; current?: Run; best?: Run; k: number; n?: number }) {
  const ranked = [...runs].sort((a, b) => b.valAcc - a.valAcc || a.id - b.id);
  const top = ranked.slice(0, n);
  if (current && !top.includes(current)) top.push(current);
  return (
    <div>
      <table className="w-full text-xs">
        <thead className="text-left text-muted"><tr><th className="py-1">#</th><th>run</th><th>lr</th><th>depth</th><th>train</th><th className="text-right">val</th></tr></thead>
        <tbody className="font-mono">
          {top.map((x) => (
            <motion.tr layout key={x.id} transition={{ layout: { duration: 0.5 * k, ease } }} initial={x.id === current?.id ? { opacity: 0, x: -20 } : false} animate={{ opacity: 1, x: 0 }}
              className={cx("border-t border-line", x.id === current?.id && "bg-accent-soft", x.id === best?.id && "font-semibold")}>
              <td className="py-1">{ranked.indexOf(x) + 1}</td><td>{x.id}{x.id === best?.id && " ★"}</td><td>{x.lr}</td><td>{x.depth}</td><td>{x.trainAcc.toFixed(3)}</td><td className="text-right">{x.valAcc.toFixed(3)}</td>
            </motion.tr>
          ))}
        </tbody>
      </table>
      {runs.length > top.length && <p className="mt-1 text-[11px] text-muted">… {runs.length - top.length} more runs in the tracker</p>}
    </div>
  );
}

function StepDetail({ s, r, k }: { s: ETState; r: ETResult; k: number }) {
  const n = s.valSize;
  if (s.phase === "define") {
    if (s.mode === "seeds") {
      const w = r.all[0];
      return <p className="text-sm">Config fixed at <span className="font-mono">lr = {w.lr}, depth = {w.depth}</span> (the sweep winner). {r.all.length} runs with seeds {r.all[0].seed}–{r.all.at(-1)!.seed}. Validation set: {n} examples.</p>;
    }
    return (
      <div className="grid gap-3 sm:grid-cols-2">
        {[["learning rate η", "log-uniform 1e-4 … 0.3"], ["tree depth d", "integer 2 … 12"]].map(([a, b], i) => (
          <motion.div key={a} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: (0.2 + i * 0.3) * k }} className="rounded-lg border border-line bg-surface-2 p-3">
            <div className="text-xs text-muted">{a}</div><div className="font-mono text-sm">{b}</div>
          </motion.div>
        ))}
        <p className="text-xs text-muted sm:col-span-2">Each run logs: params (lr, depth, seed), metrics (train and validation accuracy) and a run id. The test set is not touched until the end.</p>
      </div>
    );
  }
  if (s.phase === "run" && s.current) {
    const c = s.current;
    const correct = Math.round(c.valAcc * n);
    return (
      <div className="grid gap-4 md:grid-cols-[220px_minmax(0,1fr)]">
        <div className="space-y-2">
          <div className="text-sm font-semibold">run {c.id} <span className="font-normal text-muted">· seed {c.seed}</span></div>
          <KV rows={[["lr", c.lr], ["depth", c.depth], ["train acc", c.trainAcc.toFixed(3)]]} />
          <div className="text-xs text-muted">validation: correct / {n}</div>
          <div className="relative h-5 overflow-hidden rounded bg-surface-2">
            <motion.div className="absolute inset-y-0 left-0 bg-accent/70" initial={{ width: 0 }} animate={{ width: `${c.valAcc * 100}%` }} transition={{ delay: T.run.count * k, duration: 0.7 * k, ease }} />
            <span className="absolute inset-0 flex items-center justify-center font-mono text-xs">
              <AnimatedNumber value={correct} from={0} digits={0} delay={T.run.count * k} duration={0.7 * k} />&nbsp;/ {n} =&nbsp;<AnimatedNumber value={c.valAcc} from={0} digits={3} delay={T.run.count * k} duration={0.7 * k} />
            </span>
          </div>
          {s.best?.id === c.id && <motion.div initial={{ opacity: 0, scale: 0.8 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: T.run.best * k }}><Badge tone="ok">★ new best</Badge></motion.div>}
        </div>
        <Leaderboard runs={s.runs} current={c} best={s.best} k={k} />
      </div>
    );
  }
  if (s.phase === "best" && s.best) {
    const ranked = [...s.runs].sort((a, b) => b.valAcc - a.valAcc).slice(0, 5);
    const lo = Math.min(...ranked.map((x) => x.valAcc - 2 * standardError(x.valAcc, n))) - 0.005;
    const hi = Math.max(...ranked.map((x) => x.valAcc + 2 * standardError(x.valAcc, n))) + 0.005;
    const pct = (v: number) => `${((v - lo) / (hi - lo)) * 100}%`;
    return (
      <div className="space-y-1.5">
        <h4 className="text-sm font-semibold">Top {ranked.length} by validation, with ±2 SE intervals (n = {n})</h4>
        {ranked.map((x, i) => {
          const e = 2 * standardError(x.valAcc, n);
          return (
            <div key={x.id} className="grid grid-cols-[70px_minmax(0,1fr)_52px] items-center gap-2 text-xs">
              <span className="font-mono">run {x.id}{i === 0 && " ★"}</span>
              <div className="relative h-4 rounded bg-surface-2">
                <motion.div className={cx("absolute inset-y-1 rounded", i === 0 ? "bg-ok/50" : "bg-accent/35")} style={{ left: pct(x.valAcc - e) }} initial={{ width: 0 }} animate={{ width: `calc(${pct(x.valAcc + e)} - ${pct(x.valAcc - e)})` }} transition={{ delay: T.best.interval(i) * k, duration: 0.5 * k, ease }} />
                <motion.div className="absolute inset-y-0 w-1 -translate-x-1/2 rounded bg-fg" style={{ left: pct(x.valAcc) }} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: T.best.interval(i) * k }} />
              </div>
              <span className="text-right font-mono">{x.valAcc.toFixed(3)}</span>
            </div>
          );
        })}
        <p className="text-xs text-muted">{s.mode === "seeds" ? `Mean ${r.seedStats!.mean.toFixed(3)}, sd ${r.seedStats!.sd.toFixed(3)} across ${s.runs.length} seeds of one config.` : "Overlapping intervals mean the ranking among these runs is not settled by this validation set."}</p>
      </div>
    );
  }
  if (s.phase === "compare" && s.compare) {
    const [a, b] = s.compare;
    const rows: [string, (x: Run) => string, number | null][] = [
      ["learning rate", (x) => String(x.lr), null], ["depth", (x) => String(x.depth), null], ["seed", (x) => String(x.seed), null],
      ["train acc", (x) => x.trainAcc.toFixed(3), a.trainAcc - b.trainAcc], ["val acc", (x) => x.valAcc.toFixed(3), a.valAcc - b.valAcc],
    ];
    return (
      <table className="w-full text-xs">
        <thead className="text-left text-muted">
          <tr><th className="py-1" />{[a, b].map((x, i) => <motion.th key={x.id} initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: T.compare.col(i) * k }}>run {x.id}{x.id === s.best?.id && " ★"}</motion.th>)}<th>A − B</th></tr>
        </thead>
        <tbody className="font-mono">
          {rows.map(([name, f, d], i) => {
            const changed = f(a) !== f(b);
            return (
              <motion.tr key={name} className={cx("border-t border-line", changed && "bg-warn-soft/60")} initial={{ opacity: 0, x: -12 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: T.compare.row(i) * k }}>
                <td className="py-1 font-sans">{name}</td><td>{f(a)}</td><td>{f(b)}</td>
                <td>{d === null ? (changed ? <Badge tone="warn">changed</Badge> : <span className="text-muted">same</span>) : `${d >= 0 ? "+" : ""}${(d * 100).toFixed(1)} pts`}</td>
              </motion.tr>
            );
          })}
        </tbody>
      </table>
    );
  }
  // test
  const ranked = [...s.runs].sort((a, b) => b.valAcc - a.valAcc).slice(0, 6);
  return (
    <div className="space-y-2">
      <table className="w-full text-xs">
        <thead className="text-left text-muted"><tr><th className="py-1">run</th><th>val (n={n})</th><th>test (n={TEST_SIZE.toLocaleString("en-US")})</th><th>val − test</th></tr></thead>
        <tbody className="font-mono">
          {ranked.map((x, i) => (
            <motion.tr key={x.id} className={cx("border-t border-line", x.id === r.bestByTest.id && "bg-ok-soft")} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: T.test.seg(i) * k }}>
              <td className="py-1">{x.id}{x.id === s.best?.id && " ★val"}{x.id === r.bestByTest.id && " ✓test"}</td><td>{x.valAcc.toFixed(3)}</td>
              <td><AnimatedNumber value={x.testAcc} from={x.valAcc} digits={3} delay={(T.test.seg(i) + 0.2) * k} duration={0.5 * k} /></td>
              <td className={x.valAcc - x.testAcc > 0.01 ? "text-err" : undefined}>{(x.valAcc - x.testAcc >= 0 ? "+" : "") + ((x.valAcc - x.testAcc) * 100).toFixed(1)} pts</td>
            </motion.tr>
          ))}
        </tbody>
      </table>
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: T.test.verdict * k }}>
        <StatusBadge status={r.bestByTest.id === s.best?.id ? "ok" : "warning"} label={r.bestByTest.id === s.best?.id ? "validation winner holds up on test" : `best by chance: run ${r.bestByTest.id} is better on test`} />
      </motion.div>
    </div>
  );
}
