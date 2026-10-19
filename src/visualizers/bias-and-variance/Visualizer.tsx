import { useCallback, useMemo, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { RotateCcw } from "lucide-react";
import { usePlayback } from "../../hooks/usePlayback";
import { PlaybackControls } from "../../components/visualizer/PlaybackControls";
import { EventLog } from "../../components/visualizer/EventLog";
import { LineChart } from "../../components/visualizer/LineChart";
import { StepExplainer, Workspace } from "../../components/visualizer/Workspace";
import { AnimatedNumber } from "../../components/visualizer/AnimatedNumber";
import { Badge, Button, KV, Panel, Slider } from "../../components/ui";
import { Tex, TexBlock } from "../../components/Tex";
import type { VisualizerTab } from "../types";
import { EASE, GrowBar, StepCaption } from "../decision-trees/StepCaption";
import { trueFn } from "../model-evaluation/simulation";
import { GRID, MAX_DEGREE, simulate, type BvConfig, type BvPhase, type BvSim, type BvState } from "./simulation";

const DEFAULT: BvConfig = { degree: 3, noise: 0.3, n: 15, sets: 20, seed: 1 };
const PRESETS: { label: string; cfg: Partial<BvConfig> }[] = [
  { label: "Underfit (degree 1)", cfg: { degree: 1, n: 15 } },
  { label: "Sweet spot (degree 3)", cfg: { degree: 3, n: 15 } },
  { label: "Break it: degree 10, 12 points", cfg: { degree: 10, n: 12 } },
  { label: "More data tames variance (degree 10, 40 points)", cfg: { degree: 10, n: 40 } },
];
const ORDER: BvPhase[] = ["truth", "one", "many", "mean", "variance", "decompose", "sweep"];
const SPOTS = [-0.6, 0, 0.6];
/** Stagger between curves in the "many" step, so all M curves fit in about three seconds. */
const stagger = (m: number) => Math.min(0.15, 3 / m);
const fmt = (v: number) => (Math.abs(v) >= 100 ? v.toExponential(2) : v.toFixed(4));
const COLORS = { bias: "var(--err)", variance: "var(--violet)", noise: "var(--muted)" };

export default function BiasVarianceVisualizer({ tab }: { tab: VisualizerTab }) {
  const [cfg, setCfg] = useState(DEFAULT);
  const set = (p: Partial<BvConfig>) => setCfg((c) => ({ ...c, ...p }));
  const sim = useMemo(() => simulate(cfg), [cfg]);
  const duration = useCallback((i: number) => {
    const ph = sim.steps[i]?.state.phase;
    return ph === "many" ? (0.5 + sim.cfg.sets * stagger(sim.cfg.sets) + 1.2) * 1000 : ph === "decompose" ? 3600 : 3000;
  }, [sim]);
  const pb = usePlayback(sim.steps.length, duration);
  const [replay, setReplay] = useState(0);
  const step = sim.steps[pb.index];
  const k = 1 / pb.speed;
  const animKey = `${step.id}-${replay}`;
  if (tab === "metrics") return null;
  const st = step.state;

  return (
    <Workspace
      main={
        <>
          <Panel title="Many training sets, one model class" actions={<span className="flex gap-1"><Badge tone="err">bias² {fmt(sim.bias2)}</Badge><Badge tone="accent">var {fmt(sim.variance)}</Badge></span>}>
            <StepCaption index={pb.index} total={sim.steps.length} stepId={step.id} headline={st.headline} sub={st.sub} onReplay={() => setReplay((r) => r + 1)} />
            <Plot sim={sim} st={st} k={k} animKey={animKey} />
            <div className="mt-1 flex flex-wrap gap-3 text-[11px] text-muted">
              <span>- - - true function sin(3x)</span>
              <span className="text-accent">— one fitted curve per training set</span>
              <span style={{ color: "var(--warn)" }}>━ average of all curves</span>
              <span style={{ color: COLORS.bias }}>▒ bias (gap to truth)</span>
              <span style={{ color: COLORS.variance }}>▒ ±1 sd band (variance)</span>
              <span>y clipped to [−2.5, 2.5]</span>
            </div>
          </Panel>
          <Panel title="What's happening in this step">
            <AnimatePresence mode="wait" initial={false}>
              <motion.div key={animKey} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: 0.2 }}>
                <StepCard sim={sim} st={st} k={k} />
              </motion.div>
            </AnimatePresence>
          </Panel>
          <PlaybackControls pb={pb} stepTitles={sim.steps.map((s) => s.title)} />
          <Panel title="Parameters" actions={<Button size="sm" variant="ghost" onClick={() => setCfg(DEFAULT)}><RotateCcw size={12} /> Reset</Button>}>
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              <Slider label="Model complexity (degree)" value={cfg.degree} min={1} max={MAX_DEGREE} onChange={(v) => set({ degree: v })} />
              <Slider label={<>Noise <Tex>{"\\sigma"}</Tex></>} value={cfg.noise} min={0} max={0.8} step={0.05} format={(v) => v.toFixed(2)} onChange={(v) => set({ noise: v })} hint="Irreducible error is σ²." />
              <Slider label="Points per training set" value={cfg.n} min={12} max={60} onChange={(v) => set({ n: v })} />
              <Slider label="Number of training sets" value={cfg.sets} min={5} max={40} onChange={(v) => set({ sets: v })} hint="More sets = steadier estimates of bias and variance." />
              <Slider label="Random seed" value={cfg.seed} min={1} max={50} onChange={(v) => set({ seed: v })} />
            </div>
            <div className="mt-3 flex flex-wrap gap-1.5">
              {PRESETS.map((p) => <Button key={p.label} size="sm" onClick={() => set(p.cfg)}>{p.label}</Button>)}
            </div>
          </Panel>
        </>
      }
      inspector={
        <>
          <StepExplainer index={pb.index} title={step.title} what={step.description}
            why="Test error has three sources. Bias: the model class can't represent the truth. Variance: the fitted model depends too much on which sample you got. Noise: randomness in the labels. Complexity trades the first two against each other." />
          <Panel title="Calculation">
            <TexBlock>{`\\begin{aligned} \\bar{f}(x) &= \\tfrac{1}{M}\\textstyle\\sum_{m=1}^{${sim.cfg.sets}} \\hat{f}_m(x) \\\\ \\text{bias}^2 &= \\overline{(\\bar{f}(x) - f(x))^2} = ${fmt(sim.bias2)} \\\\ \\text{var} &= \\overline{\\tfrac{1}{M}\\textstyle\\sum_m (\\hat{f}_m(x) - \\bar{f}(x))^2} = ${fmt(sim.variance)} \\\\ \\sigma^2 &= ${sim.cfg.noise}^2 = ${fmt(sim.noise2)} \\\\ \\mathbb{E}[(y - \\hat{f})^2] &= ${fmt(sim.total)} \\end{aligned}`}</TexBlock>
            <p className="mt-2 text-[11px] text-muted">Overlines average over {GRID.length} evenly spaced x in [−1, 1]. Each <Tex>{"\\hat{f}_m"}</Tex> is a least-squares polynomial of degree {sim.cfg.degree}.</p>
          </Panel>
          <Panel title="Outputs">
            <KV rows={[
              ["bias²", fmt(sim.bias2)],
              ["variance", fmt(sim.variance)],
              ["noise σ²", fmt(sim.noise2)],
              ["expected test error", fmt(sim.total)],
              ["measured (fresh labels)", fmt(sim.measured)],
              ["avg training MSE", fmt(sim.trainMse)],
              ["best degree (sweep)", sim.best.degree],
            ]} />
          </Panel>
          <Panel title="Events"><EventLog events={step.events} /></Panel>
        </>
      }
    />
  );
}

const W = 600, H = 320, P = 30, YR = 2.5;
const sx = (x: number) => P + ((x + 1) / 2) * (W - 2 * P);
const sy = (y: number) => H / 2 - (Math.max(-50, Math.min(50, y)) / YR) * (H / 2 - P);
const path = (ys: number[]) => ys.map((y, i) => `${i ? "L" : "M"}${sx(GRID[i]).toFixed(1)} ${sy(y).toFixed(1)}`).join(" ");

function Plot({ sim, st, k, animKey }: { sim: BvSim; st: BvState; k: number; animKey: string }) {
  const phase = ORDER.indexOf(st.phase);
  const at = (p: BvPhase) => phase >= ORDER.indexOf(p);
  const M = sim.cfg.sets, sg = stagger(M);
  const truth = useMemo(() => path(GRID.map(trueFn)), []);
  const sd = sim.varx.map(Math.sqrt);
  const band = `${path(sim.avg.map((v, g) => v + sd[g]))} ${sim.avg.map((_, g) => `L${sx(GRID[GRID.length - 1 - g]).toFixed(1)} ${sy(sim.avg[GRID.length - 1 - g] - sd[GRID.length - 1 - g]).toFixed(1)}`).join(" ")} Z`;
  const biasArea = `${path(sim.avg)} ${GRID.map((_, g) => { const i = GRID.length - 1 - g; return `L${sx(GRID[i]).toFixed(1)} ${sy(trueFn(GRID[i])).toFixed(1)}`; }).join(" ")} Z`;
  const many = st.phase === "many";
  const spotIdx = SPOTS.map((x) => GRID.findIndex((g) => Math.abs(g - x) < 1e-9));
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label="Fitted curves from many training sets around the true function">
      <line x1={P} x2={W - P} y1={H / 2} y2={H / 2} stroke="var(--line)" />
      <line x1={W / 2} x2={W / 2} y1={P / 2} y2={H - P / 2} stroke="var(--line)" />
      {[-1, -0.5, 0, 0.5, 1].map((x) => <text key={x} x={sx(x)} y={H - 6} fontSize={9} textAnchor="middle" fill="var(--muted)">{x}</text>)}
      <clipPath id="bv-clip"><rect x={P} y={P / 2} width={W - 2 * P} height={H - P} /></clipPath>
      <g clipPath="url(#bv-clip)">
        {at("variance") && <motion.path key={`band-${st.phase === "variance" ? animKey : ""}`} d={band} fill={COLORS.variance} initial={{ fillOpacity: 0 }} animate={{ fillOpacity: 0.16 }} transition={{ delay: (st.phase === "variance" ? 0.2 : 0) * k, duration: 0.8 * k }} />}
        {at("mean") && <motion.path key={`bias-${st.phase === "mean" ? animKey : ""}`} d={biasArea} fill={COLORS.bias} initial={{ fillOpacity: 0 }} animate={{ fillOpacity: 0.22 }} transition={{ delay: (st.phase === "mean" ? 1.3 : 0) * k, duration: 0.8 * k }} />}
        <motion.path key={`truth-${st.phase === "truth" ? animKey : ""}`} d={truth} fill="none" stroke="var(--fg)" strokeDasharray="6 4" strokeWidth={1.8}
          initial={{ pathLength: st.phase === "truth" ? 0 : 1 }} animate={{ pathLength: 1 }} transition={{ delay: 0.2 * k, duration: 1.4 * k, ease: "easeInOut" }} />
        {/* the cloud of fitted curves */}
        {at("one") && sim.curves.map((c, m) => {
          if (m > 0 && !at("many")) return null;
          const drawing = (st.phase === "one" && m === 0) || (many && m > 0);
          const delay = st.phase === "one" ? 1.2 : 0.4 + (m - 1) * sg;
          const strong = st.phase === "one" || (many && m === M - 1);
          return (
            <motion.path key={`c-${m}-${drawing ? animKey : ""}`} d={path(c)} fill="none" stroke="var(--accent)" strokeWidth={strong ? 2.2 : 1.2}
              initial={drawing ? { pathLength: 0, strokeOpacity: 0.9 } : false} animate={{ pathLength: 1, strokeOpacity: at("mean") ? 0.16 : st.phase === "one" ? 0.95 : 0.4 }}
              transition={{ pathLength: { delay: delay * k, duration: (st.phase === "one" ? 1 : 0.5) * k, ease: "easeInOut" }, strokeOpacity: { delay: drawing ? (delay + 0.5) * k : 0, duration: 0.6 * k } }} />
          );
        })}
        {/* training points: set #1 on the "one" step, each set flashes as its curve is drawn */}
        {st.phase === "one" && sim.sets[0].map((p, i) => (
          <motion.circle key={`p-${i}-${animKey}`} cx={sx(p.x)} cy={sy(p.y)} r={4} fill="var(--surface)" stroke="var(--accent)" strokeWidth={2} initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ delay: (0.1 + i * 0.05) * k, duration: 0.3 * k }} />
        ))}
        {many && sim.sets.slice(1).map((s, m) => (
          <motion.g key={`ps-${m}-${animKey}`} initial={{ opacity: 0 }} animate={{ opacity: [0, 0.9, 0] }} transition={{ delay: (0.4 + m * sg - 0.1) * k, duration: Math.max(0.4, sg * 3) * k }}>
            {s.map((p, i) => <circle key={i} cx={sx(p.x)} cy={sy(p.y)} r={2.5} fill="var(--accent)" />)}
          </motion.g>
        ))}
        {at("mean") && <motion.path key={`avg-${st.phase === "mean" ? animKey : ""}`} d={path(sim.avg)} fill="none" stroke="var(--warn)" strokeWidth={3.2}
          initial={{ pathLength: st.phase === "mean" ? 0 : 1 }} animate={{ pathLength: 1 }} transition={{ delay: 0.2 * k, duration: 1.1 * k, ease: "easeInOut" }} />}
        {st.phase === "variance" && spotIdx.map((g, s) => (
          <g key={`spot-${s}-${animKey}`}>
            <line x1={sx(GRID[g])} x2={sx(GRID[g])} y1={P / 2} y2={H - P / 2} stroke="var(--violet)" strokeDasharray="2 3" />
            {sim.curves.map((c, m) => (
              <motion.circle key={m} cx={sx(GRID[g])} r={3} fill="var(--violet)" fillOpacity={0.75} initial={{ cy: sy(sim.avg[g]), opacity: 0 }} animate={{ cy: sy(c[g]), opacity: 1 }}
                transition={{ delay: (0.9 + s * 0.3) * k, duration: 0.7 * k, ease: EASE }} />
            ))}
          </g>
        ))}
      </g>
    </svg>
  );
}

function StepCard({ sim, st, k }: { sim: BvSim; st: BvState; k: number }) {
  const max = Math.max(sim.total, 1e-9);
  if (st.phase === "truth") {
    return (
      <div className="grid gap-2 sm:grid-cols-3">
        {[["true function", "f(x) = sin(3x)"], ["label noise", `ε ~ N(0, ${sim.cfg.noise}²)`], ["irreducible error", `σ² = ${fmt(sim.noise2)}`]].map(([a, b], i) => (
          <motion.div key={a} className="rounded-md border border-line p-2" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: (0.3 + i * 0.3) * k }}>
            <div className="text-[11px] text-muted">{a}</div><div className="font-mono text-sm">{b}</div>
          </motion.div>
        ))}
        <p className="text-xs text-muted sm:col-span-3">Plan: draw {sim.cfg.sets} training sets of {sim.cfg.n} points each, fit the same degree-{sim.cfg.degree} model to every one, and look at the family of curves.</p>
      </div>
    );
  }
  if (st.phase === "one") {
    return (
      <div className="space-y-2 text-xs">
        <p>Degree {sim.cfg.degree} means {sim.cfg.degree + 1} coefficients fitted to {sim.cfg.n} points.</p>
        <GrowBar label="training MSE, set #1" value={sim.firstTrainMse} max={Math.max(sim.firstTrainMse, sim.total)} color="var(--accent)" delay={1.4 * k} duration={0.7 * k} digits={4} />
        <GrowBar label="expected test error" value={sim.total} max={Math.max(sim.firstTrainMse, sim.total)} color="var(--warn)" delay={1.8 * k} duration={0.7 * k} digits={4} />
        <p className="text-muted">A single fit can look great on its own points. The question is how it would have looked on a different sample.</p>
      </div>
    );
  }
  if (st.phase === "many") {
    const sg = stagger(sim.cfg.sets);
    return (
      <div className="flex flex-wrap items-center gap-6">
        <div>
          <div className="text-[11px] text-muted">curves fitted</div>
          <div className="font-mono text-3xl font-semibold"><AnimatedNumber value={sim.cfg.sets} from={1} digits={0} delay={0.4 * k} duration={(sim.cfg.sets - 1) * sg * k} /> <span className="text-base text-muted">/ {sim.cfg.sets}</span></div>
        </div>
        <p className="max-w-md text-xs leading-relaxed">Every curve uses the same recipe; only the random sample changes. {sim.cfg.degree >= 7 ? "A flexible model chases each sample's noise, so the curves fan out, especially near the edges." : sim.cfg.degree <= 2 ? "A rigid model barely reacts to the sample, so the curves stay bunched, but bunched in the wrong place." : "A moderate model follows the shared shape and ignores most of the noise."}</p>
      </div>
    );
  }
  if (st.phase === "mean") {
    return (
      <div className="space-y-2">
        <GrowBar label="bias²" value={sim.bias2} max={max} color={COLORS.bias} delay={1.3 * k} duration={0.8 * k} digits={4} />
        <p className="text-xs text-muted">Bias² is how far the <em>average</em> model is from the truth. Averaging cancels sample-to-sample noise, so what remains is the model class's systematic error. Raise the degree to shrink it.</p>
      </div>
    );
  }
  if (st.phase === "variance") {
    return (
      <div className="space-y-2">
        <GrowBar label="variance" value={sim.variance} max={max} color={COLORS.variance} delay={0.3 * k} duration={0.8 * k} digits={4} />
        <table className="w-full font-mono text-xs">
          <thead className="text-left text-muted"><tr><th>x</th><th>truth f(x)</th><th>average f̄(x)</th><th>spread (sd)</th></tr></thead>
          <tbody>
            {SPOTS.map((x, s) => {
              const g = GRID.findIndex((v) => Math.abs(v - x) < 1e-9);
              return (
                <motion.tr key={x} className="border-t border-line" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: (1.2 + s * 0.3) * k }}>
                  <td>{x}</td><td>{trueFn(x).toFixed(3)}</td><td>{sim.avg[g].toFixed(3)}</td><td>{Math.sqrt(sim.varx[g]).toFixed(3)}</td>
                </motion.tr>
              );
            })}
          </tbody>
        </table>
      </div>
    );
  }
  if (st.phase === "decompose") {
    const parts = [
      { name: "bias²", v: sim.bias2, c: COLORS.bias },
      { name: "variance", v: sim.variance, c: COLORS.variance },
      { name: "noise σ²", v: sim.noise2, c: COLORS.noise },
    ];
    let acc = 0;
    return (
      <div className="space-y-2">
        <div className="relative h-8 overflow-hidden rounded-md bg-surface-2" aria-label="Stacked decomposition of expected error">
          {parts.map((p, i) => {
            const left = (acc / max) * 100, w = (p.v / max) * 100;
            acc += p.v;
            return <motion.div key={p.name} className="absolute inset-y-0" style={{ left: `${left}%`, background: p.c }} initial={{ width: 0 }} animate={{ width: `${w}%` }} transition={{ delay: (0.3 + i * 0.6) * k, duration: 0.6 * k, ease: EASE }} title={`${p.name} ${fmt(p.v)}`} />;
          })}
        </div>
        {parts.map((p, i) => <GrowBar key={p.name} label={p.name} value={p.v} max={max} color={p.c} delay={(0.3 + i * 0.6) * k} duration={0.6 * k} digits={4} />)}
        <motion.div className="grid gap-1 border-t border-line pt-2 text-xs sm:grid-cols-2" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 2.2 * k }}>
          <span>sum = <strong className="font-mono">{fmt(sim.total)}</strong>; measured on fresh labels: <span className="font-mono">{fmt(sim.measured)}</span></span>
          <span>average training MSE: <span className="font-mono">{fmt(sim.trainMse)}</span>{sim.trainMse < sim.total * 0.7 && " (looks much better than it is)"}</span>
        </motion.div>
      </div>
    );
  }
  return (
    <div>
      <LineChart logY markerX={sim.cfg.degree} xLabel="polynomial degree" yLabel="error" height={220} series={[
        { name: "bias²", points: sim.sweep.map((s) => [s.degree, s.bias2]), color: COLORS.bias },
        { name: "variance", points: sim.sweep.map((s) => [s.degree, s.variance]), color: COLORS.variance },
        { name: "noise σ²", points: sim.sweep.map((s) => [s.degree, s.noise2]), color: COLORS.noise, dashed: true },
        { name: "expected test error", points: sim.sweep.map((s) => [s.degree, s.total]), color: "var(--warn)" },
      ]} />
      <p className="mt-1 text-xs text-muted">Computed by refitting all {sim.cfg.sets} training sets at every degree. Dashed vertical line: your degree ({sim.cfg.degree}). Lowest total: degree {sim.best.degree}.</p>
    </div>
  );
}
