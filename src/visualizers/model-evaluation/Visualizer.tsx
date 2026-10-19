import { useMemo, useState } from "react";
import { RotateCcw } from "lucide-react";
import { motion } from "motion/react";
import { usePlayback } from "../../hooks/usePlayback";
import { PlaybackControls } from "../../components/visualizer/PlaybackControls";
import { EventLog } from "../../components/visualizer/EventLog";
import { LineChart } from "../../components/visualizer/LineChart";
import { StepExplainer, Workspace } from "../../components/visualizer/Workspace";
import { fmtErr } from "./simulation";
import { Badge, Button, Callout, KV, Panel, Slider } from "../../components/ui";
import { Tex, TexBlock } from "../../components/Tex";
import type { VisualizerTab } from "../types";
import { MAX_DEGREE, simulate, trueFn, type EvalConfig, type EvalPhase, type Point } from "./simulation";

const LAMBDAS = [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1];
const DEFAULT: EvalConfig = { n: 24, noise: 0.3, degree: 3, testFraction: 0.3, lambda: 0, seed: 11 };
const PRESETS: { label: string; cfg: Partial<EvalConfig> }[] = [
  { label: "Underfit (degree 1)", cfg: { degree: 1, lambda: 0 } },
  { label: "Good fit (degree 4)", cfg: { degree: 4, lambda: 0 } },
  { label: "Overfit (degree 14)", cfg: { degree: 14, lambda: 0 } },
  { label: "Overfit + ridge", cfg: { degree: 14, lambda: 1e-3 } },
];

export default function ModelEvaluationVisualizer({ tab }: { tab: VisualizerTab }) {
  const [cfg, setCfg] = useState(DEFAULT);
  const set = (p: Partial<EvalConfig>) => setCfg((c) => ({ ...c, ...p }));
  const sim = useMemo(() => simulate(cfg), [cfg]);
  const pb = usePlayback(sim.steps.length);
  const step = sim.steps[pb.index];

  if (tab === "metrics") {
    return (
      <div className="space-y-3">
        <Panel title="Error vs. model complexity (same data, every degree refit)">
          <LineChart
            logY
            markerX={cfg.degree}
            xLabel="polynomial degree"
            yLabel="MSE"
            series={[
              { name: "training MSE", points: sim.sweep.map((s) => [s.degree, s.trainMse]), color: "var(--accent)" },
              { name: "test MSE", points: sim.sweep.map((s) => [s.degree, s.testMse]), color: "var(--warn)" },
              { name: "noise floor σ²", points: [[1, cfg.noise ** 2], [MAX_DEGREE, cfg.noise ** 2]], color: "var(--muted)", dashed: true },
            ]}
          />
          <p className="mt-2 text-xs text-muted">Dashed vertical line: your current degree. Training error keeps falling with complexity; test error falls, bottoms out (best here: degree {sim.bestDegree}), then rises.</p>
        </Panel>
        <MetricsTable sim={sim} />
      </div>
    );
  }

  return (
    <Workspace
      main={
        <>
          <Panel title="Data and model" actions={<span className="flex gap-1"><Badge tone={sim.verdict === "good fit" ? "ok" : "warn"}>{sim.verdict}</Badge></span>}>
            <FitPlot train={sim.train} test={sim.test} predict={sim.model.predict} phase={step.state.phase} />
          </Panel>
          <PlaybackControls pb={pb} stepTitles={sim.steps.map((s) => s.title)} />
          <Panel title="Parameters" actions={<Button size="sm" variant="ghost" onClick={() => setCfg(DEFAULT)}><RotateCcw size={12} /> Reset</Button>}>
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              <Slider label="Model complexity (degree)" value={cfg.degree} min={1} max={MAX_DEGREE} onChange={(v) => set({ degree: v })} />
              <Slider label="Dataset size" value={cfg.n} min={10} max={200} step={2} onChange={(v) => set({ n: v })} />
              <Slider label={<>Noise <Tex>{"\\sigma"}</Tex></>} value={cfg.noise} min={0} max={1} step={0.05} format={(v) => v.toFixed(2)} onChange={(v) => set({ noise: v })} />
              <Slider label="Test split" value={cfg.testFraction} min={0.1} max={0.5} step={0.05} format={(v) => `${Math.round(v * 100)}%`} onChange={(v) => set({ testFraction: v })} />
              <Slider label={<>Ridge <Tex>{"\\lambda"}</Tex></>} value={LAMBDAS.indexOf(cfg.lambda)} min={0} max={LAMBDAS.length - 1} format={(i) => (LAMBDAS[i] === 0 ? "off" : String(LAMBDAS[i]))} onChange={(i) => set({ lambda: LAMBDAS[i] })} />
              <Slider label="Random seed" value={cfg.seed} min={1} max={50} onChange={(v) => set({ seed: v })} hint="Same seed → same data and split." />
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
            why="A model is only useful on data it has not seen. Training error measures memorization; held-out error estimates generalization." />
          <Panel title="Outputs">
            <KV rows={[
              [<Tex>{"\\text{MSE}_{\\text{train}}"}</Tex>, <span data-testid="train-mse">{fmtErr(sim.trainMse)}</span>],
              [<Tex>{"\\text{MSE}_{\\text{test}}"}</Tex>, <span data-testid="test-mse">{fmtErr(sim.testMse)}</span>],
              [<>gap <Tex>{"\\text{MSE}_{\\text{test}} - \\text{MSE}_{\\text{train}}"}</Tex></>, fmtErr((sim.testMse - sim.trainMse))],
              [<Tex>{"R^2_{\\text{train}}"}</Tex>, sim.trainR2.toFixed(3)],
              [<Tex>{"R^2_{\\text{test}}"}</Tex>, fmtErr(sim.testR2)],
              [<>noise floor <Tex>{"\\sigma^2"}</Tex></>, (cfg.noise ** 2).toFixed(4)],
              ["best degree (sweep)", sim.bestDegree],
            ]} />
          </Panel>
          <Panel title="Formulas">
            <TexBlock>{`\\begin{aligned}
\\hat{y}(x) &= \\sum_{k=0}^{${cfg.degree}} w_k\\, x^k \\\\
\\min_{w}\\; &\\sum_{i} (y_i - \\hat{y}_i)^2 + \\lambda\\, n \\sum_{k=1}^{${cfg.degree}} w_k^2 \\quad (\\lambda = ${cfg.lambda}) \\\\
\\text{MSE} &= \\frac{1}{n} \\sum_{i} (y_i - \\hat{y}_i)^2 \\\\
R^2 &= 1 - \\frac{SS_{\\text{res}}}{SS_{\\text{tot}}}
\\end{aligned}`}</TexBlock>
            <p className="mt-2 text-[11px] text-muted">Solved exactly via the normal equations <Tex>{"(X^\\top X + \\lambda n I')\\,w = X^\\top y"}</Tex>. When <Tex>{"\\lambda"}</Tex> is off a 1e-10 jitter keeps the system solvable.</p>
          </Panel>
          <Panel title="Events"><EventLog events={step.events} /></Panel>
        </>
      }
    />
  );
}

function FitPlot({ train, test, predict, phase }: { train: Point[]; test: Point[]; predict: (x: number) => number; phase: EvalPhase }) {
  const W = 600, H = 320, P = 30, YR = 2.5;
  const sx = (x: number) => P + ((x + 1) / 2) * (W - 2 * P);
  const sy = (y: number) => H / 2 - (Math.max(-50, Math.min(50, y)) / YR) * (H / 2 - P);
  const order: EvalPhase[] = ["data", "split", "fit", "train-eval", "test-eval", "diagnose"];
  const at = (p: EvalPhase) => order.indexOf(phase) >= order.indexOf(p);
  // Always 241 points, so Motion can morph one curve into the next when the model changes.
  const curve = (f: (x: number) => number) => Array.from({ length: 241 }, (_, i) => { const x = -1 + i / 120; return `${i ? "L" : "M"}${sx(x).toFixed(1)} ${sy(f(x)).toFixed(1)}`; }).join(" ");
  const showTrainRes = phase === "train-eval";
  const showTestRes = phase === "test-eval" || phase === "diagnose";
  return (
    <figure>
      <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label="Scatter plot of training and test points with fitted curve">
        <line x1={P} x2={W - P} y1={H / 2} y2={H / 2} stroke="var(--line)" />
        <line x1={W / 2} x2={W / 2} y1={P / 2} y2={H - P / 2} stroke="var(--line)" />
        <clipPath id="fit-clip"><rect x={P} y={P / 2} width={W - 2 * P} height={H - P} /></clipPath>
        <g clipPath="url(#fit-clip)">
        <path d={curve(trueFn)} fill="none" stroke="var(--muted)" strokeDasharray="5 4" strokeWidth={1.5} />
        {at("fit") && <motion.path initial={{ pathLength: 0, d: curve(predict) }} animate={{ pathLength: 1, d: curve(predict) }} transition={{ pathLength: { duration: 1.1, ease: "easeInOut" }, d: { duration: 0.6, ease: "easeInOut" } }} fill="none" stroke="var(--accent)" strokeWidth={2.5} />}
        {showTrainRes && train.map((p, i) => <motion.line key={i} x1={sx(p.x)} x2={sx(p.x)} y1={sy(p.y)} initial={{ y2: sy(p.y) }} animate={{ y2: sy(predict(p.x)) }} transition={{ delay: i * 0.02, duration: 0.4 }} stroke="var(--accent)" strokeOpacity={0.6} />)}
        {showTestRes && test.map((p, i) => <motion.line key={i} x1={sx(p.x)} x2={sx(p.x)} y1={sy(p.y)} initial={{ y2: sy(p.y) }} animate={{ y2: sy(predict(p.x)) }} transition={{ delay: i * 0.04, duration: 0.4 }} stroke="var(--warn)" strokeWidth={1.5} />)}
        {train.map((p, i) => <motion.circle key={`tr${i}`} initial={false} animate={{ cx: sx(p.x), cy: sy(p.y), fill: at("split") ? "var(--accent)" : "var(--muted)" }} transition={{ duration: 0.5 }} r={4} />)}
        {test.map((p, i) => at("split")
          ? <rect key={`te${i}`} x={sx(p.x) - 4} y={sy(p.y) - 4} width={8} height={8} fill="var(--surface)" stroke="var(--warn)" strokeWidth={2} />
          : <circle key={`te${i}`} cx={sx(p.x)} cy={sy(p.y)} r={4} fill="var(--muted)" />)}
        </g>
      </svg>
      <figcaption className="mt-1 flex flex-wrap gap-3 text-[11px] text-muted">
        <span>● training point</span><span>□ test point (held out)</span><span>— fitted model</span><span>- - true function sin(3x)</span><span>│ residual</span><span>curve clipped to y ∈ [−2.5, 2.5]</span>
      </figcaption>
    </figure>
  );
}

function MetricsTable({ sim }: { sim: ReturnType<typeof simulate> }) {
  return (
    <Panel title="Per-degree results">
      <Callout>Every row refits the model on the same training split. Values are computed, not illustrative.</Callout>
      <div className="mt-2 overflow-x-auto">
        <table className="w-full text-xs">
          <thead className="text-left text-muted"><tr><th className="py-1">degree</th><th>train MSE</th><th>test MSE</th><th>gap</th></tr></thead>
          <tbody className="font-mono">
            {sim.sweep.map((s) => (
              <tr key={s.degree} className={s.degree === sim.bestDegree ? "border-t border-line bg-ok-soft" : "border-t border-line"}>
                <td className="py-1">{s.degree}{s.degree === sim.bestDegree && " (best test)"}</td><td>{fmtErr(s.trainMse)}</td><td>{fmtErr(s.testMse)}</td><td>{fmtErr((s.testMse - s.trainMse))}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}
