import { useMemo, useState } from "react";
import { motion } from "motion/react";
import { RotateCcw, Shuffle } from "lucide-react";
import { usePlayback } from "../../hooks/usePlayback";
import { PlaybackControls } from "../../components/visualizer/PlaybackControls";
import { EventLog } from "../../components/visualizer/EventLog";
import { StepExplainer, Workspace } from "../../components/visualizer/Workspace";
import { AnimatedNumber } from "../../components/visualizer/AnimatedNumber";
import { KV, Panel, Select, SimulatedLabel, Slider, Button, cx } from "../../components/ui";
import { TexBlock } from "../../components/Tex";
import { fmt, sigmoid } from "../../lib/math";
import type { VisualizerTab } from "../types";
import { StepCaption, ease, scale } from "../linear-regression/kit";
import { BOX, DEFAULT_LOG, PRESETS, boundary, prob, simulate, type Epoch, type LogConfig, type LogState, type Params, type Preset, type Pt } from "./simulation";

/** One schedule (seconds at 1×) for boundary, shading, loss curve, sigmoid dots and counters. */
const T = { pop: (i: number) => 0.15 + i * 0.05, shade: 0.2, line: 0.6, move: 0.2, moveDur: 1.1 };
const STEP_MS = { data: 2200, init: 2600, epoch: 2400, result: 2600 } as const;
const LRS = [0.05, 0.2, 0.5, 2, 6, 15];

const PW = 400, PP = 22, G = 20;
const sx = scale(-BOX, BOX, PP, PW - PP), sy = scale(-BOX, BOX, PW - PP, PP);

export default function LogisticRegressionVisualizer({ tab }: { tab: VisualizerTab }) {
  const [cfg, setCfg] = useState<LogConfig>(DEFAULT_LOG);
  const set = (p: Partial<LogConfig>) => setCfg((c) => ({ ...c, ...p }));
  const sim = useMemo(() => simulate(cfg), [cfg]);
  const duration = useMemo(() => (i: number) => STEP_MS[sim.steps[i]?.state.kind ?? "epoch"], [sim]);
  const pb = usePlayback(sim.steps.length, duration);
  const step = sim.steps[pb.index];
  const [replay, setReplay] = useState(0);
  const k = 1 / pb.speed;
  const animKey = `${step.id}-${replay}-${JSON.stringify(cfg)}`;
  if (tab === "metrics") return null;
  const s = step.state;
  const cur = sim.history[s.epoch];
  const prev = sim.history[s.prevEpoch];

  return (
    <Workspace
      main={
        <>
          <Panel title="Training by gradient descent" actions={<SimulatedLabel>Synthetic data (seeded)</SimulatedLabel>}>
            <StepCaption index={pb.index} total={sim.steps.length} stepKey={step.id} headline={s.headline} sub={s.sub} onReplay={() => setReplay((r) => r + 1)} />
            <div className="grid gap-3 md:grid-cols-[minmax(0,1fr)_250px]">
              <Plane pts={sim.pts} state={s} cur={cur} prev={prev} k={k} animKey={animKey} />
              <div className="space-y-2">
                <LossChart history={sim.history} state={s} k={k} animKey={animKey} />
                <KV rows={[
                  ["epoch", <AnimatedNumber key={`e-${animKey}`} value={cur.epoch} from={prev.epoch} digits={0} delay={T.move * k} duration={T.moveDur * k} />],
                  ["loss", <AnimatedNumber key={`l-${animKey}`} value={cur.loss} from={prev.loss} digits={4} delay={T.move * k} duration={T.moveDur * k} />],
                  ["accuracy", `${(cur.acc * 100).toFixed(0)}%`],
                ]} />
              </div>
            </div>
            <div className="mt-2 flex flex-wrap gap-3 text-[11px] text-muted">
              <span className="flex items-center gap-1"><svg width="10" height="10"><circle cx="5" cy="5" r="4" fill="var(--accent)" /></svg>class 1</span>
              <span className="flex items-center gap-1"><svg width="10" height="10"><rect x="1" y="1" width="8" height="8" fill="var(--warn)" /></svg>class 0</span>
              <span className="flex items-center gap-1"><svg width="12" height="12"><circle cx="6" cy="6" r="5" fill="none" stroke="var(--err)" strokeWidth="1.5" strokeDasharray="2 2" /></svg>misclassified</span>
              <span>shading: blue = P(class 1) high, orange = low</span>
              <span className="flex items-center gap-1"><svg width="22" height="6"><line x1="0" y1="3" x2="22" y2="3" stroke="var(--fg)" strokeWidth="2.5" /></svg>boundary (P = 0.5)</span>
            </div>
          </Panel>
          <Panel title="What's happening in this step">
            <motion.div key={animKey} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.2 }}>
              <Card pts={sim.pts} state={s} cur={cur} prev={prev} lr={sim.cfg.lr} k={k} />
            </motion.div>
          </Panel>
          <PlaybackControls pb={pb} stepTitles={sim.steps.map((x) => x.title)} />
          <Panel title="Parameters" actions={<Button size="sm" variant="ghost" onClick={() => { setCfg(DEFAULT_LOG); pb.restart(); }}><RotateCcw size={12} /> Reset</Button>}>
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              <Select label="Dataset" value={cfg.preset} options={PRESETS.map((p) => ({ value: p.id, label: p.label }))} onChange={(v: Preset) => set({ preset: v })} />
              <Slider label="Learning rate η" value={Math.max(0, LRS.indexOf(cfg.lr))} min={0} max={LRS.length - 1} format={(i) => `${LRS[i]}${LRS[i] >= 6 ? " (too high?)" : ""}`} onChange={(i) => set({ lr: LRS[i] })} />
              <Slider label="L2 regularization λ" value={cfg.lambda} min={0} max={0.5} step={0.01} format={(v) => v.toFixed(2)} onChange={(v) => set({ lambda: v })} hint="Penalizes large weights: softer boundary." />
              <Slider label="Epochs" value={cfg.epochs} min={10} max={200} step={10} onChange={(v) => set({ epochs: v })} />
            </div>
            <div className="mt-3 flex flex-wrap gap-1.5">
              <Button size="sm" onClick={() => set({ seed: cfg.seed + 1 })}><Shuffle size={12} /> New sample (seed {cfg.seed})</Button>
              <Button size="sm" onClick={() => set({ preset: "xor", epochs: 200 })}>Break it: XOR</Button>
              <Button size="sm" onClick={() => set({ preset: "overlap", lr: 15 })}>Break it: η = 15</Button>
            </div>
          </Panel>
        </>
      }
      inspector={
        <>
          <StepExplainer index={pb.index} title={step.title} what={step.description} />
          <Panel title="Calculation"><Calc state={s} cur={cur} prev={prev} lr={sim.cfg.lr} lambda={sim.cfg.lambda} /></Panel>
          <Panel title="Weights">
            <KV rows={[["w₁", fmt(cur.w1, 4)], ["w₂", fmt(cur.w2, 4)], ["b", fmt(cur.b, 4)], ["‖w‖", fmt(Math.hypot(cur.w1, cur.w2), 4)], ["∂L/∂w₁", fmt(cur.grad.w1, 4)], ["∂L/∂w₂", fmt(cur.grad.w2, 4)], ["∂L/∂b", fmt(cur.grad.b, 4)]]} />
          </Panel>
          <Panel title="Events"><EventLog events={step.events} /></Panel>
        </>
      }
    />
  );
}

/** Background cells shaded by σ(w·x + b); fill transitions are CSS so 400 cells stay cheap. */
function Shading({ w, k, visible }: { w: Params; k: number; visible: boolean }) {
  const cell = (PW - 2 * PP) / G;
  return (
    <g>
      {Array.from({ length: G * G }, (_, i) => {
        const cx = -BOX + ((i % G) + 0.5) * (2 * BOX / G), cy = -BOX + (Math.floor(i / G) + 0.5) * (2 * BOX / G);
        const p = prob(w, [cx, cy]);
        return (
          <rect key={i} x={sx(cx) - cell / 2} y={sy(cy) - cell / 2} width={cell} height={cell} shapeRendering="crispEdges" fill={p >= 0.5 ? "var(--accent)" : "var(--warn)"}
            fillOpacity={visible ? Math.abs(p - 0.5) * 0.7 : 0} style={{ transition: `fill-opacity ${T.moveDur * k}s ease ${T.move * k}s, fill ${T.moveDur * k}s ease ${T.move * k}s` }} />
        );
      })}
    </g>
  );
}

function Plane({ pts, state, cur, prev, k, animKey }: { pts: Pt[]; state: LogState; cur: Epoch; prev: Epoch; k: number; animKey: string }) {
  const seg = boundary(cur), pseg = boundary(prev);
  const show = state.kind !== "data";
  const toPx = (s: [[number, number], [number, number]]) => ({ x1: sx(s[0][0]), y1: sy(s[0][1]), x2: sx(s[1][0]), y2: sy(s[1][1]) });
  const norm = Math.hypot(cur.w1, cur.w2) || 1;
  const mid = seg ? [(seg[0][0] + seg[1][0]) / 2, (seg[0][1] + seg[1][1]) / 2] : null;
  return (
    <div className="mx-auto w-full max-w-[460px]">
      <svg viewBox={`0 0 ${PW} ${PW}`} className="h-auto w-full" role="img" aria-label={`Points and decision boundary at epoch ${cur.epoch}`}>
        <rect x={PP} y={PP} width={PW - 2 * PP} height={PW - 2 * PP} fill="var(--surface-2)" />
        <Shading w={cur} k={state.kind === "init" ? 1 : k} visible={show} />
        <line x1={sx(0)} x2={sx(0)} y1={PP} y2={PW - PP} stroke="var(--line)" />
        <line x1={PP} x2={PW - PP} y1={sy(0)} y2={sy(0)} stroke="var(--line)" />
        <text x={PW - PP - 2} y={sy(0) - 4} fontSize={10} textAnchor="end" fill="var(--muted)">x₁</text>
        <text x={sx(0) + 4} y={PP + 10} fontSize={10} fill="var(--muted)">x₂</text>
        {show && seg && (
          <motion.line key={state.kind === "init" ? `${animKey}-b` : "boundary"} stroke="var(--fg)" strokeWidth={2.5} strokeLinecap="round"
            initial={state.kind === "init" ? { ...toPx(seg), pathLength: 0 } : pseg ? toPx(pseg) : toPx(seg)}
            animate={{ ...toPx(seg), pathLength: 1 }}
            transition={state.kind === "init" ? { delay: T.line * k, duration: 0.8 * k, ease } : { delay: T.move * k, duration: T.moveDur * k, ease }} />
        )}
        <defs>
          <marker id="lr-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M0 0 L10 5 L0 10 Z" fill="var(--fg)" />
          </marker>
        </defs>
        {show && mid && (
          <motion.line stroke="var(--fg)" strokeWidth={2} markerEnd="url(#lr-arrow)" initial={false}
            animate={{ x1: sx(mid[0]), y1: sy(mid[1]), x2: sx(mid[0]) + (38 * cur.w1) / norm, y2: sy(mid[1]) - (38 * cur.w2) / norm }}
            transition={{ delay: T.move * k, duration: T.moveDur * k, ease }}>
            <title>w points toward class 1</title>
          </motion.line>
        )}
        {pts.map((p, i) => {
          const q = prob(cur, p.x);
          const wrong = show && (q >= 0.5 ? 1 : 0) !== p.y;
          const [x, y] = [sx(p.x[0]), sy(p.x[1])];
          return (
            <motion.g key={`${state.kind === "data" ? animKey : "pt"}-${i}`} initial={state.kind === "data" ? { scale: 0, opacity: 0 } : false} animate={{ scale: 1, opacity: 1 }}
              style={{ transformOrigin: `${x}px ${y}px`, transformBox: "view-box" }} transition={{ delay: T.pop(i) * k, type: "spring", stiffness: 300, damping: 18 }}>
              <title>{`class ${p.y}, P(class 1) = ${q.toFixed(3)}`}</title>
              {p.y ? <circle cx={x} cy={y} r={6} fill="var(--accent)" stroke="var(--surface)" strokeWidth={1.5} /> : <rect x={x - 5.5} y={y - 5.5} width={11} height={11} fill="var(--warn)" stroke="var(--surface)" strokeWidth={1.5} />}
              <circle cx={x} cy={y} r={11} fill="none" stroke="var(--err)" strokeWidth={1.5} strokeDasharray="3 2" opacity={wrong ? 1 : 0} style={{ transition: `opacity .3s ease ${(T.move + T.moveDur) * k}s` }} />
            </motion.g>
          );
        })}
      </svg>
    </div>
  );
}

function LossChart({ history, state, k, animKey }: { history: Epoch[]; state: LogState; k: number; animKey: string }) {
  const W = 250, H = 170, L = 34, R = 8, TP = 8, B = 26;
  const finite = history.map((h) => (Number.isFinite(h.loss) ? h.loss : Number.NaN));
  const top = Math.max(0.8, ...finite.filter(Number.isFinite).map((v) => Math.min(v, 5)));
  const E = history.length - 1;
  const x = scale(0, E, L, W - R), y = scale(0, top, H - B, TP);
  const yc = (v: number) => y(Math.min(Number.isFinite(v) ? v : top, top));
  const path = (a: number, b: number) => history.slice(a, b + 1).map((h, i) => `${i ? "L" : "M"}${x(h.epoch).toFixed(1)} ${yc(h.loss).toFixed(1)}`).join(" ");
  const show = state.kind !== "data";
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label={`Log-loss curve up to epoch ${state.epoch}`}>
      {[0, top / 2, top].map((t) => (
        <g key={t}>
          <line x1={L} x2={W - R} y1={y(t)} y2={y(t)} stroke="var(--line)" />
          <text x={L - 4} y={y(t) + 3} fontSize={9} textAnchor="end" fill="var(--muted)" fontFamily="var(--font-mono)">{t.toFixed(2)}</text>
        </g>
      ))}
      <line x1={L} x2={W - R} y1={y(Math.LN2)} y2={y(Math.LN2)} stroke="var(--muted)" strokeDasharray="3 3" />
      <text x={W - R} y={y(Math.LN2) - 3} fontSize={9} textAnchor="end" fill="var(--muted)">ln 2 = coin flip</text>
      <text x={L} y={H - 12} fontSize={9} fill="var(--muted)" fontFamily="var(--font-mono)">0</text>
      <text x={W - R} y={H - 12} fontSize={9} textAnchor="end" fill="var(--muted)" fontFamily="var(--font-mono)">{E}</text>
      <text x={(L + W) / 2} y={H - 2} fontSize={10} textAnchor="middle" fill="var(--muted)">epoch → log loss</text>
      {show && state.prevEpoch > 0 && <path d={path(0, state.prevEpoch)} fill="none" stroke="var(--violet)" strokeWidth={2} />}
      {show && state.epoch > state.prevEpoch && (
        <motion.path key={animKey} d={path(state.prevEpoch, state.epoch)} fill="none" stroke="var(--violet)" strokeWidth={2.5}
          initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ delay: T.move * k, duration: T.moveDur * k, ease: "linear" }} />
      )}
      {show && (
        <motion.circle r={4} fill="var(--violet)" stroke="var(--surface)" strokeWidth={1.5} initial={false}
          animate={{ cx: x(state.epoch), cy: yc(history[state.epoch].loss) }} transition={{ delay: T.move * k, duration: T.moveDur * k, ease: "linear" }} />
      )}
    </svg>
  );
}

/** Each point dropped onto the sigmoid at its score z: training slides class-1 points right and class-0 points left. */
function SigmoidDots({ pts, cur, prev, k, appear }: { pts: Pt[]; cur: Params; prev: Params; k: number; appear: boolean }) {
  const W = 320, H = 150, P = 18, Z = 8;
  const zx = scale(-Z, Z, P, W - P), py = scale(0, 1, H - P, P);
  const curve = Array.from({ length: 81 }, (_, i) => { const z = -Z + (2 * Z * i) / 80; return `${i ? "L" : "M"}${zx(z).toFixed(1)} ${py(sigmoid(z)).toFixed(1)}`; }).join(" ");
  const zOf = (w: Params, p: Pt) => Math.max(-Z, Math.min(Z, w.w1 * p.x[0] + w.w2 * p.x[1] + w.b));
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full max-w-[340px]" role="img" aria-label="Every point placed on the sigmoid curve at its score z">
      <line x1={P} x2={W - P} y1={py(0.5)} y2={py(0.5)} stroke="var(--line)" strokeDasharray="3 3" />
      <line x1={zx(0)} x2={zx(0)} y1={P} y2={H - P} stroke="var(--line)" />
      <text x={W - P} y={py(0.5) - 4} fontSize={9} textAnchor="end" fill="var(--muted)">P = 0.5</text>
      <text x={W - P} y={H - 4} fontSize={9} textAnchor="end" fill="var(--muted)">z = w·x + b →</text>
      <text x={P} y={P - 6} fontSize={9} fill="var(--muted)">σ(z) = P(class 1)</text>
      <motion.path d={curve} fill="none" stroke="var(--violet)" strokeWidth={2} initial={appear ? { pathLength: 0 } : false} animate={{ pathLength: 1 }} transition={{ duration: 0.7 * k }} />
      {pts.map((p, i) => {
        const [z0, z1] = [zOf(prev, p), zOf(cur, p)];
        const jitter = (i % 5) - 2;
        return p.y ? (
          <motion.circle key={i} r={4.5} fill="var(--accent)" stroke="var(--surface)" initial={{ cx: zx(z0), cy: py(sigmoid(z0)) + jitter, opacity: appear ? 0 : 1 }}
            animate={{ cx: zx(z1), cy: py(sigmoid(z1)) + jitter, opacity: 1 }} transition={{ delay: (appear ? 0.6 + i * 0.03 : T.move) * k, duration: T.moveDur * k, ease }} />
        ) : (
          <motion.rect key={i} width={8} height={8} fill="var(--warn)" stroke="var(--surface)" initial={{ x: zx(z0) - 4, y: py(sigmoid(z0)) - 4 + jitter, opacity: appear ? 0 : 1 }}
            animate={{ x: zx(z1) - 4, y: py(sigmoid(z1)) - 4 + jitter, opacity: 1 }} transition={{ delay: (appear ? 0.6 + i * 0.03 : T.move) * k, duration: T.moveDur * k, ease }} />
        );
      })}
    </svg>
  );
}

function Card({ pts, state, cur, prev, lr, k }: { pts: Pt[]; state: LogState; cur: Epoch; prev: Epoch; lr: number; k: number }) {
  if (state.kind === "data") {
    const ones = pts.filter((p) => p.y).length;
    return (
      <div className="flex flex-wrap items-center gap-3 text-sm">
        {[["class 1", ones, "bg-accent"], ["class 0", pts.length - ones, "bg-warn"]].map(([n, c, bg], i) => (
          <motion.div key={i} initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: (0.3 + i * 0.4) * k, duration: 0.4 * k, ease }} className="flex items-center gap-2 rounded-lg border border-line px-3 py-2">
            <span className={cx("h-3 w-3", bg as string, i ? "" : "rounded-full")} /> <span className="font-mono">{c as number}</span> {n as string} points
          </motion.div>
        ))}
        <p className="text-muted">Model: <span className="font-mono">P(class 1) = σ(w₁x₁ + w₂x₂ + b)</span>. Training = finding w₁, w₂, b.</p>
      </div>
    );
  }
  const params: [string, keyof Params][] = [["w₁", "w1"], ["w₂", "w2"], ["b", "b"]];
  const ps = pts.map((p) => ({ y: p.y, q: prob(cur, p.x), q0: prob(prev, p.x) }));
  return (
    <div className="grid items-start gap-4 lg:grid-cols-[340px_minmax(0,1fr)]">
      <SigmoidDots pts={pts} cur={cur} prev={prev} k={k} appear={state.kind === "init"} />
      <div className="space-y-3">
        {state.kind === "epoch" && (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead className="text-left text-muted"><tr><th className="py-0.5">param</th><th>before</th><th>gradient (avg.)</th><th>after</th></tr></thead>
              <tbody className="font-mono">
                {params.map(([label, key]) => (
                  <tr key={key} className="border-t border-line">
                    <td className="py-1">{label}</td>
                    <td>{prev[key].toFixed(3)}</td>
                    <td className={prev.grad[key] > 0 ? "text-err" : "text-ok"}>{prev.grad[key] > 0 ? "+" : ""}{prev.grad[key].toFixed(3)}</td>
                    <td className="font-semibold"><AnimatedNumber value={cur[key]} from={prev[key]} digits={3} delay={T.move * k} duration={T.moveDur * k} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="mt-1 text-[11px] text-muted">Each epoch: w ← w − {lr}·∇L. The gradient shown is at the start of this jump ({cur.epoch - prev.epoch} epoch{cur.epoch - prev.epoch > 1 ? "s" : ""}).</p>
          </div>
        )}
        <div>
          <div className="mb-1 flex justify-between text-[11px] text-muted"><span>P(class 1) per point · ▼ class-1 target is the top, ▲ class-0 target is 0</span><span>{ps.filter((p) => (p.q >= 0.5 ? 1 : 0) === p.y).length}/{ps.length} correct</span></div>
          <div className="flex h-16 items-end gap-0.5 rounded bg-surface-2 p-1">
            {ps.map((p, i) => (
              <div key={i} className="relative flex h-full flex-1 flex-col justify-end">
                <motion.div className={cx("w-full rounded-t-sm", p.y ? "bg-accent/80" : "bg-warn/80")} initial={{ height: `${(state.kind === "init" ? 0 : p.q0) * 100}%` }}
                  animate={{ height: `${p.q * 100}%` }} transition={{ delay: (state.kind === "init" ? 0.6 + i * 0.03 : T.move) * k, duration: T.moveDur * k, ease }} />
                <span className={cx("absolute left-1/2 -translate-x-1/2 text-[8px] leading-none", p.y ? "-top-0.5 text-accent" : "-bottom-2.5 text-warn")} aria-hidden>{p.y ? "▼" : "▲"}</span>
              </div>
            ))}
          </div>
          <p className="mt-3 text-[11px] text-muted">Blue bars (class 1) should rise to the top, orange bars (class 0) should shrink to 0. {state.kind === "result" && "Remaining gaps are the points the line cannot fix."}</p>
        </div>
      </div>
    </div>
  );
}

function Calc({ state, cur, prev, lr, lambda }: { state: LogState; cur: Epoch; prev: Epoch; lr: number; lambda: number }) {
  const f = (v: number, d = 3) => (v < 0 ? `(${v.toFixed(d)})` : v.toFixed(d));
  if (state.kind === "data") return <TexBlock>{"P(y{=}1 \\mid x) = \\sigma(w_1 x_1 + w_2 x_2 + b) = \\frac{1}{1 + e^{-(w^\\top x + b)}}"}</TexBlock>;
  if (state.kind === "init" || state.kind === "result")
    return (
      <div className="space-y-2">
        <TexBlock>{`\\mathcal{L} = -\\frac{1}{n}\\sum_i \\big[y_i \\log p_i + (1-y_i)\\log(1-p_i)\\big] + \\tfrac{\\lambda}{2}\\lVert w \\rVert^2 = ${cur.loss.toFixed(4)}`}</TexBlock>
        <TexBlock>{`\\nabla_w \\mathcal{L} = \\frac{1}{n}\\sum_i (p_i - y_i)\\,x_i + \\lambda w \\qquad \\lambda = ${lambda}`}</TexBlock>
      </div>
    );
  return (
    <TexBlock>{`\\begin{aligned} w_1 &\\leftarrow ${f(prev.w1)} - ${lr} \\cdot ${f(prev.grad.w1)} ${cur.epoch - prev.epoch > 1 ? "\\;\\cdots" : ""} = ${cur.w1.toFixed(3)} \\\\ w_2 &\\leftarrow ${f(prev.w2)} - ${lr} \\cdot ${f(prev.grad.w2)} ${cur.epoch - prev.epoch > 1 ? "\\;\\cdots" : ""} = ${cur.w2.toFixed(3)} \\\\ b &\\leftarrow ${f(prev.b)} - ${lr} \\cdot ${f(prev.grad.b)} ${cur.epoch - prev.epoch > 1 ? "\\;\\cdots" : ""} = ${cur.b.toFixed(3)} \\\\ \\mathcal{L} &: ${prev.loss.toFixed(4)} \\to ${cur.loss.toFixed(4)} \\end{aligned}`}</TexBlock>
  );
}
