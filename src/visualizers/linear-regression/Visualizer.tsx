import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { animate, motion } from "motion/react";
import { RotateCcw, Shuffle, Sparkles } from "lucide-react";
import { usePlayback } from "../../hooks/usePlayback";
import { PlaybackControls } from "../../components/visualizer/PlaybackControls";
import { EventLog } from "../../components/visualizer/EventLog";
import { StepExplainer, Workspace } from "../../components/visualizer/Workspace";
import { AnimatedNumber } from "../../components/visualizer/AnimatedNumber";
import { Badge, Button, KV, Panel, SimulatedLabel, Slider, Toggle, cx } from "../../components/ui";
import { TexBlock } from "../../components/Tex";
import { fmt } from "../../lib/math";
import type { VisualizerTab } from "../types";
import { Axes, StepCaption, ease, scale } from "./kit";
import { DEFAULT_LR, predict, residuals, simulate, sse, type Fit, type LRConfig, type Phase, type Pt } from "./simulation";

const PHASES: Phase[] = ["data", "mean-x", "mean-y", "products", "slope", "intercept", "residuals", "squares", "r2"];
/** One schedule (seconds at 1×) for the canvas, the card and the counters. Multiply by k = 1/speed. */
const T = {
  drop: (i: number) => 0.2 + i * 0.12,
  slide: 0.2, slideDur: 1,
  rect: (i: number) => 0.3 + i * 0.28,
  rotate: 0.3, rotateDur: 1.4,
  extend: 0.2, extendDur: 1,
  seg: (i: number) => 0.2 + i * 0.16,
  sq: (i: number) => 0.2 + i * 0.22,
};
const stepMs = (phase: Phase, n: number) =>
  1000 * ({ data: T.drop(n) + 1.2, "mean-x": 2.4, "mean-y": 2.6, products: T.rect(n) + 1.6, slope: 2.8, intercept: 2.6, residuals: T.seg(n) + 1.4, squares: T.sq(n) + 1.8, r2: 3 } as const)[phase];

const W = 640, H = 360, L = 44, R = 628, TOP = 12, B = 318;

export default function LinearRegressionVisualizer({ tab }: { tab: VisualizerTab }) {
  const [cfg, setCfg] = useState<LRConfig>(DEFAULT_LR);
  const set = (p: Partial<LRConfig>) => setCfg((c) => ({ ...c, ...p }));
  const sim = useMemo(() => simulate(cfg), [cfg]);
  const n = sim.pts.length;
  const duration = useMemo(() => (i: number) => stepMs(PHASES[i] ?? "data", n), [n]);
  const pb = usePlayback(sim.steps.length, duration);
  const step = sim.steps[pb.index];
  const [replay, setReplay] = useState(0);
  const k = 1 / pb.speed;
  const animKey = `${step.id}-${replay}-${cfg.seed}-${cfg.outlier}-${cfg.n}-${cfg.noise}`;
  if (tab === "metrics") return null;
  const { fit } = sim;

  return (
    <Workspace
      main={
        <>
          <Panel title="Least squares, step by step" actions={<SimulatedLabel>Synthetic data (seeded)</SimulatedLabel>}>
            <StepCaption index={pb.index} total={sim.steps.length} stepKey={step.id} headline={step.state.headline} sub={step.state.sub} onReplay={() => setReplay((r) => r + 1)} />
            <Plot pts={sim.pts} fit={fit} phase={step.state.phase} k={k} animKey={animKey} />
            <Legend />
          </Panel>
          <Panel title="What's happening in this step">
            <motion.div key={animKey} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.2 }}>
              <Card phase={step.state.phase} pts={sim.pts} fit={fit} res={sim.res} k={k} />
            </motion.div>
          </Panel>
          <PlaybackControls pb={pb} stepTitles={sim.steps.map((s) => s.title)} />
          <Panel title="Data" actions={<Button size="sm" variant="ghost" onClick={() => { setCfg(DEFAULT_LR); pb.restart(); }}><RotateCcw size={12} /> Reset</Button>}>
            <div className="grid gap-3 sm:grid-cols-3">
              <Slider label="Points" value={cfg.n} min={3} max={14} onChange={(v) => set({ n: v })} />
              <Slider label="Noise (σ)" value={cfg.noise} min={0} max={3} step={0.1} format={(v) => v.toFixed(1)} onChange={(v) => set({ noise: v })} hint="0 = every point exactly on the hidden line y = 2 + 0.7x" />
              <div className="space-y-2">
                <Toggle label="Add an outlier at (8.5, 0.5)" checked={cfg.outlier} onChange={(v) => set({ outlier: v })} hint="Break it: one bad point drags the whole line." />
                <Button size="sm" onClick={() => set({ seed: cfg.seed + 1 })}><Shuffle size={12} /> New sample (seed {cfg.seed})</Button>
              </div>
            </div>
          </Panel>
          <ManualFit pts={sim.pts} fit={fit} />
        </>
      }
      inspector={
        <>
          <StepExplainer index={pb.index} title={step.title} what={step.description} />
          <Panel title="Calculation"><Calc phase={step.state.phase} fit={fit} /></Panel>
          <Panel title="Fitted values">
            <KV rows={[
              ["n", fit.n], ["x̄", fmt(fit.xm, 3)], ["ȳ", fmt(fit.ym, 3)], ["slope b", fmt(fit.slope, 4)], ["intercept a", fmt(fit.intercept, 4)],
              ["SSE", fmt(fit.sse, 3)], ["R²", fmt(fit.r2, 4)],
            ]} />
          </Panel>
          <Panel title="Events"><EventLog events={step.events} /></Panel>
        </>
      }
    />
  );
}

function useScales(pts: Pt[]) {
  const ys = pts.map((p) => p.y);
  const y0 = Math.floor(Math.min(0, ...ys) - 0.5), y1 = Math.ceil(Math.max(...ys, 1) + 1);
  return { sx: scale(0, 10, L, R), sy: scale(y0, y1, B, TOP), yDomain: [y0, y1] as [number, number] };
}

function Legend() {
  return (
    <div className="mt-2 flex flex-wrap gap-3 text-[11px] text-muted">
      <span className="flex items-center gap-1"><svg width="10" height="10"><circle cx="5" cy="5" r="4" fill="var(--accent)" /></svg>data point</span>
      <span className="flex items-center gap-1"><svg width="10" height="10"><path d="M1 1 L9 9 M9 1 L1 9" stroke="var(--err)" strokeWidth="2" /></svg>outlier</span>
      <span className="flex items-center gap-1"><svg width="22" height="6"><line x1="0" y1="3" x2="22" y2="3" stroke="var(--violet)" strokeWidth="2.5" /></svg>fitted line</span>
      <span className="flex items-center gap-1"><svg width="12" height="12"><rect x="1" y="1" width="10" height="10" fill="var(--warn)" fillOpacity="0.3" stroke="var(--warn)" /></svg>squared residual (area = e²)</span>
    </div>
  );
}

function PointMark({ x, y, outlier }: { x: number; y: number; outlier?: boolean }) {
  return outlier
    ? <path d={`M${x - 5} ${y - 5} L${x + 5} ${y + 5} M${x + 5} ${y - 5} L${x - 5} ${y + 5}`} stroke="var(--err)" strokeWidth={2.5} />
    : <circle cx={x} cy={y} r={5} fill="var(--accent)" stroke="var(--surface)" strokeWidth={1.5} />;
}

function Plot({ pts, fit, phase, k, animKey }: { pts: Pt[]; fit: Fit; phase: Phase; k: number; animKey: string }) {
  const { sx, sy, yDomain } = useScales(pts);
  const at = PHASES.indexOf(phase);
  const reached = (p: Phase) => at >= PHASES.indexOf(p);
  const t = (delay: number, duration = 0.5) => ({ delay: delay * k, duration: duration * k, ease });
  const xs = pts.map((p) => p.x);
  const [xa, xb] = [Math.min(...xs), Math.max(...xs)];
  const lineFrom = reached("intercept") ? 0 : xa;
  const lineTo = reached("residuals") ? 10 : xb;
  const res = residuals(pts, fit.slope, fit.intercept);
  const pxPerY = Math.abs(sy(1) - sy(0));
  return (
    <div className="overflow-x-auto">
      <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full min-w-[480px]" role="img" aria-label={`Scatter plot of ${pts.length} points with the least-squares line, phase ${phase}`}>
        <Axes sx={sx} sy={sy} xDomain={[0, 10]} yDomain={yDomain} L={L} R={R} T={TOP} B={B} xLabel="x" yLabel="y" />
        {/* Covariance rectangles from each point to the centre. */}
        {phase === "products" && pts.map((p, i) => {
          const pos = (p.x - fit.xm) * (p.y - fit.ym) >= 0;
          const x0 = Math.min(sx(p.x), sx(fit.xm)), y0 = Math.min(sy(p.y), sy(fit.ym));
          const w = Math.abs(sx(p.x) - sx(fit.xm)), h = Math.abs(sy(p.y) - sy(fit.ym));
          return (
            <motion.rect key={`${animKey}-r${i}`} x={x0} y={y0} width={w} height={h} fill={pos ? "var(--ok)" : "var(--err)"} stroke={pos ? "var(--ok)" : "var(--err)"}
              style={{ transformOrigin: `${sx(fit.xm)}px ${sy(fit.ym)}px`, transformBox: "view-box" }}
              initial={{ scale: 0, fillOpacity: 0.35 }} animate={{ scale: 1, fillOpacity: 0.16 }} transition={t(T.rect(i), 0.5)} />
          );
        })}
        {/* Squared residuals: real squares, side |e| in y-units. */}
        {(phase === "squares" || phase === "r2") && pts.map((p, i) => {
          const s = Math.abs(res[i]) * pxPerY;
          const left = sx(p.x) + s > R;
          const top = Math.min(sy(p.y), sy(predict(fit.slope, fit.intercept, p.x)));
          return (
            <motion.rect key={`${animKey}-s${i}`} y={top} height={s} fill="var(--warn)" fillOpacity={0.22} stroke="var(--warn)" strokeWidth={1}
              initial={phase === "squares" ? { width: 0, x: sx(p.x) } : false} animate={{ width: s, x: left ? sx(p.x) - s : sx(p.x) }} transition={t(T.sq(i), 0.5)} />
          );
        })}
        {/* SST: squares to the flat mean line, outlined for comparison. */}
        {phase === "r2" && pts.map((p, i) => {
          const s = Math.abs(p.y - fit.ym) * pxPerY;
          const left = sx(p.x) - s >= L;
          return (
            <motion.rect key={`${animKey}-t${i}`} y={Math.min(sy(p.y), sy(fit.ym))} height={s} fill="none" stroke="var(--muted)" strokeDasharray="3 3"
              initial={{ width: 0, x: sx(p.x) }} animate={{ width: s, x: left ? sx(p.x) - s : sx(p.x) }} transition={t(0.4 + i * 0.12, 0.6)} />
          );
        })}
        {/* Mean guides */}
        {reached("mean-x") && (
          <motion.line key={phase === "mean-x" ? `${animKey}-mx` : "mx"} y1={TOP} y2={B} stroke="var(--accent)" strokeDasharray="5 4" strokeOpacity={0.7}
            initial={phase === "mean-x" ? { x1: L, x2: L } : false} animate={{ x1: sx(fit.xm), x2: sx(fit.xm) }} transition={t(T.slide, T.slideDur)} />
        )}
        {reached("mean-y") && (
          <motion.line key={phase === "mean-y" ? `${animKey}-my` : "my"} x1={L} x2={R} stroke={phase === "r2" ? "var(--muted)" : "var(--accent)"} strokeDasharray="5 4" strokeOpacity={0.7}
            initial={phase === "mean-y" ? { y1: B, y2: B } : false} animate={{ y1: sy(fit.ym), y2: sy(fit.ym) }} transition={t(T.slide, T.slideDur)} />
        )}
        {phase === "r2" && <text x={R - 4} y={sy(fit.ym) - 5} textAnchor="end" fontSize={10} fill="var(--muted)">mean line ȳ (baseline for SST)</text>}
        {/* Residual segments */}
        {reached("residuals") && pts.map((p, i) => {
          const yh = sy(predict(fit.slope, fit.intercept, p.x));
          return (
            <motion.line key={`${phase === "residuals" ? animKey : "seg"}-${i}`} x1={sx(p.x)} x2={sx(p.x)} y1={yh} stroke={res[i] >= 0 ? "var(--ok)" : "var(--err)"} strokeWidth={2}
              initial={phase === "residuals" ? { y2: yh } : false} animate={{ y2: sy(p.y) }} transition={t(T.seg(i), 0.4)} />
          );
        })}
        {/* Fitted line: rotates around (x̄, ȳ), then extends to x = 0 */}
        {reached("slope") && (
          <motion.line key={phase === "slope" || phase === "intercept" ? `${animKey}-fit` : "fit"} stroke="var(--violet)" strokeWidth={3} strokeLinecap="round"
            initial={phase === "slope" ? { x1: sx(xa), x2: sx(xb), y1: sy(fit.ym), y2: sy(fit.ym) } : phase === "intercept" ? { x1: sx(xa), x2: sx(xb), y1: sy(predict(fit.slope, fit.intercept, xa)), y2: sy(predict(fit.slope, fit.intercept, xb)) } : false}
            animate={{ x1: sx(lineFrom), x2: sx(lineTo), y1: sy(predict(fit.slope, fit.intercept, lineFrom)), y2: sy(predict(fit.slope, fit.intercept, lineTo)) }}
            transition={phase === "slope" ? t(T.rotate, T.rotateDur) : t(T.extend, T.extendDur)} />
        )}
        {reached("intercept") && (
          <motion.g key={phase === "intercept" ? `${animKey}-icpt` : "icpt"} initial={phase === "intercept" ? { opacity: 0, scale: 0 } : false} animate={{ opacity: 1, scale: 1 }}
            style={{ transformOrigin: `${sx(0)}px ${sy(fit.intercept)}px`, transformBox: "view-box" }} transition={t(T.extend + T.extendDur, 0.4)}>
            <circle cx={sx(0)} cy={sy(fit.intercept)} r={6} fill="var(--violet)" stroke="var(--surface)" strokeWidth={2} />
            <text x={sx(0) + 9} y={sy(fit.intercept) - 8} fontSize={11} fontFamily="var(--font-mono)" fill="var(--violet)">a = {fit.intercept.toFixed(2)}</text>
          </motion.g>
        )}
        {/* Points */}
        {pts.map((p, i) => (
          <motion.g key={`${phase === "data" ? animKey : "pt"}-${i}`} initial={phase === "data" ? { opacity: 0, y: -40 } : false} animate={{ opacity: 1, y: 0 }}
            transition={{ delay: T.drop(i) * k, type: "spring", stiffness: 260, damping: 18 }}>
            <PointMark x={sx(p.x)} y={sy(p.y)} outlier={p.outlier} />
          </motion.g>
        ))}
        {/* Centre crosshair */}
        {reached("mean-y") && (
          <g>
            {phase === "mean-y" && (
              <motion.circle key={animKey} cx={sx(fit.xm)} cy={sy(fit.ym)} fill="none" stroke="var(--accent)" strokeWidth={2}
                initial={{ r: 8, opacity: 0 }} animate={{ r: [8, 26], opacity: [0.8, 0] }} transition={{ delay: (T.slide + T.slideDur) * k, duration: 1.2, repeat: Infinity }} />
            )}
            <motion.g key={phase === "mean-y" ? `${animKey}-c` : "c"} initial={phase === "mean-y" ? { opacity: 0 } : false} animate={{ opacity: 1 }} transition={t(T.slide + T.slideDur, 0.3)}>
              <circle cx={sx(fit.xm)} cy={sy(fit.ym)} r={7} fill="var(--surface)" stroke="var(--accent)" strokeWidth={2.5} />
              <circle cx={sx(fit.xm)} cy={sy(fit.ym)} r={2.5} fill="var(--accent)" />
              <text x={sx(fit.xm) + 10} y={sy(fit.ym) + 16} fontSize={11} fontFamily="var(--font-mono)" fill="var(--accent)">mean point ({fit.xm.toFixed(2)}, {fit.ym.toFixed(2)})</text>
            </motion.g>
          </g>
        )}
        {phase === "mean-x" && (
          <motion.text key={`${animKey}-xl`} y={TOP + 12} fontSize={11} fontFamily="var(--font-mono)" fill="var(--accent)" initial={{ x: L + 6, opacity: 0 }} animate={{ x: sx(fit.xm) + 6, opacity: 1 }} transition={t(T.slide, T.slideDur)}>
            x̄ = {fit.xm.toFixed(2)}
          </motion.text>
        )}
      </svg>
    </div>
  );
}

const Chip = ({ children, delay, k, tone = "accent" }: { children: ReactNode; delay: number; k: number; tone?: "accent" | "violet" }) => (
  <motion.span initial={{ opacity: 0, y: -10, scale: 0.8 }} animate={{ opacity: 1, y: 0, scale: 1 }} transition={{ delay: delay * k, duration: 0.35 * k, ease }}
    className={cx("inline-block rounded border px-1.5 py-0.5 font-mono text-[11px]", tone === "accent" ? "border-accent/40 bg-accent-soft" : "border-violet/40 bg-surface-2")}>
    {children}
  </motion.span>
);

function SignedBars({ rows, k, at, total }: { rows: { label: string; v: number; tone?: "ok" | "err" | "warn" }[]; k: number; at: (i: number) => number; total?: { label: string; v: number } }) {
  const m = Math.max(1e-9, ...rows.map((r) => Math.abs(r.v)));
  const anyNeg = rows.some((r) => r.v < 0);
  return (
    <div className="space-y-1">
      {rows.map((r, i) => (
        <div key={i} className="grid grid-cols-[88px_minmax(0,1fr)_56px] items-center gap-2 text-xs">
          <span className="truncate font-mono text-[10px] text-muted">{r.label}</span>
          <div className="relative h-3.5 rounded bg-surface-2">
            {anyNeg && <div className="absolute inset-y-0 left-1/2 w-px bg-line" />}
            <motion.div className={cx("absolute inset-y-0.5 rounded", (r.tone ?? (r.v >= 0 ? "ok" : "err")) === "ok" ? "bg-ok/70" : r.tone === "warn" ? "bg-warn/70" : "bg-err/70")}
              style={anyNeg ? (r.v < 0 ? { right: "50%" } : { left: "50%" }) : { left: 0 }}
              initial={{ width: 0 }} animate={{ width: `${(Math.abs(r.v) / m) * (anyNeg ? 50 : 100)}%` }} transition={{ delay: at(i) * k, duration: 0.4 * k, ease }} />
          </div>
          <span className="text-right font-mono text-[11px]"><AnimatedNumber value={r.v} from={0} delay={at(i) * k} duration={0.4 * k} /></span>
        </div>
      ))}
      {total && (
        <div className="mt-1 flex items-center justify-between border-t border-line pt-1 text-sm font-semibold">
          <span>{total.label}</span>
          <span className="font-mono"><AnimatedNumber value={total.v} from={0} delay={at(0) * k} duration={(at(rows.length - 1) - at(0) + 0.4) * k} digits={3} /></span>
        </div>
      )}
    </div>
  );
}

function Card({ phase, pts, fit, res, k }: { phase: Phase; pts: Pt[]; fit: Fit; res: number[]; k: number }) {
  const lbl = (p: Pt) => `(${p.x}, ${p.y})${p.outlier ? " !" : ""}`;
  if (phase === "data" || phase === "mean-x" || phase === "mean-y") {
    const which = phase === "mean-y" ? "y" : "x";
    const vals = pts.map((p) => (which === "x" ? p.x : p.y));
    const m = which === "x" ? fit.xm : fit.ym;
    return (
      <div className="space-y-2 text-sm">
        <div className="flex flex-wrap gap-1">{pts.map((p, i) => <Chip key={i} k={k} delay={phase === "data" ? T.drop(i) : 0.05 * i}>{phase === "data" ? lbl(p) : `${which}=${vals[i]}`}</Chip>)}</div>
        {phase === "data" ? (
          <p className="text-muted">Each chip is one (x, y) pair. The fitted line will predict y from x as <span className="font-mono">ŷ = a + b·x</span>.</p>
        ) : (
          <div className="flex flex-wrap items-baseline gap-2 font-mono">
            <span>Σ{which} = <AnimatedNumber value={vals.reduce((s, v) => s + v, 0)} from={0} delay={T.slide * k} duration={T.slideDur * k} /></span>
            <span className="text-muted">÷ n = {pts.length}</span>
            <span className="rounded bg-accent-soft px-2 py-0.5 text-base font-semibold">{which}̄ = <AnimatedNumber value={m} from={0} delay={T.slide * k} duration={T.slideDur * k} digits={3} /></span>
          </div>
        )}
      </div>
    );
  }
  if (phase === "products") {
    return (
      <div className="space-y-2">
        <p className="text-xs text-muted">(x − x̄)·(y − ȳ) for each point. Bars appear as the matching rectangle grows on the plot.</p>
        <SignedBars rows={pts.map((p) => ({ label: lbl(p), v: (p.x - fit.xm) * (p.y - fit.ym) }))} at={T.rect} k={k} total={{ label: "Sxy = sum", v: fit.sxy }} />
        <p className="font-mono text-xs">Sxx = Σ(x − x̄)² = {fit.sxx.toFixed(3)} <span className="text-muted">(how spread out x is)</span></p>
      </div>
    );
  }
  if (phase === "slope" || phase === "intercept") {
    return phase === "slope" ? (
      <div className="flex flex-wrap items-center gap-3 font-mono text-sm">
        <div className="text-center"><div className="border-b border-fg px-2">Sxy = {fit.sxy.toFixed(3)}</div><div className="px-2">Sxx = {fit.sxx.toFixed(3)}</div></div>
        <span>=</span>
        <span className="rounded bg-violet/15 px-2 py-1 text-lg font-semibold text-violet">b = <AnimatedNumber value={fit.slope} from={0} delay={T.rotate * k} duration={T.rotateDur * k} digits={4} /></span>
        <p className="w-full font-sans text-xs text-muted">For every +1 in x, the prediction changes by {fit.slope.toFixed(3)}. Co-movement divided by x-spread: how much of x's wiggle shows up in y.</p>
      </div>
    ) : (
      <div className="flex flex-wrap items-center gap-2 font-mono text-sm">
        <span>a = ȳ − b·x̄ = {fit.ym.toFixed(3)} − {fit.slope.toFixed(3)} × {fit.xm.toFixed(3)} =</span>
        <span className="rounded bg-violet/15 px-2 py-1 text-lg font-semibold text-violet"><AnimatedNumber value={fit.intercept} from={fit.ym} delay={T.extend * k} duration={T.extendDur * k} digits={4} /></span>
        <p className="w-full font-sans text-xs text-muted">The line: <strong className="font-mono">ŷ = {fit.intercept.toFixed(3)} + {fit.slope.toFixed(3)}·x</strong>. The intercept is its prediction at x = 0.</p>
      </div>
    );
  }
  if (phase === "residuals") {
    return (
      <div className="space-y-2">
        <p className="text-xs text-muted">e = y − ŷ for each point (green above the line, red below).</p>
        <SignedBars rows={pts.map((p, i) => ({ label: lbl(p), v: res[i] }))} at={T.seg} k={k} total={{ label: "Σe (cancels out)", v: Math.abs(res.reduce((s, r) => s + r, 0)) < 1e-9 ? 0 : res.reduce((s, r) => s + r, 0) }} />
      </div>
    );
  }
  if (phase === "squares") {
    const sq = res.map((r) => r * r);
    const worst = sq.indexOf(Math.max(...sq));
    return (
      <div className="space-y-2">
        <p className="text-xs text-muted">Each square's area e² stacks into SSE. Segment width = that point's share.</p>
        <div className="flex h-7 overflow-hidden rounded bg-surface-2">
          {sq.map((s, i) => (
            <motion.div key={i} className={cx("h-full border-r border-surface", pts[i].outlier ? "bg-err/70" : i % 2 ? "bg-warn/60" : "bg-warn/85")}
              initial={{ width: 0 }} animate={{ width: `${(100 * s) / Math.max(fit.sse, 1e-9)}%` }} transition={{ delay: T.sq(i) * k, duration: 0.5 * k, ease }} title={`${lbl(pts[i])}: e² = ${s.toFixed(3)}`} />
          ))}
        </div>
        <div className="flex flex-wrap justify-between gap-2 text-sm">
          <span className="font-mono font-semibold">SSE = <AnimatedNumber value={fit.sse} from={0} delay={T.sq(0) * k} duration={(T.sq(pts.length - 1) - T.sq(0) + 0.5) * k} digits={3} /></span>
          <span className="text-xs text-muted">largest square: {lbl(pts[worst])} = {((100 * sq[worst]) / Math.max(fit.sse, 1e-9)).toFixed(0)}% of SSE{pts[worst].outlier ? " (the outlier)" : ""}</span>
        </div>
      </div>
    );
  }
  const r2 = Number.isFinite(fit.r2) ? fit.r2 : 0;
  return (
    <div className="space-y-2 text-sm">
      {[["SST (flat mean line)", fit.sst, "bg-muted/50"], ["SSE (fitted line)", fit.sse, "bg-warn/70"]].map(([name, v, c], i) => (
        <div key={i} className="grid grid-cols-[130px_minmax(0,1fr)_60px] items-center gap-2 text-xs">
          <span>{name as string}</span>
          <div className="h-4 rounded bg-surface-2"><motion.div className={cx("h-full rounded", c as string)} initial={{ width: 0 }} animate={{ width: `${(100 * (v as number)) / Math.max(fit.sst, fit.sse, 1e-9)}%` }} transition={{ delay: (0.4 + i * 0.8) * k, duration: 0.7 * k, ease }} /></div>
          <span className="text-right font-mono">{(v as number).toFixed(2)}</span>
        </div>
      ))}
      <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 2 * k }}>
        R² = 1 − SSE/SST = <strong className="font-mono"><AnimatedNumber value={r2} from={0} delay={2 * k} duration={0.6 * k} digits={3} /></strong>. {r2 > 0.8 ? "The line removes most of the squared error of just guessing the mean." : r2 > 0.4 ? "A real but partial improvement over guessing the mean." : "Barely better than predicting ȳ for everyone."}
      </motion.p>
    </div>
  );
}

function Calc({ phase, fit }: { phase: Phase; fit: Fit }) {
  const f = (v: number, d = 3) => v.toFixed(d);
  const tex: Record<Phase, string> = {
    data: "\\hat{y} = a + b\\,x \\qquad \\min_{a,b} \\sum_i (y_i - a - b x_i)^2",
    "mean-x": `\\bar{x} = \\frac{1}{n}\\sum_i x_i = ${f(fit.xm)}`,
    "mean-y": `\\bar{y} = \\frac{1}{n}\\sum_i y_i = ${f(fit.ym)}`,
    products: `\\begin{aligned} S_{xy} &= \\textstyle\\sum (x_i-\\bar{x})(y_i-\\bar{y}) = ${f(fit.sxy)} \\\\ S_{xx} &= \\textstyle\\sum (x_i-\\bar{x})^2 = ${f(fit.sxx)} \\end{aligned}`,
    slope: `b = \\frac{S_{xy}}{S_{xx}} = \\frac{${f(fit.sxy)}}{${f(fit.sxx)}} = ${f(fit.slope, 4)}`,
    intercept: `a = \\bar{y} - b\\,\\bar{x} = ${f(fit.ym)} - ${fit.slope < 0 ? `(${f(fit.slope, 4)})` : f(fit.slope, 4)} \\cdot ${f(fit.xm)} = ${f(fit.intercept, 4)}`,
    residuals: `e_i = y_i - \\hat{y}_i = y_i - (${f(fit.intercept)} + ${f(fit.slope)}\\,x_i)`,
    squares: `\\text{SSE} = \\sum_i e_i^2 = ${f(fit.sse)}`,
    r2: `R^2 = 1 - \\frac{\\text{SSE}}{\\text{SST}} = 1 - \\frac{${f(fit.sse)}}{${f(fit.sst)}} = ${f(fit.r2, 4)}`,
  };
  return <TexBlock>{tex[phase]}</TexBlock>;
}

/** Learner-controlled line: drag the sliders and watch the squares, or press Fit to glide to the OLS answer. */
function ManualFit({ pts, fit }: { pts: Pt[]; fit: Fit }) {
  const [line, setLine] = useState({ slope: 0, intercept: Math.round(fit.ym * 10) / 10 });
  const anim = useRef<{ stop: () => void } | null>(null);
  useEffect(() => () => anim.current?.stop(), []);
  const { sx, sy, yDomain } = useScales(pts);
  const res = residuals(pts, line.slope, line.intercept);
  const mine = sse(pts, line.slope, line.intercept);
  const pxPerY = Math.abs(sy(1) - sy(0));
  const ratio = mine / Math.max(fit.sse, 1e-9);
  const fitNow = () => {
    anim.current?.stop();
    const from = { ...line };
    anim.current = animate(0, 1, {
      duration: 1.4, ease,
      onUpdate: (u) => setLine({ slope: from.slope + (fit.slope - from.slope) * u, intercept: from.intercept + (fit.intercept - from.intercept) * u }),
    });
  };
  const setManual = (p: Partial<typeof line>) => { anim.current?.stop(); setLine((l) => ({ ...l, ...p })); };
  const best = Math.abs(mine - fit.sse) < 1e-3 * Math.max(1, fit.sse);
  return (
    <Panel title="Your turn: fit the line by hand" actions={<Badge tone={best ? "ok" : "neutral"}>{best ? "✓ at the minimum" : `SSE ${ratio.toFixed(2)}× the minimum`}</Badge>}>
      <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_220px]">
        <div className="overflow-x-auto">
          <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full min-w-[480px]" role="img" aria-label={`Your line: slope ${line.slope.toFixed(2)}, intercept ${line.intercept.toFixed(2)}, SSE ${mine.toFixed(2)}`}>
            <Axes sx={sx} sy={sy} xDomain={[0, 10]} yDomain={yDomain} L={L} R={R} T={TOP} B={B} xLabel="x" yLabel="y" />
            {pts.map((p, i) => {
              const s = Math.abs(res[i]) * pxPerY;
              const left = sx(p.x) + s > R;
              const yh = sy(predict(line.slope, line.intercept, p.x));
              return (
                <g key={i}>
                  <rect x={left ? sx(p.x) - s : sx(p.x)} y={Math.min(sy(p.y), yh)} width={s} height={s} fill="var(--warn)" fillOpacity={0.22} stroke="var(--warn)" />
                  <line x1={sx(p.x)} x2={sx(p.x)} y1={yh} y2={sy(p.y)} stroke={res[i] >= 0 ? "var(--ok)" : "var(--err)"} strokeWidth={2} />
                </g>
              );
            })}
            <line x1={sx(0)} x2={sx(10)} y1={sy(predict(fit.slope, fit.intercept, 0))} y2={sy(predict(fit.slope, fit.intercept, 10))} stroke="var(--muted)" strokeDasharray="4 4" strokeOpacity={0.6} />
            <line x1={sx(0)} x2={sx(10)} y1={sy(line.intercept)} y2={sy(predict(line.slope, line.intercept, 10))} stroke="var(--violet)" strokeWidth={3} strokeLinecap="round" />
            {pts.map((p, i) => <PointMark key={i} x={sx(p.x)} y={sy(p.y)} outlier={p.outlier} />)}
          </svg>
        </div>
        <div className="space-y-3">
          <Slider label="Slope b" value={Number(line.slope.toFixed(2))} min={-1} max={2} step={0.01} format={(v) => v.toFixed(2)} onChange={(v) => setManual({ slope: v })} />
          <Slider label="Intercept a" value={Number(line.intercept.toFixed(2))} min={-4} max={10} step={0.05} format={(v) => v.toFixed(2)} onChange={(v) => setManual({ intercept: v })} />
          <div className="space-y-1 text-xs">
            <div className="flex justify-between"><span>Your SSE</span><span className="font-mono">{mine.toFixed(2)}</span></div>
            <div className="h-3 rounded bg-surface-2"><div className="h-full rounded bg-warn/70 transition-[width] duration-150" style={{ width: `${Math.min(100, 100 / Math.max(ratio, 1))}%` }} /></div>
            <div className="flex justify-between text-muted"><span>OLS minimum (dashed line)</span><span className="font-mono">{fit.sse.toFixed(2)}</span></div>
            <p className="text-[11px] text-muted">Bar full = you matched the minimum. Every other line has more total square area.</p>
          </div>
          <div className="flex flex-wrap gap-1.5">
            <Button size="sm" variant="primary" onClick={fitNow}><Sparkles size={12} /> Fit (animate to OLS)</Button>
            <Button size="sm" onClick={() => setManual({ slope: 0, intercept: Math.round(fit.ym * 10) / 10 })}>Flat line at ȳ</Button>
          </div>
        </div>
      </div>
    </Panel>
  );
}
