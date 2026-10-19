import { useMemo, useState, type ReactNode } from "react";
import { motion } from "motion/react";
import { RotateCcw, Shuffle } from "lucide-react";
import { usePlayback } from "../../hooks/usePlayback";
import { PlaybackControls } from "../../components/visualizer/PlaybackControls";
import { EventLog } from "../../components/visualizer/EventLog";
import { StepExplainer, Workspace } from "../../components/visualizer/Workspace";
import { AnimatedNumber } from "../../components/visualizer/AnimatedNumber";
import { Badge, Button, KV, Panel, Select, SimulatedLabel, Slider, Toggle, cx } from "../../components/ui";
import { TexBlock } from "../../components/Tex";
import type { VisualizerTab } from "../types";
import { StepCaption, ease, scale } from "../linear-regression/kit";
import { DEFAULT_GD, SURFACES, f3, p3, simulate, type GDConfig, type GDState, type Iter, type LrPreset, type Regime, type Surface, type SurfaceId, type Vec } from "./simulation";

/** One schedule (seconds at 1×): gradient arrow → update arrow → ball + trail + loss chart. */
const T = { grad: 0.15, gradDur: 0.45, upd: 0.75, updDur: 0.4, ball: 1.25, ballDur: 0.6 };
const STEP_MS = { start: 1800, step: 2200, summary: 2600 } as const;
const PX = 56, M = 12;
const REGIME_TONE: Record<Regime, "ok" | "warn" | "err" | "neutral"> = { good: "ok", slow: "warn", oscillating: "warn", diverging: "err", noisy: "neutral" };
const PRESET_LABEL: Record<LrPreset, string> = { slow: "Slow", good: "Good", oscillating: "Oscillating", diverging: "Diverging" };

export default function GradientDescentVisualizer({ tab }: { tab: VisualizerTab }) {
  const [cfg, setCfg] = useState<GDConfig>(DEFAULT_GD);
  const set = (p: Partial<GDConfig>) => setCfg((c) => ({ ...c, ...p }));
  const sim = useMemo(() => simulate(cfg), [cfg]);
  const duration = useMemo(() => (i: number) => STEP_MS[sim.steps[i]?.state.kind ?? "step"], [sim]);
  const pb = usePlayback(sim.steps.length, duration);
  const step = sim.steps[pb.index];
  const [replay, setReplay] = useState(0);
  const k = 1 / pb.speed;
  const animKey = `${step.id}-${replay}-${JSON.stringify(cfg)}`;
  if (tab === "metrics") return null;
  const s = step.state;
  const surf = sim.surface;
  const a = sim.iters[Math.max(0, s.t - 1)], b = sim.iters[s.t];
  const changeSurface = (id: SurfaceId) => set({ surface: id, lr: SURFACES[id].lr.good });

  return (
    <Workspace
      main={
        <>
          <Panel title="Loss surface (contour view)" actions={<span className="flex gap-1.5"><Badge tone={REGIME_TONE[sim.regime]}>outcome: {sim.regime}</Badge>{cfg.stochastic && <SimulatedLabel>Gradient noise (seeded)</SimulatedLabel>}</span>}>
            <StepCaption index={pb.index} total={sim.steps.length} stepKey={step.id} headline={s.headline} sub={s.sub} onReplay={() => setReplay((r) => r + 1)} />
            <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_230px]">
              <Surface2D sim={sim} state={s} k={k} animKey={animKey} />
              <div className="space-y-2">
                <LossChart iters={sim.iters} t={s.t} k={k} animKey={animKey} kind={s.kind} />
                <KV rows={[
                  ["step", `${s.t} / ${sim.iters.length - 1}`],
                  ["position", `(${f3(b.p[0])}, ${f3(b.p[1])})`],
                  ["loss", <Num key={animKey} value={b.f} from={s.kind === "step" ? a.f : b.f} delay={T.ball * k} duration={T.ballDur * k} />],
                ]} />
              </div>
            </div>
            <div className="mt-2 flex flex-wrap gap-3 text-[11px] text-muted">
              <span>darker = lower loss</span>
              <span className="flex items-center gap-1"><svg width="22" height="8"><line x1="0" y1="4" x2="22" y2="4" stroke="var(--err)" strokeWidth="2.5" /></svg>gradient ∇f (points uphill)</span>
              <span className="flex items-center gap-1"><svg width="22" height="8"><line x1="0" y1="4" x2="22" y2="4" stroke="var(--ok)" strokeWidth="2.5" /></svg>update −η·∇f (actual move)</span>
              {cfg.momentum && <span className="flex items-center gap-1"><svg width="22" height="8"><line x1="0" y1="4" x2="22" y2="4" stroke="var(--muted)" strokeWidth="2" strokeDasharray="3 2" /></svg>move without momentum</span>}
              <span>★ minimum</span>
            </div>
          </Panel>
          <Panel title="What's happening in this step">
            <motion.div key={animKey} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.2 }}>
              <Card sim={sim} state={s} a={a} b={b} k={k} />
            </motion.div>
          </Panel>
          <PlaybackControls pb={pb} stepTitles={sim.steps.map((x) => x.title)} />
          <Panel title="Parameters" actions={<Button size="sm" variant="ghost" onClick={() => { setCfg(DEFAULT_GD); pb.restart(); }}><RotateCcw size={12} /> Reset</Button>}>
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
              <Select label="Loss surface" value={cfg.surface} options={Object.values(SURFACES).map((x) => ({ value: x.id, label: x.label }))} onChange={changeSurface} />
              <div className="space-y-1.5">
                <Slider label="Learning rate η" value={cfg.lr} min={surf.lr.slow / 2} max={+(surf.lr.diverging * 1.15).toFixed(2)} step={surf.id === "bowl" ? 0.005 : 0.01} format={(v) => v.toFixed(3)} onChange={(v) => set({ lr: v })} />
                <div className="flex flex-wrap gap-1">
                  {(Object.keys(PRESET_LABEL) as LrPreset[]).map((p) => (
                    <Button key={p} size="sm" variant={cfg.lr === surf.lr[p] ? "primary" : "secondary"} onClick={() => set({ lr: surf.lr[p] })}>{PRESET_LABEL[p]} {surf.lr[p]}</Button>
                  ))}
                </div>
              </div>
              <Slider label="Steps" value={cfg.steps} min={5} max={40} onChange={(v) => set({ steps: v })} />
              <div className="space-y-1.5">
                <Toggle label="Momentum" checked={cfg.momentum} onChange={(v) => set({ momentum: v })} hint="v ← β·v + ∇f, then move −η·v: past steps keep pushing." />
                {cfg.momentum && <Slider label="β (momentum)" value={cfg.beta} min={0.5} max={0.95} step={0.05} format={(v) => v.toFixed(2)} onChange={(v) => set({ beta: v })} />}
              </div>
              <div className="space-y-1.5">
                <Toggle label="Stochastic (noisy) gradient" checked={cfg.stochastic} onChange={(v) => set({ stochastic: v })} hint="Mimics a mini-batch: true gradient + seeded Gaussian noise." />
                {cfg.stochastic && (
                  <>
                    <Slider label="Noise σ" value={cfg.noise} min={0.5} max={10} step={0.5} format={(v) => v.toFixed(1)} onChange={(v) => set({ noise: v })} />
                    <Button size="sm" onClick={() => set({ seed: cfg.seed + 1 })}><Shuffle size={12} /> New noise (seed {cfg.seed})</Button>
                  </>
                )}
              </div>
            </div>
          </Panel>
        </>
      }
      inspector={
        <>
          <StepExplainer index={pb.index} title={step.title} what={step.description} />
          <Panel title="Calculation"><Calc sim={sim} state={s} a={a} b={b} /></Panel>
          <Panel title="Settings">
            <KV rows={[["surface", surf.id], ["η", cfg.lr], ["momentum", cfg.momentum ? `β = ${cfg.beta}` : "off"], ["gradient", cfg.stochastic ? `noisy, σ = ${cfg.noise}` : "exact (batch)"], ["outcome", sim.regime], ["stable if", surf.id === "bowl" ? "η < 2/10 = 0.2" : "η < 2/2 = 1 (near minima)"]]} />
          </Panel>
          <Panel title="Events"><EventLog events={step.events} /></Panel>
        </>
      }
    />
  );
}

function useView(s: Surface) {
  const W = (s.domain.x[1] - s.domain.x[0]) * PX + 2 * M, H = (s.domain.y[1] - s.domain.y[0]) * PX + 2 * M;
  const sx = scale(s.domain.x[0], s.domain.x[1], M, W - M), sy = scale(s.domain.y[0], s.domain.y[1], H - M, M);
  const clampP = (p: Vec): Vec => [Math.max(s.domain.x[0], Math.min(s.domain.x[1], p[0])), Math.max(s.domain.y[0], Math.min(s.domain.y[1], p[1]))];
  return { W, H, sx, sy, px: (p: Vec) => { const c = clampP(p); return [sx(c[0]), sy(c[1])] as Vec; } };
}

/** Banded heatmap of log(1 + f): 12 quantized levels read like contour rings. Static, memoized per surface. */
function Heatmap({ s }: { s: Surface }) {
  const { W, H } = useView(s);
  const cells = useMemo(() => {
    const NX = 64, NY = Math.round((NX * (s.domain.y[1] - s.domain.y[0])) / (s.domain.x[1] - s.domain.x[0]));
    const vals: number[] = [];
    for (let j = 0; j < NY; j++) for (let i = 0; i < NX; i++) vals.push(Math.log1p(s.f([s.domain.x[0] + ((i + 0.5) / NX) * (s.domain.x[1] - s.domain.x[0]), s.domain.y[0] + ((j + 0.5) / NY) * (s.domain.y[1] - s.domain.y[0])])));
    const max = Math.max(...vals);
    return { NX, NY, levels: vals.map((v) => Math.min(11, Math.floor((v / max) * 12))) };
  }, [s]);
  const cw = (W - 2 * M) / cells.NX, ch = (H - 2 * M) / cells.NY;
  return (
    <g>
      {cells.levels.map((lv, i) => (
        <rect key={i} x={M + (i % cells.NX) * cw} y={H - M - (Math.floor(i / cells.NX) + 1) * ch} width={cw} height={ch} fill="var(--violet)" fillOpacity={0.62 - lv * 0.052} shapeRendering="crispEdges" />
      ))}
    </g>
  );
}

function Arrow({ from, to, color, id, delay, dur, dashed, width = 2.5 }: { from: Vec; to: Vec; color: string; id: string; delay: number; dur: number; dashed?: boolean; width?: number }) {
  return (
    <motion.line stroke={color} strokeWidth={width} strokeDasharray={dashed ? "4 3" : undefined} markerEnd={`url(#${id})`} x1={from[0]} y1={from[1]}
      initial={{ x2: from[0], y2: from[1], opacity: 0 }} animate={{ x2: to[0], y2: to[1], opacity: 1 }} transition={{ delay, duration: dur, ease }} />
  );
}

function Surface2D({ sim, state, k, animKey }: { sim: ReturnType<typeof simulate>; state: GDState; k: number; animKey: string }) {
  const s = sim.surface;
  const { W, H, sx, sy, px } = useView(s);
  const it = sim.iters;
  const a = it[Math.max(0, state.t - 1)], b = it[state.t];
  const isStep = state.kind === "step";
  const trailTo = isStep ? state.t - 1 : state.t;
  const trail = it.slice(0, trailTo + 1).map((x) => px(x.p));
  const A = px(a.p), B = px(b.p);
  // Gradient arrow: direction of ∇f, length grows with |∇f| but stays readable (18–70 px).
  const gArrow = (g: Vec): Vec => { const n = Math.hypot(g[0], g[1]) || 1; const len = Math.max(18, Math.min(70, n * 5)); return [A[0] + (g[0] / n) * len, A[1] - (g[1] / n) * len]; };
  const plainMove: Vec = [a.p[0] - sim.cfg.lr * a.g[0], a.p[1] - sim.cfg.lr * a.g[1]];
  const off = Math.abs(b.p[0]) > s.domain.x[1] || Math.abs(b.p[1]) > s.domain.y[1];
  const markers = [["gd-g", "var(--err)"], ["gd-u", "var(--ok)"], ["gd-m", "var(--muted)"]] as const;
  return (
    <div className="overflow-x-auto">
      <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full min-w-[460px]" role="img" aria-label={`Contour plot of ${s.label}; ball at (${f3(b.p[0])}, ${f3(b.p[1])}) after ${state.t} steps`}>
        <defs>
          {markers.map(([id, c]) => (
            <marker key={id} id={id} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" fill={c} /></marker>
          ))}
        </defs>
        <motion.g initial={state.kind === "start" ? { opacity: 0 } : false} animate={{ opacity: 1 }} transition={{ duration: 0.6 * k }}><Heatmap s={s} /></motion.g>
        <line x1={sx(0)} x2={sx(0)} y1={M} y2={H - M} stroke="var(--fg)" strokeOpacity={0.15} />
        <line x1={M} x2={W - M} y1={sy(0)} y2={sy(0)} stroke="var(--fg)" strokeOpacity={0.15} />
        <text x={W - M - 4} y={sy(0) - 4} fontSize={11} textAnchor="end" fill="var(--fg)" fillOpacity={0.6}>x</text>
        <text x={sx(0) + 5} y={M + 12} fontSize={11} fill="var(--fg)" fillOpacity={0.6}>y</text>
        {s.minima.map((m, i) => <text key={i} x={sx(m[0])} y={sy(m[1]) + 6} fontSize={18} textAnchor="middle" fill="var(--warn)" stroke="var(--surface)" strokeWidth={0.6}>★</text>)}
        {s.saddle && <text x={sx(s.saddle[0]) + 8} y={sy(s.saddle[1]) - 6} fontSize={11} fill="var(--fg)" fillOpacity={0.75}>saddle (flat, not a minimum)</text>}
        {/* Trail so far */}
        {trail.length > 1 && <polyline points={trail.map((p) => p.join(",")).join(" ")} fill="none" stroke="var(--fg)" strokeOpacity={0.55} strokeWidth={1.5} />}
        {trail.map((p, i) => <circle key={i} cx={p[0]} cy={p[1]} r={2.5} fill="var(--fg)" fillOpacity={0.6} />)}
        {state.kind === "summary" && trail.length > 1 && (
          <motion.polyline key={animKey} points={trail.map((p) => p.join(",")).join(" ")} fill="none" stroke="var(--accent)" strokeWidth={3}
            initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 1.6 * k, ease: "linear" }} />
        )}
        {isStep && (
          <g key={animKey}>
            {sim.cfg.stochastic && <Arrow from={A} to={gArrow(a.gTrue)} color="var(--err)" id="gd-g" delay={T.grad * k} dur={T.gradDur * k} dashed width={1.5} />}
            <Arrow from={A} to={gArrow(a.g)} color="var(--err)" id="gd-g" delay={T.grad * k} dur={T.gradDur * k} />
            <motion.text x={gArrow(a.g)[0] + 4} y={gArrow(a.g)[1] - 4} fontSize={11} fontWeight={600} fill="var(--err)" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: (T.grad + T.gradDur) * k }}>
              ∇f{sim.cfg.stochastic ? " (noisy)" : ""}
            </motion.text>
            {sim.cfg.momentum && <Arrow from={A} to={px(plainMove)} color="var(--muted)" id="gd-m" delay={T.upd * k} dur={T.updDur * k} dashed width={1.5} />}
            <Arrow from={A} to={B} color="var(--ok)" id="gd-u" delay={T.upd * k} dur={T.updDur * k} width={3} />
            <motion.line x1={A[0]} y1={A[1]} stroke="var(--fg)" strokeOpacity={0.55} strokeWidth={1.5} initial={{ x2: A[0], y2: A[1] }} animate={{ x2: B[0], y2: B[1] }} transition={{ delay: T.ball * k, duration: T.ballDur * k, ease }} />
          </g>
        )}
        {/* The ball */}
        <motion.circle key={state.kind === "start" ? animKey : "ball"} r={8} fill="var(--accent)" stroke="var(--surface)" strokeWidth={2.5}
          initial={state.kind === "start" ? { cx: B[0], cy: B[1] - 40, opacity: 0 } : isStep ? { cx: A[0], cy: A[1] } : false}
          animate={{ cx: B[0], cy: B[1], opacity: 1 }} transition={state.kind === "start" ? { delay: 0.3 * k, type: "spring", stiffness: 200, damping: 14 } : { delay: (isStep ? T.ball : 0) * k, duration: T.ballDur * k, ease }} />
        {off && (
          <motion.text x={Math.min(W - 120, B[0] + 10)} y={Math.max(24, B[1] - 12)} fontSize={12} fontWeight={700} fill="var(--err)" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: (T.ball + T.ballDur) * k }}>
            off the chart ✗
          </motion.text>
        )}
      </svg>
    </div>
  );
}

function LossChart({ iters, t, k, animKey, kind }: { iters: Iter[]; t: number; k: number; animKey: string; kind: GDState["kind"] }) {
  const W = 230, H = 170, L = 36, R = 8, TP = 8, B = 26;
  const lg = (v: number) => Math.log10(Math.max(1e-4, Number.isFinite(v) ? v : 1e8));
  const vals = iters.map((i) => lg(i.f));
  const hi = Math.min(Math.max(...vals) + 0.2, 6), lo = Math.max(Math.min(...vals) - 0.2, -4);
  const N = Math.max(1, iters.length - 1);
  const x = scale(0, N, L, W - R), y = (v: number) => scale(lo, hi, H - B, TP)(Math.min(hi, lg(v)));
  const pts = iters.map((it) => [x(it.t), y(it.f)] as Vec);
  const prev = kind === "step" ? t - 1 : t;
  const ticks = Array.from({ length: Math.floor(hi) - Math.ceil(lo) + 1 }, (_, i) => Math.ceil(lo) + i).filter((_, i, a) => a.length <= 5 || i % 2 === 0);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label={`Loss versus step, log scale, up to step ${t}`}>
      {ticks.map((e) => (
        <g key={e}>
          <line x1={L} x2={W - R} y1={y(10 ** e)} y2={y(10 ** e)} stroke="var(--line)" />
          <text x={L - 4} y={y(10 ** e) + 3} fontSize={9} textAnchor="end" fill="var(--muted)" fontFamily="var(--font-mono)">{e >= 0 ? 10 ** e : `1e${e}`}</text>
        </g>
      ))}
      <text x={L} y={H - 12} fontSize={9} fill="var(--muted)" fontFamily="var(--font-mono)">0</text>
      <text x={W - R} y={H - 12} fontSize={9} textAnchor="end" fill="var(--muted)" fontFamily="var(--font-mono)">{N}</text>
      <text x={(L + W) / 2} y={H - 2} fontSize={10} textAnchor="middle" fill="var(--muted)">step → loss (log scale)</text>
      {prev > 0 && <polyline points={pts.slice(0, prev + 1).map((p) => p.join(",")).join(" ")} fill="none" stroke="var(--accent)" strokeWidth={2} />}
      {kind === "step" && (
        <motion.line key={animKey} x1={pts[prev][0]} y1={pts[prev][1]} stroke="var(--accent)" strokeWidth={2.5}
          initial={{ x2: pts[prev][0], y2: pts[prev][1] }} animate={{ x2: pts[t][0], y2: pts[t][1] }} transition={{ delay: T.ball * k, duration: T.ballDur * k, ease }} />
      )}
      <motion.circle r={4} fill="var(--accent)" stroke="var(--surface)" strokeWidth={1.5} initial={kind === "step" ? { cx: pts[prev][0], cy: pts[prev][1] } : false}
        animate={{ cx: pts[t][0], cy: pts[t][1] }} transition={{ delay: (kind === "step" ? T.ball : 0) * k, duration: T.ballDur * k, ease }} />
    </svg>
  );
}

/** Counts up for readable magnitudes; huge (diverged) values are shown in scientific notation instead. */
function Num({ value, from, delay, duration }: { value: number; from: number; delay: number; duration: number }) {
  return Math.max(Math.abs(value), Math.abs(from)) >= 1e4 || !Number.isFinite(value) ? <span>{f3(value)}</span> : <AnimatedNumber value={value} from={from} digits={4} delay={delay} duration={duration} />;
}

function Row({ label, children, at, k, tone }: { label: string; children: ReactNode; at: number; k: number; tone?: string }) {
  return (
    <motion.div className={cx("grid grid-cols-[150px_minmax(0,1fr)] items-baseline gap-2 border-t border-line py-1 text-sm first:border-t-0", tone)}
      initial={{ opacity: 0, x: -12 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: at * k, duration: 0.35 * k, ease }}>
      <span className="text-xs text-muted">{label}</span>
      <span className="min-w-0 break-words font-mono">{children}</span>
    </motion.div>
  );
}

function Card({ sim, state, a, b, k }: { sim: ReturnType<typeof simulate>; state: GDState; a: Iter; b: Iter; k: number }) {
  const { cfg } = sim;
  const v = (p: Vec) => `(${f3(p[0])}, ${f3(p[1])})`;
  if (state.kind === "start") {
    return (
      <div className="space-y-1 text-sm">
        <Row label="start position" at={0.2} k={k}>{v(b.p)}</Row>
        <Row label="loss there" at={0.5} k={k}>{f3(b.f)}</Row>
        <Row label="recipe (every step)" at={0.8} k={k}>p ← p − η·∇f(p){cfg.momentum ? " (with velocity v)" : ""}</Row>
        <p className="pt-1 text-xs text-muted">The surface is {sim.surface.id === "bowl" ? "10× steeper in y than in x, so one learning rate has to suit both a steep and a shallow direction." : "flat at the saddle in the middle: a ball near it barely moves until it rolls off toward a valley."}</p>
      </div>
    );
  }
  if (state.kind === "summary") {
    const f0 = sim.iters[0].f, fl = sim.iters.at(-1)!.f;
    return (
      <div className="space-y-2 text-sm">
        {[["start loss", f0], ["final loss", fl]].map(([n, val], i) => (
          <div key={i} className="grid grid-cols-[90px_minmax(0,1fr)_80px] items-center gap-2 text-xs">
            <span>{n as string}</span>
            <div className="h-4 rounded bg-surface-2"><motion.div className={cx("h-full rounded", i ? (fl > f0 ? "bg-err/70" : "bg-ok/70") : "bg-muted/50")} initial={{ width: 0 }}
              animate={{ width: `${Math.max(1, (100 * Math.log1p(val as number)) / Math.log1p(Math.max(f0, fl)))}%` }} transition={{ delay: (0.3 + i * 0.6) * k, duration: 0.7 * k, ease }} /></div>
            <span className="text-right font-mono">{f3(val as number)}</span>
          </div>
        ))}
        <p className="text-xs text-muted">Bars use a log scale. The trail replays in blue on the surface. Try the learning-rate presets above: same surface, same start, very different journeys.</p>
      </div>
    );
  }
  const up = b.f > a.f;
  return (
    <div className="space-y-0.5">
      <Row label="1. standing at" at={0} k={k}>{v(a.p)} · loss {f3(a.f)}</Row>
      <Row label="2. gradient ∇f (uphill)" at={T.grad} k={k} tone="text-err">{v(a.g)}{cfg.stochastic && <span className="text-muted"> = true {v(a.gTrue)} + noise</span>}</Row>
      {cfg.momentum && <Row label={`velocity v = ${cfg.beta}·v + ∇f`} at={T.upd - 0.2} k={k}>{v(a.v)}</Row>}
      <Row label={`3. move = −${cfg.lr}·${cfg.momentum ? "v" : "∇f"}`} at={T.upd} k={k} tone="text-ok">{v(a.move)} <span className="text-muted">· length {f3(Math.hypot(...a.move))}</span></Row>
      <Row label="4. new position" at={T.ball} k={k}>{v(b.p)}</Row>
      <Row label="5. new loss" at={T.ball} k={k} tone={up ? "text-err" : undefined}>
        <Num value={b.f} from={a.f} delay={T.ball * k} duration={T.ballDur * k} /> {up ? "↑ worse: overshot" : b.f < a.f ? `↓ ${(100 * (1 - b.f / Math.max(a.f, 1e-12))).toFixed(1)}% lower` : "→ unchanged"}
      </Row>
    </div>
  );
}

function Calc({ sim, state, a, b }: { sim: ReturnType<typeof simulate>; state: GDState; a: Iter; b: Iter }) {
  const s = sim.surface, { cfg } = sim;
  if (state.kind !== "step") return <div className="space-y-2"><TexBlock>{s.tex}</TexBlock><TexBlock>{s.gradTex}</TexBlock><TexBlock>{cfg.momentum ? "v_{t+1} = \\beta v_t + \\nabla f(p_t), \\quad p_{t+1} = p_t - \\eta\\, v_{t+1}" : "p_{t+1} = p_t - \\eta\\, \\nabla f(p_t)"}</TexBlock></div>;
  const g = cfg.momentum ? "v" : "\\nabla f";
  return (
    <TexBlock>{`\\begin{aligned} \\nabla f(p) &= (${f3(a.g[0])},\\ ${f3(a.g[1])}) ${cfg.stochastic ? "\\ \\text{(noisy)}" : ""} \\\\ ${cfg.momentum ? `v &= ${cfg.beta}\\, v_{\\text{prev}} + \\nabla f = (${f3(a.v[0])},\\ ${f3(a.v[1])}) \\\\` : ""} \\Delta &= -${cfg.lr}\\, ${g} = (${f3(a.move[0])},\\ ${f3(a.move[1])}) \\\\ p_{\\text{new}} &= p + \\Delta = (${f3(b.p[0])},\\ ${f3(b.p[1])}) \\\\ f(p_{\\text{new}}) &= ${f3(b.f)} \\quad (\\text{was } ${p3(a.f)}) \\end{aligned}`}</TexBlock>
  );
}
