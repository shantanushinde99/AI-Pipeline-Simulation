import { useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { RotateCcw } from "lucide-react";
import { usePlayback } from "../../hooks/usePlayback";
import { PlaybackControls } from "../../components/visualizer/PlaybackControls";
import { EventLog } from "../../components/visualizer/EventLog";
import { StepExplainer, Workspace } from "../../components/visualizer/Workspace";
import { AnimatedNumber } from "../../components/visualizer/AnimatedNumber";
import { Badge, Button, Callout, KV, Panel, Select, SimulatedLabel, Slider, Toggle, cx } from "../../components/ui";
import { TexBlock } from "../../components/Tex";
import type { VisualizerTab } from "../types";
import { MetricBar, StepCaption, ease, useTicks } from "../classification-metrics/kit";
import { DATASETS, DEFAULT_REG, fmtNum, simulate, unitTex, type DatasetId, type RegConfig, type RegPhase, type RegSim } from "./simulation";

/* ---------- one timing schedule (seconds at 1×) ---------- */
const FLY = 0.6;
const contribAt = (i: number, n: number) => 0.4 + i * Math.min(0.35, 3 / n);
const divAt = (n: number) => contribAt(n - 1, n) + FLY + 0.3;
const R2 = { mean: 0.2, dev: (i: number) => 0.8 + i * 0.08, tot: (i: number) => 1.8 + i * 0.1, res: (i: number, n: number) => 1.9 + n * 0.1 + i * 0.1, ratio: (n: number) => 2.2 + 2 * n * 0.1 };
const DURATIONS: Record<RegPhase, number> = { data: 3200, residuals: 3200, mae: 6000, mse: 6000, rmse: 3800, r2: 6000, mape: 6000, compare: 3800 };

const PRESETS: { label: string; cfg: Partial<RegConfig> }[] = [
  { label: "Clean house prices", cfg: { dataset: "houses", outlier: false, altUnits: false, noise: 1 } },
  { label: "Break it: one wild prediction", cfg: { outlier: true } },
  { label: "Targets near zero (MAPE explodes)", cfg: { dataset: "energy", outlier: false } },
  { label: "Change units ×1000", cfg: {} },
];

type Acc = "mae" | "mse" | "mape";
const accOf = (s: RegSim, p: Acc) => (p === "mae" ? s.m.abs : p === "mse" ? s.m.sq : s.m.ape);

const NARRATE: Record<RegPhase, (s: RegSim) => { headline: string; sub: string }> = {
  data: (s) => ({ headline: `${s.m.n} predictions next to the truth`, sub: `Filled dot = actual ${s.ds.what}, hollow dot = model prediction, in ${s.unit}.` }),
  residuals: (s) => ({ headline: `Each bar is one miss; the biggest is #${s.worst + 1} (${fmtNum(s.m.res[s.worst])} ${s.unit})`, sub: "Residual e = actual − predicted. Every regression metric is built from these bars." }),
  mae: (s) => ({ headline: `Drop every |miss| into one total, divide by ${s.m.n}: MAE = ${fmtNum(s.m.mae)} ${s.unit}`, sub: "Each bar keeps its length, so every unit of error counts the same." }),
  mse: (s) => ({ headline: `Square each miss first: MSE = ${fmtNum(s.m.mse)} ${s.unit}²`, sub: "Watch the biggest bar take over the total. Squared units are hard to read." }),
  rmse: (s) => ({ headline: `√MSE brings it back to ${s.unit}: RMSE = ${fmtNum(s.m.rmse)}`, sub: `The shaded band is ±RMSE around each prediction; the dashed one is ±MAE (${fmtNum(s.m.mae)}).` }),
  r2: (s) => ({ headline: `R² = ${Number.isFinite(s.m.r2) ? s.m.r2.toFixed(3) : "undefined"}: how much better than guessing the mean`, sub: "Purple: error of always predicting the mean. Orange: the model's error. R² = 1 − orange / purple." }),
  mape: (s) => ({ headline: `Divide each miss by its own truth: MAPE = ${fmtNum(s.m.mape)}%`, sub: s.m.ape[s.worstApe] > 100 ? `#${s.worstApe + 1} has a tiny actual value, so a small miss becomes ${fmtNum(s.m.ape[s.worstApe])}%.` : "Percentages are unit-free, but explode when the truth is near zero." }),
  compare: (s) => ({ headline: s.cfg.outlier ? "One wild prediction: MSE/RMSE jump, MAE moves less, R² collapses" : "Flip the outlier or the units and see which metrics react", sub: "MAE/RMSE scale with units, MSE with units², R² and MAPE are unit-free." }),
};
const WHY: Record<RegPhase, string> = {
  data: "A single number will summarize all these gaps; which number you pick decides what 'good' means.",
  residuals: "Plot residuals before trusting any summary: one bad point can dominate.",
  mae: "MAE is robust and easy to explain: 'on average we miss by X'.",
  mse: "MSE is what most models minimize; it punishes large errors heavily.",
  rmse: "RMSE is in the target's units but still outlier-sensitive.",
  r2: "R² is scale-free and compares against a trivial baseline; it can be negative.",
  mape: "MAPE is undefined at y = 0 and unstable near it; prefer MAE or sMAPE/WAPE there.",
  compare: "Report a robust metric (MAE) beside a sensitive one (RMSE) to spot outliers.",
};

export default function RegressionMetricsVisualizer({ tab }: { tab: VisualizerTab }) {
  const [cfg, setCfg] = useState(DEFAULT_REG);
  const [replay, setReplay] = useState(0);
  const set = (p: Partial<RegConfig>) => setCfg((c) => ({ ...c, ...p }));
  const sim = simulate(cfg);
  const pb = usePlayback(sim.steps.length, (i) => DURATIONS[sim.steps[i]?.state.phase ?? "data"]);
  const step = sim.steps[pb.index];
  const phase = step.state.phase;
  const k = 1 / pb.speed;
  const animKey = `${step.id}-${replay}`;
  if (tab === "metrics") return null;
  const story = NARRATE[phase](sim);

  return (
    <Workspace
      main={
        <>
          <Panel title="Predictions, residuals and running totals" actions={<SimulatedLabel>Toy data</SimulatedLabel>}>
            <StepCaption index={pb.index} total={sim.steps.length} stepId={step.id} headline={story.headline} sub={story.sub} onReplay={() => setReplay((r) => r + 1)} />
            <div className="overflow-x-auto">
              <div className="min-w-[560px]">
                <Chart sim={sim} phase={phase} k={k} animKey={animKey} />
              </div>
            </div>
            <div className="mt-1 flex flex-wrap gap-3 text-[11px] text-muted">
              <span className="flex items-center gap-1"><svg width="10" height="10"><circle cx="5" cy="5" r="4" fill="var(--accent)" /></svg>actual y</span>
              <span className="flex items-center gap-1"><svg width="10" height="10"><circle cx="5" cy="5" r="3.5" fill="var(--surface)" stroke="var(--violet)" strokeWidth="1.5" /></svg>prediction ŷ</span>
              <span className="flex items-center gap-1"><svg width="10" height="10"><rect x="3" y="0" width="4" height="10" fill="var(--warn)" /></svg>residual (miss)</span>
              <span>units: {sim.unit}</span>
            </div>
          </Panel>
          <Panel title="What's happening in this step">
            <AnimatePresence mode="wait" initial={false}>
              <motion.div key={animKey} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: 0.2 }}>
                <StepCard sim={sim} phase={phase} k={k} animKey={animKey} />
              </motion.div>
            </AnimatePresence>
          </Panel>
          <PlaybackControls pb={pb} stepTitles={sim.steps.map((s) => s.title)} />
          <Panel title="Parameters" actions={<Button size="sm" variant="ghost" onClick={() => setCfg(DEFAULT_REG)}><RotateCcw size={12} /> Reset</Button>}>
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              <Select label="Dataset" value={cfg.dataset} options={(Object.keys(DATASETS) as DatasetId[]).map((d) => ({ value: d, label: DATASETS[d].name }))} onChange={(v) => set({ dataset: v })} />
              <Slider label="Typical model error" value={cfg.noise} min={0.2} max={3} step={0.1} format={(v) => `×${v.toFixed(1)}`} onChange={(v) => set({ noise: v })} />
              <Slider label="Examples" value={cfg.n} min={4} max={14} onChange={(v) => set({ n: v })} />
              <Slider label="Random seed" value={cfg.seed} min={1} max={40} onChange={(v) => set({ seed: v })} />
              <div className="space-y-2 pt-1 sm:col-span-2">
                <Toggle label="Break it: one wild prediction (outlier)" checked={cfg.outlier} onChange={(v) => set({ outlier: v })} hint={`Example #${Math.min(7, cfg.n)} is predicted far too high.`} />
                <Toggle label={`Measure in ${sim.ds.altUnit} instead of ${sim.ds.unit} (×${sim.ds.factor})`} checked={cfg.altUnits} onChange={(v) => set({ altUnits: v })} hint="Same data, different units." />
              </div>
            </div>
            <div className="mt-3 flex flex-wrap gap-1.5">
              {PRESETS.map((p) => <Button key={p.label} size="sm" onClick={() => set(p.label.startsWith("Change") ? { altUnits: !cfg.altUnits } : p.cfg)}>{p.label}</Button>)}
            </div>
          </Panel>
        </>
      }
      inspector={
        <>
          <StepExplainer index={pb.index} title={step.title} what={step.description} why={WHY[phase]} />
          <Panel title="Calculation"><Calculation sim={sim} phase={phase} /></Panel>
          <Panel title="All metrics">
            <KV rows={[
              ["MAE", `${fmtNum(sim.m.mae)} ${sim.unit}`],
              ["MSE", `${fmtNum(sim.m.mse)} ${sim.unit}²`],
              ["RMSE", `${fmtNum(sim.m.rmse)} ${sim.unit}`],
              ["R²", Number.isFinite(sim.m.r2) ? sim.m.r2.toFixed(3) : "undefined"],
              ["MAPE", `${fmtNum(sim.m.mape)}%`],
            ]} />
          </Panel>
          <Panel title="Events"><EventLog events={step.events} /></Panel>
        </>
      }
    />
  );
}

/* ---------- canvas ---------- */
const W = 680, X0 = 64, X1 = 664, T = 16, B = 226, STRIP = 268, SH = 18;

function Chart({ sim, phase, k, animKey }: { sim: RegSim; phase: RegPhase; k: number; animKey: string }) {
  const { pts, m } = sim;
  const n = pts.length;
  const sw = (X1 - X0) / n;
  const xi = (i: number) => X0 + (i + 0.5) * sw;
  const vals = pts.flatMap((p) => [p.y, p.yhat]);
  const lo = Math.min(0, ...vals), hi = Math.max(...vals) * 1.08 || 1;
  const sy = (v: number) => B - ((v - lo) / (hi - lo)) * (B - T);
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => lo + f * (hi - lo));
  const acc = phase === "mae" || phase === "mse" || phase === "mape" ? phase : null;
  const isR2 = phase === "r2";
  const H = isR2 ? 346 : 312;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label={`${n} actual values and predictions with residual bars`}>
      {ticks.map((t) => (
        <g key={t}>
          <line x1={X0} x2={X1} y1={sy(t)} y2={sy(t)} stroke="var(--line)" />
          <text x={X0 - 6} y={sy(t) + 3} textAnchor="end" fontSize={10} fill="var(--muted)" fontFamily="var(--font-mono)">{fmtNum(t)}</text>
        </g>
      ))}
      <text x={12} y={(T + B) / 2} fontSize={10} fill="var(--muted)" textAnchor="middle" transform={`rotate(-90 12 ${(T + B) / 2})`}>{sim.ds.what} ({sim.unit})</text>

      {/* ±RMSE / ±MAE bands */}
      {phase === "rmse" && pts.map((p, i) => (
        <g key={`band-${i}-${animKey}`}>
          <motion.rect x={xi(i) - sw * 0.35} width={sw * 0.7} fill="var(--accent)" fillOpacity={0.12} initial={{ y: sy(p.yhat), height: 0 }}
            animate={{ y: sy(p.yhat + m.rmse), height: Math.max(0, sy(p.yhat - m.rmse) - sy(p.yhat + m.rmse)) }} transition={{ delay: (0.6 + i * 0.05) * k, duration: 0.8 * k, ease }} />
          <motion.rect x={xi(i) - sw * 0.3} width={sw * 0.6} fill="none" stroke="var(--fg)" strokeOpacity={0.5} strokeDasharray="3 3" initial={{ y: sy(p.yhat), height: 0, opacity: 0 }}
            animate={{ y: sy(p.yhat + m.mae), height: Math.max(0, sy(p.yhat - m.mae) - sy(p.yhat + m.mae)), opacity: 1 }} transition={{ delay: (1.6 + i * 0.05) * k, duration: 0.6 * k, ease }} />
        </g>
      ))}

      {/* mean line and deviations for R² */}
      {isR2 && (
        <g key={`r2-${animKey}`}>
          <motion.line x1={X0} x2={X1} y1={sy(m.ybar)} y2={sy(m.ybar)} stroke="var(--violet)" strokeWidth={2} strokeDasharray="6 4" initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ delay: R2.mean * k, duration: 0.6 * k }} />
          <motion.text x={X1} y={sy(m.ybar) - 5} textAnchor="end" fontSize={10} fill="var(--violet)" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.6 * k }}>mean ȳ = {fmtNum(m.ybar)}</motion.text>
          {pts.map((p, i) => (
            <motion.line key={i} x1={xi(i) - 6} x2={xi(i) - 6} y1={sy(m.ybar)} stroke="var(--violet)" strokeWidth={3} strokeOpacity={0.55} initial={{ y2: sy(m.ybar) }} animate={{ y2: sy(p.y) }} transition={{ delay: R2.dev(i) * k, duration: 0.4 * k }} />
          ))}
        </g>
      )}

      {/* residual bars */}
      {phase !== "data" && pts.map((p, i) => (
        <motion.line key={`res-${i}-${phase === "residuals" ? animKey : "s"}`} x1={xi(i)} x2={xi(i)} y1={sy(p.y)} stroke="var(--warn)" strokeWidth={6} strokeOpacity={0.8}
          initial={phase === "residuals" ? { y2: sy(p.y) } : false} animate={{ y2: sy(p.yhat) }} transition={{ delay: (0.3 + i * 0.12) * k, duration: 0.5 * k, ease }} />
      ))}

      {/* dots */}
      {pts.map((p, i) => {
        const pop = phase === "data";
        return (
          <g key={`pt-${i}-${pop ? animKey : "s"}`}>
            <motion.circle cx={xi(i)} cy={sy(p.yhat)} r={5.5} fill="var(--surface)" stroke="var(--violet)" strokeWidth={2}
              initial={pop ? { scale: 0, opacity: 0 } : false} animate={{ scale: 1, opacity: 1, cy: sy(p.yhat) }} transition={{ delay: pop ? (0.9 + i * 0.12) * k : 0, duration: 0.4 * k }} />
            <motion.circle cx={xi(i)} r={5.5} fill="var(--accent)"
              initial={pop ? { scale: 0, opacity: 0, cy: sy(p.y) } : false} animate={{ scale: 1, opacity: 1, cy: sy(p.y) }} transition={{ delay: pop ? (0.2 + i * 0.12) * k : 0, duration: 0.4 * k }} />
            <text x={xi(i)} y={B + 14} textAnchor="middle" fontSize={10} fill={i === sim.worst && phase !== "data" ? "var(--warn)" : "var(--muted)"} fontFamily="var(--font-mono)">#{i + 1}</text>
          </g>
        );
      })}

      {acc && <Accumulator key={animKey} sim={sim} phase={acc} k={k} xi={xi} sy={sy} animKey={animKey} />}
      {isR2 && <R2Strips key={animKey} sim={sim} k={k} />}
      {!acc && !isR2 && (
        <text x={X0} y={STRIP + 12} fontSize={11} fill="var(--muted)">
          {phase === "data" ? "Running totals for each metric will build up here." : phase === "residuals" ? "Next: these bars get added up, three different ways." : phase === "rmse" ? `RMSE ${fmtNum(m.rmse)} ≥ MAE ${fmtNum(m.mae)}: the gap reveals unevenly sized errors.` : `MAE ${fmtNum(m.mae)} · RMSE ${fmtNum(m.rmse)} · R² ${m.r2.toFixed(3)} · MAPE ${fmtNum(m.mape)}%`}
        </text>
      )}
    </svg>
  );
}

function Accumulator({ sim, phase, k, xi, sy, animKey }: { sim: RegSim; phase: Acc; k: number; xi: (i: number) => number; sy: (v: number) => number; animKey: string }) {
  const c = accOf(sim, phase);
  const n = c.length;
  const finite = c.map((v) => (Number.isFinite(v) ? v : 0));
  const total = finite.reduce((s, v) => s + v, 0);
  const scale = (X1 - X0) / (total || 1);
  const starts = finite.map((_, i) => X0 + finite.slice(0, i).reduce((s, v) => s + v, 0) * scale);
  const landed = useTicks(c.map((_, i) => (contribAt(i, n) + FLY) * k), animKey);
  const partial = finite.slice(0, landed).reduce((s, v) => s + v, 0);
  const sym = phase === "mae" ? "Σ|e|" : phase === "mse" ? "Σe²" : "Σ|e|/|y|·100";
  const unit = phase === "mae" ? sim.unit : phase === "mse" ? `${sim.unit}²` : "%";
  const mean = total / n;
  const name = phase.toUpperCase();
  const fill = phase === "mae" ? "var(--warn)" : phase === "mse" ? "var(--err)" : "var(--violet)";
  return (
    <g>
      <rect x={X0} y={STRIP} width={X1 - X0} height={SH} rx={3} fill="var(--surface-2)" stroke="var(--line)" />
      <text x={X0} y={STRIP - 6} fontSize={11} fontFamily="var(--font-mono)" fill="var(--fg)">{sym} = {fmtNum(partial)} {unit} <tspan fill="var(--muted)">({landed}/{n} added)</tspan></text>
      {sim.pts.map((p, i) => {
        const w = Math.max(1, finite[i] * scale - 1);
        const top = Math.min(sy(p.y), sy(p.yhat));
        const h = Math.max(2, Math.abs(sy(p.y) - sy(p.yhat)));
        return (
          <motion.g key={i} initial={{ x: xi(i) - 3, y: top, opacity: 0 }} animate={{ x: [xi(i) - 3, xi(i) - 3, starts[i]], y: [top, top, STRIP], opacity: [0, 1, 1] }}
            transition={{ delay: contribAt(i, n) * k, duration: FLY * k, times: [0, 0.15, 1], ease }}>
            <motion.rect initial={{ width: 6, height: h }} animate={{ width: [6, 6, w], height: [h, h, SH] }} transition={{ delay: contribAt(i, n) * k, duration: FLY * k, times: [0, 0.15, 1], ease }}
              fill={fill} fillOpacity={0.85} stroke="var(--surface)" strokeWidth={0.5} />
            {w > 34 && <motion.text x={w / 2} y={13} textAnchor="middle" fontSize={9.5} fill="var(--surface)" fontFamily="var(--font-mono)" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: (contribAt(i, n) + FLY) * k }}>#{i + 1}</motion.text>}
          </motion.g>
        );
      })}
      {/* ÷ n: the average is the length that, laid n times, fills the total */}
      <motion.g initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: divAt(n) * k, duration: 0.4 * k }}>
        {Array.from({ length: n }, (_, j) => <line key={j} x1={X0 + ((j + 1) * (X1 - X0)) / n} x2={X0 + ((j + 1) * (X1 - X0)) / n} y1={STRIP + SH} y2={STRIP + SH + 6} stroke="var(--muted)" />)}
        <motion.rect x={X0} y={STRIP + SH + 8} height={8} rx={2} fill="var(--fg)" initial={{ width: 0 }} animate={{ width: (X1 - X0) / n }} transition={{ delay: (divAt(n) + 0.2) * k, duration: 0.5 * k }} />
        <text x={X0 + (X1 - X0) / n + 8} y={STRIP + SH + 16} fontSize={11} fontWeight={600} fontFamily="var(--font-mono)" fill="var(--fg)">÷ {n} = {name} {fmtNum(mean)} {unit}</text>
      </motion.g>
    </g>
  );
}

function R2Strips({ sim, k }: { sim: RegSim; k: number }) {
  const { m } = sim;
  const n = m.n;
  const scale = (X1 - X0) / Math.max(m.ssTot, m.ssRes, 1e-12);
  const row = (vals: number[], y: number, color: string, at: (i: number) => number, label: string, total: number) => {
    const starts = vals.map((_, i) => X0 + vals.slice(0, i).reduce((a, b) => a + b, 0) * scale);
    return (
      <g>
        <text x={X0} y={y - 5} fontSize={11} fontFamily="var(--font-mono)" fill={color}>{label} = {fmtNum(total)}</text>
        <rect x={X0} y={y} width={X1 - X0} height={SH - 4} rx={3} fill="var(--surface-2)" />
        {vals.map((v, i) => {
          const w = v * scale;
          return <motion.rect key={i} x={starts[i]} y={y} height={SH - 4} fill={color} fillOpacity={0.8} stroke="var(--surface)" strokeWidth={0.5} initial={{ width: 0 }} animate={{ width: w }} transition={{ delay: at(i) * k, duration: 0.25 * k }} />;
        })}
      </g>
    );
  };
  return (
    <g>
      {row(m.tot, STRIP - 8, "var(--violet)", R2.tot, "SS_tot (predict the mean)", m.ssTot)}
      {row(m.sq, STRIP + 32, "var(--warn)", (i) => R2.res(i, n), "SS_res (the model)", m.ssRes)}
      <motion.text x={X1} y={STRIP + 64} textAnchor="end" fontSize={12} fontWeight={600} fontFamily="var(--font-mono)" fill="var(--fg)" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: R2.ratio(n) * k }}>
        R² = 1 − {fmtNum(m.ssRes)} / {fmtNum(m.ssTot)} = {Number.isFinite(m.r2) ? m.r2.toFixed(3) : "undefined"}
      </motion.text>
    </g>
  );
}

/* ---------- step card ---------- */
function StepCard({ sim, phase, k, animKey }: { sim: RegSim; phase: RegPhase; k: number; animKey: string }) {
  const { pts, m } = sim;
  if (phase === "data" || phase === "residuals") {
    const maxAbs = Math.max(...m.abs, 1e-12);
    return (
      <div className="max-h-72 overflow-y-auto">
        <table className="w-full text-xs">
          <thead className="text-left text-muted"><tr><th className="py-1">#</th><th>actual y</th><th>predicted ŷ</th><th className="w-1/2">{phase === "residuals" ? "residual e = y − ŷ" : ""}</th></tr></thead>
          <tbody className="font-mono">
            {pts.map((p, i) => (
              <motion.tr key={i} className={cx("border-t border-line", i === sim.worst && phase === "residuals" && "bg-warn-soft")} initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: (0.2 + i * 0.12) * k, duration: 0.3 * k }}>
                <td className="py-0.5">{i + 1}</td><td>{fmtNum(p.y)}</td><td>{fmtNum(p.yhat)}</td>
                <td>
                  {phase === "residuals" && (
                    <div className="grid grid-cols-[minmax(0,1fr)_64px] items-center gap-2">
                      <div className="relative h-3 rounded bg-surface-2">
                        <div className="absolute inset-y-0 left-1/2 w-px bg-line" />
                        <motion.div className="absolute inset-y-0 rounded bg-warn" style={m.res[i] < 0 ? { right: "50%" } : { left: "50%" }} initial={{ width: 0 }}
                          animate={{ width: `${(m.abs[i] / maxAbs) * 50}%` }} transition={{ delay: (0.3 + i * 0.12) * k, duration: 0.5 * k, ease }} />
                      </div>
                      <span className="text-right">{m.res[i] > 0 ? "+" : ""}{fmtNum(m.res[i])}</span>
                    </div>
                  )}
                </td>
              </motion.tr>
            ))}
          </tbody>
        </table>
        {phase === "residuals" && <p className="mt-1 text-[11px] text-muted">Left of centre: the model predicted too high. Right: too low.</p>}
      </div>
    );
  }
  if (phase === "mae" || phase === "mse" || phase === "mape") return <AccCard sim={sim} phase={phase} k={k} animKey={animKey} />;
  if (phase === "rmse") {
    const top = Math.max(m.rmse, m.mae, 1e-12);
    return (
      <div className="space-y-2">
        <div className="flex flex-wrap items-center gap-2 font-mono text-sm">
          <Badge>MSE = {fmtNum(m.mse)} {sim.unit}²</Badge><span>→ √ →</span>
          <Badge tone="accent">RMSE = <AnimatedNumber value={m.rmse} from={m.mse} digits={m.rmse >= 100 ? 1 : 2} delay={0.3 * k} duration={0.8 * k} /> {sim.unit}</Badge>
        </div>
        <MetricBar label="RMSE" value={m.rmse / top} k={k} delay={0.6} note={`${fmtNum(m.rmse)} ${sim.unit}`} />
        <MetricBar label="MAE" value={m.mae / top} k={k} delay={1.6} tone="warn" note={`${fmtNum(m.mae)} ${sim.unit}`} />
        <p className="text-xs text-muted">RMSE / MAE = {fmtNum(m.rmse / m.mae)}. Equal sized errors give 1.0; a ratio well above 1.2 means a few misses are much larger than the rest.</p>
      </div>
    );
  }
  if (phase === "r2") {
    const top = Math.max(m.ssTot, m.ssRes, 1e-12);
    return (
      <div className="space-y-2">
        <MetricBar label="SS_tot (mean)" value={m.ssTot / top} k={k} delay={R2.tot(0)} tone="violet" note={fmtNum(m.ssTot)} />
        <MetricBar label="SS_res (model)" value={m.ssRes / top} k={k} delay={R2.res(0, m.n)} tone="warn" note={fmtNum(m.ssRes)} />
        <p className="font-mono text-sm">R² = <span className="text-lg font-semibold">{Number.isFinite(m.r2) ? <AnimatedNumber value={m.r2} from={0} digits={3} delay={R2.ratio(m.n) * k} duration={0.6 * k} /> : "undefined"}</span></p>
        {m.r2 < 0 && <Callout tone="warn">R² is negative: the model is worse than always predicting the mean.</Callout>}
      </div>
    );
  }
  const rows: { name: string; now: number; alt: number; scaled: number; unit: string }[] = [
    { name: "MAE", now: m.mae, alt: sim.other.mae, scaled: sim.scaled.mae, unit: sim.unit },
    { name: "MSE", now: m.mse, alt: sim.other.mse, scaled: sim.scaled.mse, unit: `${sim.unit}²` },
    { name: "RMSE", now: m.rmse, alt: sim.other.rmse, scaled: sim.scaled.rmse, unit: sim.unit },
    { name: "R²", now: m.r2, alt: sim.other.r2, scaled: sim.scaled.r2, unit: "" },
    { name: "MAPE", now: m.mape, alt: sim.other.mape, scaled: sim.scaled.mape, unit: "%" },
  ];
  const otherUnit = sim.cfg.altUnits ? sim.ds.unit : sim.ds.altUnit;
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead className="text-left text-muted"><tr><th className="py-1">metric</th><th>now</th><th>{sim.cfg.outlier ? "without outlier" : "with outlier"}</th><th>in {otherUnit}</th></tr></thead>
        <tbody className="font-mono">
          {rows.map((r, i) => {
            const ratio = r.name === "R²" ? Math.abs(r.alt - r.now) : r.alt / r.now;
            const jumps = r.name === "R²" ? ratio > 0.2 : ratio > 2 || ratio < 0.5;
            const scales = Math.abs(r.scaled / r.now - 1) > 1e-6;
            return (
              <motion.tr key={r.name} className="border-t border-line" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: (0.2 + i * 0.3) * k }}>
                <td className="py-1 font-sans font-medium">{r.name}</td>
                <td>{r.name === "R²" ? r.now.toFixed(3) : fmtNum(r.now)} {r.unit}</td>
                <td className={jumps ? "text-warn" : undefined}>{r.name === "R²" ? r.alt.toFixed(3) : fmtNum(r.alt)} {jumps && <Badge tone="warn">{r.name === "R²" ? "shifts" : `×${fmtNum(ratio)}`}</Badge>}</td>
                <td>{r.name === "R²" ? r.scaled.toFixed(3) : fmtNum(r.scaled)} {scales ? <Badge tone="accent">×{fmtNum(r.scaled / r.now)}</Badge> : <Badge tone="ok">same</Badge>}</td>
              </motion.tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function AccCard({ sim, phase, k, animKey }: { sim: RegSim; phase: Acc; k: number; animKey: string }) {
  const c = accOf(sim, phase);
  const n = c.length;
  const landed = useTicks(c.map((_, i) => (contribAt(i, n) + FLY) * k), `${animKey}-card`);
  const max = Math.max(...c.filter(Number.isFinite), 1e-12);
  const cum = c.map((_, i) => c.slice(0, i + 1).reduce((a, b) => a + b, 0));
  const run = cum[n - 1];
  const unit = phase === "mae" ? sim.unit : phase === "mse" ? `${sim.unit}²` : "%";
  const head = phase === "mae" ? "|e|" : phase === "mse" ? "e²" : "|e| / |y| × 100";
  const big = c.indexOf(max);
  return (
    <div className="space-y-2">
      <div className="max-h-64 overflow-y-auto">
        <table className="w-full text-xs">
          <thead className="text-left text-muted"><tr><th className="py-0.5">#</th><th>{head}</th><th className="w-2/5" /><th className="text-right">running total</th></tr></thead>
          <tbody className="font-mono">
            {c.map((v, i) => {
              const on = i < landed;
              return (
                <tr key={i} className={cx("border-t border-line transition-opacity", on ? "opacity-100" : "opacity-35", i === big && on && "bg-warn-soft")}>
                  <td className="py-0.5">{i + 1}</td>
                  <td>{fmtNum(v)}{phase === "mape" && `  (${fmtNum(sim.m.abs[i])}/${fmtNum(sim.pts[i].y)})`}</td>
                  <td><div className="h-2.5 rounded bg-surface-2"><motion.div className="h-full rounded bg-warn" initial={{ width: 0 }} animate={{ width: `${(v / max) * 100}%` }} transition={{ delay: (contribAt(i, n) + FLY) * k, duration: 0.3 * k }} /></div></td>
                  <td className="text-right">{on ? fmtNum(cum[i]) : "…"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <motion.p className="font-mono text-sm" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: divAt(n) * k }}>
        {phase.toUpperCase()} = {fmtNum(run)} / {n} = <strong className="text-base">{fmtNum(run / n)} {unit}</strong>
      </motion.p>
      {phase === "mse" && <p className="text-xs text-muted">#{big + 1} contributes {((max / (run || 1)) * 100).toFixed(0)}% of the total on its own.</p>}
      {phase === "mape" && max > 100 && <Callout tone="warn" title="Targets near zero">#{big + 1} is only {fmtNum(sim.pts[big].y)} {sim.unit}, so a miss of {fmtNum(sim.m.abs[big])} is {fmtNum(max)}% of it. MAPE is dominated by the smallest targets, not the biggest errors.</Callout>}
    </div>
  );
}

function Calculation({ sim, phase }: { sim: RegSim; phase: RegPhase }) {
  const { m, pts } = sim;
  const u = unitTex(sim.unit);
  const show = pts.slice(0, 3);
  const more = pts.length > 3 ? " + \\cdots" : "";
  const p = (v: number) => (v < 0 ? `(${fmtNum(v)})` : fmtNum(v));
  const tex: Record<RegPhase, string> = {
    data: `\\{(y_i, \\hat{y}_i)\\}_{i=1}^{${m.n}}, \\quad \\bar{y} = ${fmtNum(m.ybar)}\\ ${u}`,
    residuals: `e_i = y_i - \\hat{y}_i: \\; ${show.map((q, i) => `e_{${i + 1}} = ${fmtNum(q.y)} - ${fmtNum(q.yhat)} = ${fmtNum(m.res[i])}`).join(",\\; ")}`,
    mae: `\\text{MAE} = \\frac{1}{n}\\sum_i |e_i| = \\frac{${show.map((_, i) => fmtNum(m.abs[i])).join(" + ")}${more}}{${m.n}} = ${fmtNum(m.mae)}\\ ${u}`,
    mse: `\\text{MSE} = \\frac{1}{n}\\sum_i e_i^2 = \\frac{${show.map((_, i) => `${p(m.res[i])}^2`).join(" + ")}${more}}{${m.n}} = ${fmtNum(m.mse)}\\ ${u}^2`,
    rmse: `\\text{RMSE} = \\sqrt{\\text{MSE}} = \\sqrt{${fmtNum(m.mse)}} = ${fmtNum(m.rmse)}\\ ${u} \\;\\ge\\; \\text{MAE} = ${fmtNum(m.mae)}`,
    r2: `R^2 = 1 - \\frac{\\sum (y_i - \\hat{y}_i)^2}{\\sum (y_i - \\bar{y})^2} = 1 - \\frac{${fmtNum(m.ssRes)}}{${fmtNum(m.ssTot)}} = ${Number.isFinite(m.r2) ? m.r2.toFixed(3) : "\\text{undef.}"}`,
    mape: `\\text{MAPE} = \\frac{100\\%}{n}\\sum_i \\frac{|e_i|}{|y_i|} = \\frac{${show.map((q, i) => `\\frac{${fmtNum(m.abs[i])}}{${fmtNum(q.y)}}`).join(" + ")}${more}}{${m.n}} \\cdot 100\\% = ${fmtNum(m.mape)}\\%`,
    compare: `y \\to c\\,y: \\; \\text{MAE},\\text{RMSE} \\times c,\\; \\text{MSE} \\times c^2,\\; R^2, \\text{MAPE}\\ \\text{unchanged}`,
  };
  return <TexBlock>{tex[phase]}</TexBlock>;
}
