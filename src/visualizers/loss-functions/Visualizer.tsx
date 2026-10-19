import { useMemo, useState } from "react";
import { motion } from "motion/react";
import { RotateCcw, Shuffle } from "lucide-react";
import { usePlayback } from "../../hooks/usePlayback";
import { PlaybackControls } from "../../components/visualizer/PlaybackControls";
import { EventLog } from "../../components/visualizer/EventLog";
import { StepExplainer, Workspace } from "../../components/visualizer/Workspace";
import { AnimatedNumber } from "../../components/visualizer/AnimatedNumber";
import { Button, KV, Panel, Select, SimulatedLabel, Slider, Toggle, cx } from "../../components/ui";
import { TexBlock } from "../../components/Tex";
import type { VisualizerTab } from "../types";
import { Axes, StepCaption, ease, scale } from "../linear-regression/kit";
import {
  DEFAULT_LF, LOSS_IDS, LOSS_NAME, bce, bceGrad, gradOf, lossOf, mseGrad, mseProb, qOf, residuals, simulate, totalLoss,
  type ClsPhase, type LFConfig, type LossId, type Mode, type RegPhase,
} from "./simulation";

type Sim = ReturnType<typeof simulate>;
const COLOR: Record<LossId, string> = { mse: "var(--accent)", mae: "var(--violet)", huber: "var(--warn)" };
const DASH: Record<LossId, string | undefined> = { mse: undefined, mae: "7 4", huber: "2 3" };
const BG: Record<LossId, string> = { mse: "bg-accent/75", mae: "bg-violet/75", huber: "bg-warn/75" };
/** One schedule (seconds at 1×) shared by both charts and the card. */
const T = { drop: (i: number) => 0.15 + i * 0.08, line: 0.2, lineDur: 0.8, seg: (i: number) => 0.9 + i * 0.08, curve: 0, curveDur: 0.9, pop: (i: number) => 1 + i * 0.08, rise: 1.9, riseDur: 0.7, bar: (i: number) => 0.2 + i * 0.1, fit: 0.2, fitDur: 1.2 };
const STEP_MS: Record<RegPhase | ClsPhase, number> = {
  data: 1900, candidate: 2600, curves: 3300, bars: 2600, grads: 2600, "fit-mse": 2400, "fit-mae": 2400, "fit-huber": 2400, compare: 2200,
  examples: 2000, bce: 3000, mse: 3000, cbars: 2400, cgrads: 2600, csummary: 2000,
};

export default function LossFunctionsVisualizer({ tab }: { tab: VisualizerTab }) {
  const [cfg, setCfg] = useState<LFConfig>(DEFAULT_LF);
  const set = (p: Partial<LFConfig>) => setCfg((c) => ({ ...c, ...p }));
  const sim = useMemo(() => simulate(cfg), [cfg]);
  const duration = useMemo(() => (i: number) => STEP_MS[sim.steps[i]?.state.phase ?? "data"], [sim]);
  const pb = usePlayback(sim.steps.length, duration);
  const step = sim.steps[pb.index];
  const [replay, setReplay] = useState(0);
  const k = 1 / pb.speed;
  const animKey = `${step.id}-${replay}-${JSON.stringify(cfg)}`;
  if (tab === "metrics") return null;
  const phase = step.state.phase;
  const reg = cfg.mode === "regression";

  return (
    <Workspace
      main={
        <>
          <Panel title={reg ? "Regression losses: MSE vs MAE vs Huber" : "Classification losses: cross-entropy vs MSE"} actions={<SimulatedLabel>{reg ? "Synthetic data (seeded)" : "Hand-set predictions"}</SimulatedLabel>}>
            <StepCaption index={pb.index} total={sim.steps.length} stepKey={step.id} headline={step.state.headline} sub={step.state.sub} onReplay={() => setReplay((r) => r + 1)} />
            {reg ? (
              <div className="grid gap-3 md:grid-cols-2">
                <Scatter sim={sim} phase={phase as RegPhase} k={k} animKey={animKey} />
                <ResidualCurves sim={sim} phase={phase as RegPhase} k={k} animKey={animKey} />
              </div>
            ) : (
              <ProbCurves sim={sim} phase={phase as ClsPhase} k={k} animKey={animKey} />
            )}
            <div className="mt-2 flex flex-wrap gap-3 text-[11px] text-muted">
              {(reg ? LOSS_IDS : (["mse", "mae"] as LossId[])).map((id) => (
                <span key={id} className="flex items-center gap-1">
                  <svg width="22" height="6"><line x1="0" y1="3" x2="22" y2="3" stroke={COLOR[id]} strokeWidth="2.5" strokeDasharray={reg ? DASH[id] : id === "mae" ? DASH.mae : undefined} /></svg>
                  {reg ? `${LOSS_NAME[id]}${id === "huber" ? ` (δ = ${cfg.delta})` : ""}` : id === "mse" ? "binary cross-entropy −log q" : "MSE (1 − q)²"}
                </span>
              ))}
              {reg && <span className="flex items-center gap-1"><svg width="10" height="10"><path d="M1 1 L9 9 M9 1 L1 9" stroke="var(--err)" strokeWidth="2" /></svg>outlier</span>}
            </div>
          </Panel>
          <Panel title="What's happening in this step">
            <motion.div key={animKey} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.2 }}>
              {reg ? <RegCard sim={sim} phase={phase as RegPhase} k={k} /> : <ClsCard sim={sim} phase={phase as ClsPhase} k={k} />}
            </motion.div>
          </Panel>
          <PlaybackControls pb={pb} stepTitles={sim.steps.map((s) => s.title)} />
          <Panel title="Parameters" actions={<Button size="sm" variant="ghost" onClick={() => { setCfg(DEFAULT_LF); pb.restart(); }}><RotateCcw size={12} /> Reset</Button>}>
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              <Select label="Task" value={cfg.mode} options={[{ value: "regression", label: "Regression (MSE / MAE / Huber)" }, { value: "classification", label: "Classification (cross-entropy vs MSE)" }]} onChange={(v: Mode) => { set({ mode: v }); pb.restart(); }} />
              {reg ? (
                <>
                  <div className="space-y-2">
                    <Toggle label="Outlier" checked={cfg.outlier} onChange={(v) => set({ outlier: v })} hint="Break it: one point far below the trend." />
                    <Button size="sm" onClick={() => set({ seed: cfg.seed + 1 })}><Shuffle size={12} /> New sample (seed {cfg.seed})</Button>
                  </div>
                  <Slider label="Outlier distance" value={cfg.outlierDrop} min={1} max={12} step={0.5} format={(v) => v.toFixed(1)} onChange={(v) => set({ outlierDrop: v })} />
                  <Slider label="Huber δ" value={cfg.delta} min={0.25} max={4} step={0.25} format={(v) => v.toFixed(2)} onChange={(v) => set({ delta: v })} hint="Small δ ≈ MAE, large δ ≈ MSE (scaled by ½)." />
                </>
              ) : (
                <Toggle label="Include a confidently wrong prediction (G)" checked={cfg.confidentWrong} onChange={(v) => set({ confidentWrong: v })} hint="True class 1, model says 0.02." />
              )}
            </div>
          </Panel>
        </>
      }
      inspector={
        <>
          <StepExplainer index={pb.index} title={step.title} what={step.description} />
          <Panel title="Calculation">{reg ? <RegCalc sim={sim} phase={phase as RegPhase} /> : <ClsCalc sim={sim} />}</Panel>
          {reg && (
            <Panel title="Fitted lines">
              <KV rows={[["candidate", `${sim.candidate.intercept.toFixed(3)} + ${sim.candidate.slope.toFixed(3)}x`], ...LOSS_IDS.map((id) => [LOSS_NAME[id], `${sim.fits[id].intercept.toFixed(3)} + ${sim.fits[id].slope.toFixed(3)}x`] as [string, string]), ["true trend", "1 + 0.5x"]]} />
            </Panel>
          )}
          <Panel title="Events"><EventLog events={step.events} /></Panel>
        </>
      }
    />
  );
}

const REG_ORDER: RegPhase[] = ["data", "candidate", "curves", "bars", "grads", "fit-mse", "fit-mae", "fit-huber", "compare"];
const fitPhase = (p: RegPhase): LossId | null => (p.startsWith("fit-") ? (p.slice(4) as LossId) : null);

function Scatter({ sim, phase, k, animKey }: { sim: Sim; phase: RegPhase; k: number; animKey: string }) {
  const W = 380, H = 300, L = 36, R = 370, TP = 10, B = 266;
  const ys = sim.pts.map((p) => p.y);
  const yD: [number, number] = [Math.floor(Math.min(0, ...ys) - 0.5), Math.ceil(Math.max(...ys) + 0.8)];
  const sx = scale(0, 10, L, R), sy = scale(yD[0], yD[1], B, TP);
  const at = REG_ORDER.indexOf(phase);
  const fp = fitPhase(phase);
  const doneFits = LOSS_IDS.filter((id) => at > REG_ORDER.indexOf(`fit-${id}`) || phase === "compare");
  const lineXY = (l: { slope: number; intercept: number }) => ({ x1: sx(0), x2: sx(10), y1: sy(l.intercept), y2: sy(l.intercept + 10 * l.slope) });
  const segLine = fp ? sim.fits[fp] : sim.candidate;
  const segRes = residuals(sim.pts, segLine);
  const showSegs = at >= 1 && phase !== "compare";
  return (
    <div className="overflow-x-auto">
      <svg viewBox={`0 0 ${W} ${H + 20}`} className="h-auto w-full min-w-[300px]" role="img" aria-label="Scatter plot with candidate and fitted lines">
        <Axes sx={sx} sy={sy} xDomain={[0, 10]} yDomain={yD} L={L} R={R} T={TP} B={B} xLabel="x" yLabel="y" />
        {at >= 1 && (
          <motion.line key={phase === "candidate" ? `${animKey}-c` : "cand"} {...lineXY(sim.candidate)} stroke="var(--muted)" strokeWidth={2} strokeDasharray="5 4"
            initial={phase === "candidate" ? { pathLength: 0 } : false} animate={{ pathLength: 1, opacity: fp || phase === "compare" ? 0.35 : 1 }} transition={{ delay: T.line * k, duration: T.lineDur * k, ease }} />
        )}
        {doneFits.map((id) => <line key={id} {...lineXY(sim.fits[id])} stroke={COLOR[id]} strokeWidth={phase === "compare" ? 2.5 : 1.5} strokeDasharray={DASH[id]} opacity={phase === "compare" ? 1 : 0.45} />)}
        {fp && (
          <motion.line key={`${animKey}-fit`} stroke={COLOR[fp]} strokeWidth={3} strokeDasharray={DASH[fp]} initial={lineXY(sim.candidate)} animate={lineXY(sim.fits[fp])} transition={{ delay: T.fit * k, duration: T.fitDur * k, ease }} />
        )}
        {phase === "compare" && LOSS_IDS.map((id, i) => (
          <text key={id} x={sx(10) - 4} y={sy(sim.fits[id].intercept + 10 * sim.fits[id].slope) + (i - 1) * 12 - 4} fontSize={10} textAnchor="end" fill={COLOR[id]} fontWeight={600}>{LOSS_NAME[id]}</text>
        ))}
        {showSegs && sim.pts.map((p, i) => {
          const yh = sy(segLine.intercept + segLine.slope * p.x), y0 = sy(sim.candidate.intercept + sim.candidate.slope * p.x);
          return (
            <motion.line key={`${phase === "candidate" || fp ? animKey : "seg"}-${i}`} x1={sx(p.x)} x2={sx(p.x)} y2={sy(p.y)} stroke={segRes[i] >= 0 ? "var(--ok)" : "var(--err)"} strokeWidth={1.8}
              initial={phase === "candidate" ? { y1: sy(p.y), opacity: 0 } : fp ? { y1: y0 } : false} animate={{ y1: yh, opacity: 0.9 }}
              transition={phase === "candidate" ? { delay: T.seg(i) * k, duration: 0.35 * k } : { delay: T.fit * k, duration: T.fitDur * k, ease }} />
          );
        })}
        {sim.pts.map((p, i) => (
          <motion.g key={`${phase === "data" ? animKey : "pt"}-${i}`} initial={phase === "data" ? { opacity: 0, y: -30 } : false} animate={{ opacity: 1, y: 0 }} transition={{ delay: T.drop(i) * k, type: "spring", stiffness: 260, damping: 18 }}>
            {p.outlier ? <path d={`M${sx(p.x) - 5} ${sy(p.y) - 5} L${sx(p.x) + 5} ${sy(p.y) + 5} M${sx(p.x) + 5} ${sy(p.y) - 5} L${sx(p.x) - 5} ${sy(p.y) + 5}`} stroke="var(--err)" strokeWidth={2.5} />
              : <circle cx={sx(p.x)} cy={sy(p.y)} r={4.5} fill="var(--fg)" stroke="var(--surface)" strokeWidth={1.5} />}
          </motion.g>
        ))}
      </svg>
    </div>
  );
}

function ResidualCurves({ sim, phase, k, animKey }: { sim: Sim; phase: RegPhase; k: number; animKey: string }) {
  const W = 380, H = 300, L = 36, R = 370, TP = 10, B = 266;
  const delta = sim.cfg.delta;
  const Rm = Math.max(3, Math.ceil(Math.max(...sim.res.map(Math.abs)) + 0.5));
  const yMax = Math.max(4, Rm * 1.15);
  const sx = scale(-Rm, Rm, L, R), sy = scale(0, yMax, B, TP);
  const at = REG_ORDER.indexOf(phase);
  const fp = fitPhase(phase);
  const showCurves = at >= 2;
  const curve = (id: LossId) => Array.from({ length: 121 }, (_, i) => { const r = -Rm + (2 * Rm * i) / 120; return `${i ? "L" : "M"}${sx(r).toFixed(1)} ${sy(Math.min(lossOf(id, r, delta), yMax * 1.3)).toFixed(1)}`; }).join(" ");
  const fitRes = fp ? residuals(sim.pts, sim.fits[fp]) : null;
  const ids = fp ? [fp] : LOSS_IDS;
  const showDots = at >= 2 && phase !== "compare";
  const big = sim.pts.findIndex((p) => p.outlier);
  return (
    <div className="overflow-x-auto">
      <svg viewBox={`0 0 ${W} ${H + 20}`} className="h-auto w-full min-w-[300px]" role="img" aria-label="Loss as a function of residual for MSE, MAE and Huber, with one dot per data point">
        <defs><clipPath id="lf-clip"><rect x={L} y={TP} width={R - L} height={B - TP} /></clipPath></defs>
        <Axes sx={sx} sy={sy} xDomain={[-Rm, Rm]} yDomain={[0, yMax]} L={L} R={R} T={TP} B={B} xLabel="residual r = y − ŷ" yLabel="loss ℓ(r)" />
        {!showCurves && <text x={(L + R) / 2} y={(TP + B) / 2} textAnchor="middle" fontSize={12} fill="var(--muted)">loss curves appear in step 3</text>}
        <g clipPath="url(#lf-clip)">
          {showCurves && LOSS_IDS.map((id) => (
            <motion.path key={`${phase === "curves" ? animKey : "c"}-${id}`} d={curve(id)} fill="none" stroke={COLOR[id]} strokeWidth={2.5} strokeDasharray={DASH[id]}
              initial={phase === "curves" ? { pathLength: 0 } : false} animate={{ pathLength: 1, opacity: fp && fp !== id ? 0.15 : phase === "compare" ? 0.6 : 1 }} transition={{ delay: (T.curve + LOSS_IDS.indexOf(id) * 0.15) * k, duration: T.curveDur * k, ease }} />
          ))}
          {/* Tangent at the largest residual: the gradient each loss feeds back. */}
          {phase === "grads" && LOSS_IDS.map((id, j) => {
            const r0 = sim.res[big >= 0 ? big : sim.res.map(Math.abs).indexOf(Math.max(...sim.res.map(Math.abs)))];
            const g = gradOf(id, r0, delta), y0 = lossOf(id, r0, delta), dx = 0.9;
            return (
              <motion.line key={`${animKey}-t${id}`} x1={sx(r0 - dx)} y1={sy(y0 - g * dx)} x2={sx(r0 + dx)} y2={sy(y0 + g * dx)} stroke={COLOR[id]} strokeWidth={4} strokeLinecap="round" strokeOpacity={0.55}
                initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ delay: (0.3 + j * 0.4) * k, duration: 0.5 * k }} />
            );
          })}
          {showDots && sim.res.map((r0, i) => ids.map((id) => {
            const r1 = fitRes ? fitRes[i] : r0;
            const yv = Math.min(lossOf(id, r1, delta), yMax * 1.02);
            const isCurves = phase === "curves";
            return (
              <motion.circle key={`${isCurves || fp ? animKey : "d"}-${i}-${id}`} r={sim.pts[i].outlier ? 5.5 : 4} fill={COLOR[id]} stroke={sim.pts[i].outlier ? "var(--err)" : "var(--surface)"} strokeWidth={sim.pts[i].outlier ? 2 : 1}
                initial={isCurves ? { cx: sx(r0), cy: sy(0), opacity: 0 } : fp ? { cx: sx(r0), cy: sy(Math.min(lossOf(id, r0, delta), yMax * 1.02)) } : false}
                animate={isCurves ? { cx: sx(r0), cy: [sy(0), sy(0), sy(yv)], opacity: [0, 1, 1] } : { cx: sx(r1), cy: sy(yv), opacity: 1 }}
                transition={isCurves ? { delay: T.pop(i) * k, duration: (T.rise + T.riseDur - T.pop(i)) * k, times: [0, (T.rise - T.pop(i)) / (T.rise + T.riseDur - T.pop(i)), 1], ease: "easeOut" } : { delay: T.fit * k, duration: T.fitDur * k, ease }} />
            );
          }))}
        </g>
        {showDots && big >= 0 && !fp && lossOf("mse", sim.res[big], delta) > yMax && (
          <motion.text key={`${animKey}-off`} x={sx(Math.max(-Rm + 0.3, sim.res[big])) + 8} y={TP + 14} fontSize={11} fill="var(--accent)" fontWeight={600}
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: (phase === "curves" ? T.rise + T.riseDur : 0) * k }}>
            ↑ MSE = {lossOf("mse", sim.res[big], delta).toFixed(1)} (off the chart)
          </motion.text>
        )}
      </svg>
    </div>
  );
}

const CLS_ORDER: ClsPhase[] = ["examples", "bce", "mse", "cbars", "cgrads", "csummary"];

function ProbCurves({ sim, phase, k, animKey }: { sim: Sim; phase: ClsPhase; k: number; animKey: string }) {
  const W = 640, H = 300, L = 40, R = 628, TP = 10, B = 262;
  const yMax = 4.2;
  const sx = scale(0, 1, L, R), sy = scale(0, yMax, B, TP);
  const at = CLS_ORDER.indexOf(phase);
  const path = (fn: (q: number) => number) => Array.from({ length: 141 }, (_, i) => { const q = 0.005 + (0.995 * i) / 140; return `${i ? "L" : "M"}${sx(q).toFixed(1)} ${sy(Math.min(fn(q), yMax * 1.2)).toFixed(1)}`; }).join(" ");
  const curves = [{ id: "b", fn: bce, color: COLOR.mse, from: 1, dash: undefined }, { id: "m", fn: mseProb, color: COLOR.mae, from: 2, dash: DASH.mae }];
  return (
    <div className="overflow-x-auto">
      <svg viewBox={`0 0 ${W} ${H + 18}`} className="h-auto w-full min-w-[480px]" role="img" aria-label="Loss versus probability assigned to the true class">
        <defs><clipPath id="lf-cclip"><rect x={L} y={TP} width={R - L} height={B - TP} /></clipPath></defs>
        <Axes sx={sx} sy={sy} xDomain={[0, 1]} yDomain={[0, yMax]} L={L} R={R} T={TP} B={B} xLabel="q = probability given to the true class (1 = right & confident)" yLabel="loss" xCount={5} />
        <g clipPath="url(#lf-cclip)">
          {curves.map((c) => at >= c.from && (
            <motion.path key={`${CLS_ORDER[c.from] === phase ? animKey : "c"}-${c.id}`} d={path(c.fn)} fill="none" stroke={c.color} strokeWidth={2.5} strokeDasharray={c.dash}
              initial={CLS_ORDER[c.from] === phase ? { pathLength: 0 } : false} animate={{ pathLength: 1 }} transition={{ duration: T.curveDur * k, ease }} />
          ))}
          {sim.examples.map((e, i) => {
            const q = qOf(e);
            return curves.map((c) => {
              if (at < c.from && !(c.from === 1 && at === 0)) return null;
              const onCurve = at >= c.from;
              const rising = CLS_ORDER[c.from] === phase;
              return (
                <motion.circle key={`${rising || phase === "examples" ? animKey : "d"}-${e.id}-${c.id}`} cx={sx(q)} r={e.wrong ? 6 : 4.5} fill={c.color} stroke={e.wrong ? "var(--err)" : "var(--surface)"} strokeWidth={e.wrong ? 2 : 1}
                  initial={rising || phase === "examples" ? { cy: sy(0), opacity: 0 } : false}
                  animate={{ cy: onCurve ? sy(Math.min(c.fn(q), yMax)) : sy(0), opacity: 1 }}
                  transition={{ delay: (rising ? T.curveDur + 0.1 + i * 0.12 : T.drop(i)) * k, duration: (rising ? 0.6 : 0.3) * k, ease }} />
              );
            });
          })}
        </g>
        {sim.examples.map((e, i) => (
          <motion.text key={`${phase === "examples" ? animKey : "l"}-${e.id}`} x={sx(qOf(e))} y={B + 26} fontSize={10} textAnchor="middle" fill={e.wrong ? "var(--err)" : "var(--fg)"} fontWeight={600}
            initial={phase === "examples" ? { opacity: 0 } : false} animate={{ opacity: 1 }} transition={{ delay: T.drop(i) * k }}>
            {e.id}
          </motion.text>
        ))}
      </svg>
    </div>
  );
}

function Bars({ rows, k, max, digits = 2 }: { rows: { label: string; v: number; cls: string; hl?: boolean }[]; k: number; max?: number; digits?: number }) {
  const m = max ?? Math.max(1e-9, ...rows.map((r) => Math.abs(r.v)));
  return (
    <div className="space-y-1">
      {rows.map((r, i) => (
        <div key={i} className={cx("grid grid-cols-[96px_minmax(0,1fr)_52px] items-center gap-2 text-xs", r.hl && "font-semibold")}>
          <span className={cx("truncate font-mono text-[10px]", r.hl ? "text-err" : "text-muted")}>{r.label}</span>
          <div className="h-3 rounded bg-surface-2"><motion.div className={cx("h-full rounded", r.cls)} initial={{ width: 0 }} animate={{ width: `${(100 * Math.abs(r.v)) / m}%` }} transition={{ delay: T.bar(i) * k, duration: 0.45 * k, ease }} /></div>
          <span className="text-right font-mono text-[11px]"><AnimatedNumber value={r.v} from={0} digits={digits} delay={T.bar(i) * k} duration={0.45 * k} /></span>
        </div>
      ))}
    </div>
  );
}

function RegCard({ sim, phase, k }: { sim: Sim; phase: RegPhase; k: number }) {
  const { pts, res, cfg } = sim;
  const lbl = (i: number) => `${pts[i].outlier ? "outlier " : ""}(${pts[i].x}, ${pts[i].y})`;
  if (phase === "data") return <p className="text-sm text-muted">{pts.length} points. Next we measure how far each one is from a candidate line, then price those misses three different ways.</p>;
  if (phase === "candidate") return <Bars k={k} rows={res.map((r, i) => ({ label: lbl(i), v: r, cls: r >= 0 ? "bg-ok/70" : "bg-err/70", hl: pts[i].outlier }))} />;
  if (phase === "curves") {
    return (
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead className="text-left text-muted"><tr><th className="py-0.5">point</th><th>r</th>{LOSS_IDS.map((id) => <th key={id} style={{ color: COLOR[id] }}>{LOSS_NAME[id]}</th>)}</tr></thead>
          <tbody className="font-mono">
            {res.map((r, i) => (
              <motion.tr key={i} className={cx("border-t border-line", pts[i].outlier && "text-err")} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: T.pop(i) * k }}>
                <td className="py-0.5">{lbl(i)}</td><td>{r.toFixed(2)}</td>
                {LOSS_IDS.map((id) => <td key={id}><AnimatedNumber value={lossOf(id, r, cfg.delta)} from={0} delay={T.rise * k} duration={T.riseDur * k} /></td>)}
              </motion.tr>
            ))}
            <tr className="border-t-2 border-line font-semibold"><td className="py-0.5">mean</td><td />{LOSS_IDS.map((id) => <td key={id}>{totalLoss(id, pts, sim.candidate, cfg.delta).toFixed(2)}</td>)}</tr>
          </tbody>
        </table>
      </div>
    );
  }
  if (phase === "bars") {
    return (
      <div className="space-y-2">
        <p className="text-xs text-muted">Each bar is 100% of that loss's total; segments are points{cfg.outlier ? ", the red-outlined segment is the outlier" : ""}.</p>
        {LOSS_IDS.map((id, j) => {
          const ls = res.map((r) => lossOf(id, r, cfg.delta));
          const tot = ls.reduce((s, v) => s + v, 0) || 1;
          return (
            <div key={id} className="grid grid-cols-[52px_minmax(0,1fr)_44px] items-center gap-2 text-xs">
              <span style={{ color: COLOR[id] }} className="font-semibold">{LOSS_NAME[id]}</span>
              <div className="flex h-5 overflow-hidden rounded bg-surface-2">
                {ls.map((v, i) => (
                  <motion.div key={i} className={cx("h-full border-r border-surface", BG[id], pts[i].outlier && "outline-2 -outline-offset-2 outline-err")} style={{ opacity: pts[i].outlier ? 1 : 0.55 + (i % 2) * 0.2 }}
                    initial={{ width: 0 }} animate={{ width: `${(100 * v) / tot}%` }} transition={{ delay: (0.2 + j * 0.5 + i * 0.04) * k, duration: 0.4 * k, ease }} />
                ))}
              </div>
              <span className="text-right font-mono">{cfg.outlier ? `${(100 * sim.share(id)).toFixed(0)}%` : ""}</span>
            </div>
          );
        })}
      </div>
    );
  }
  if (phase === "grads") {
    const m = Math.max(...res.flatMap((r) => LOSS_IDS.map((id) => Math.abs(gradOf(id, r, cfg.delta)))));
    return (
      <div className="space-y-2">
        <p className="text-xs text-muted">|dℓ/dr| per point: how strongly it pulls the line toward itself.</p>
        <div className="grid gap-3 sm:grid-cols-3">
          {LOSS_IDS.map((id) => (
            <div key={id}>
              <div className="mb-1 text-xs font-semibold" style={{ color: COLOR[id] }}>{LOSS_NAME[id]} · {id === "mse" ? "2r" : id === "mae" ? "sign(r)" : `clip(r, ±${cfg.delta})`}</div>
              <Bars k={k} max={m} rows={res.map((r, i) => ({ label: pts[i].outlier ? "outlier" : `pt ${i + 1}`, v: Math.abs(gradOf(id, r, cfg.delta)), cls: BG[id], hl: pts[i].outlier }))} />
            </div>
          ))}
        </div>
      </div>
    );
  }
  const fp = fitPhase(phase);
  if (fp) {
    const l = sim.fits[fp];
    return (
      <div className="grid gap-2 text-sm sm:grid-cols-3">
        {[["slope", sim.candidate.slope, l.slope], ["intercept", sim.candidate.intercept, l.intercept], [`mean ${LOSS_NAME[fp]}`, totalLoss(fp, pts, sim.candidate, cfg.delta), totalLoss(fp, pts, l, cfg.delta)]].map(([n, a, b]) => (
          <div key={n as string} className="rounded-lg border border-line p-2">
            <div className="text-xs text-muted">{n as string}</div>
            <div className="font-mono text-lg" style={{ color: COLOR[fp] }}><AnimatedNumber value={b as number} from={a as number} digits={3} delay={T.fit * k} duration={T.fitDur * k} /></div>
            <div className="text-[11px] text-muted">candidate: {(a as number).toFixed(3)}</div>
          </div>
        ))}
        <p className="text-xs text-muted sm:col-span-3">{fp === "mse" ? "Closed-form least squares." : "Found by iteratively reweighted least squares: points with big residuals get smaller weights each round."} The residual dots on the right slide to their new positions under this fit.</p>
      </div>
    );
  }
  return (
    <div className="space-y-1">
      <p className="text-xs text-muted">Slope of each fit vs the true trend slope 0.5 (bar = |slope − 0.5|, shorter is better).</p>
      <Bars k={k} digits={3} max={Math.max(0.3, ...LOSS_IDS.map((id) => Math.abs(sim.fits[id].slope - 0.5)))} rows={LOSS_IDS.map((id) => ({ label: `${LOSS_NAME[id]} ${sim.fits[id].slope.toFixed(3)}`, v: Math.abs(sim.fits[id].slope - 0.5), cls: BG[id] }))} />
    </div>
  );
}

function ClsCard({ sim, phase, k }: { sim: Sim; phase: ClsPhase; k: number }) {
  const ex = sim.examples;
  const label = (i: number) => `${ex[i].id}: y=${ex[i].y} p=${ex[i].p}`;
  if (phase === "examples") {
    return (
      <div className="flex flex-wrap gap-1.5">
        {ex.map((e, i) => (
          <motion.span key={e.id} initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: T.drop(i) * k }} className={cx("rounded border px-2 py-1 font-mono text-xs", e.wrong ? "border-err/50 bg-err-soft" : "border-line bg-surface-2")}>
            {e.id}: true {e.y}, p(1) = {e.p} → q = {qOf(e).toFixed(2)}
          </motion.span>
        ))}
      </div>
    );
  }
  if (phase === "bce" || phase === "mse") {
    const fn = phase === "bce" ? bce : mseProb;
    return <Bars k={k} max={Math.max(1, ...ex.map((e) => fn(qOf(e))))} rows={ex.map((e, i) => ({ label: label(i), v: fn(qOf(e)), cls: phase === "bce" ? BG.mse : BG.mae, hl: e.wrong }))} />;
  }
  if (phase === "cbars" || phase === "cgrads") {
    const [fa, fb] = phase === "cbars" ? [bce, mseProb] : [bceGrad, mseGrad];
    const m = Math.max(...ex.flatMap((e) => [fa(qOf(e)), fb(qOf(e))]));
    return (
      <div className="grid gap-3 sm:grid-cols-2">
        {[["cross-entropy", fa, BG.mse], ["MSE", fb, BG.mae]].map(([n, fn, c]) => (
          <div key={n as string}>
            <div className="mb-1 text-xs font-semibold">{n as string} {phase === "cgrads" ? "|∂/∂z|" : "loss"}</div>
            <Bars k={k} max={m} digits={3} rows={ex.map((e, i) => ({ label: label(i), v: (fn as (q: number) => number)(qOf(e)), cls: c as string, hl: e.wrong }))} />
          </div>
        ))}
      </div>
    );
  }
  const mb = ex.reduce((s, e) => s + bce(qOf(e)), 0) / ex.length, mm = ex.reduce((s, e) => s + mseProb(qOf(e)), 0) / ex.length;
  return (
    <div className="space-y-2 text-sm">
      <KV rows={[["mean cross-entropy", mb.toFixed(4)], ["mean MSE", mm.toFixed(4)]]} />
      <p className="text-xs text-muted">Cross-entropy is the negative log-likelihood of the labels: minimizing it is maximum-likelihood training of a probabilistic classifier. That is why logistic regression and neural classifiers use it.</p>
    </div>
  );
}

function RegCalc({ sim, phase }: { sim: Sim; phase: RegPhase }) {
  const d = sim.cfg.delta;
  const big = sim.pts.findIndex((p) => p.outlier);
  const r = big >= 0 ? sim.res[big] : sim.res[0];
  const f = (v: number) => v.toFixed(3);
  const n = (v: number) => (v < 0 ? `(${f(v)})` : f(v));
  const defs = `\\begin{aligned} \\ell_{\\text{MSE}}(r) &= r^2 \\\\ \\ell_{\\text{MAE}}(r) &= |r| \\\\ \\ell_{\\text{Huber}}(r) &= \\begin{cases} \\tfrac12 r^2 & |r| \\le \\delta \\\\ \\delta\\,(|r| - \\tfrac12\\delta) & |r| > \\delta \\end{cases} \\end{aligned}`;
  if (phase === "grads")
    return <TexBlock>{`\\begin{aligned} r &= ${f(r)} \\\\ \\tfrac{d\\ell_{\\text{MSE}}}{dr} &= 2r = ${f(2 * r)} \\\\ \\tfrac{d\\ell_{\\text{MAE}}}{dr} &= \\operatorname{sign}(r) = ${Math.sign(r)} \\\\ \\tfrac{d\\ell_{\\text{Huber}}}{dr} &= \\operatorname{clip}(r, \\pm ${d}) = ${f(gradOf("huber", r, d))} \\end{aligned}`}</TexBlock>;
  if (phase === "curves" || phase === "bars")
    return (
      <div className="space-y-2">
        <TexBlock>{defs}</TexBlock>
        <TexBlock>{`r = ${f(r)}: \\quad ${n(r)}^2 = ${f(r * r)},\\ \\ |r| = ${f(Math.abs(r))},\\ \\ \\text{Huber} = ${f(lossOf("huber", r, d))}`}</TexBlock>
      </div>
    );
  const fp = fitPhase(phase);
  if (fp) return <TexBlock>{`\\min_{a,b}\\ \\frac{1}{n}\\sum_i \\ell_{\\text{${LOSS_NAME[fp]}}}\\big(y_i - a - b x_i\\big) \\;\\Rightarrow\\; \\hat{y} = ${f(sim.fits[fp].intercept)} + ${f(sim.fits[fp].slope)}\\,x`}</TexBlock>;
  return <TexBlock>{defs}</TexBlock>;
}

function ClsCalc({ sim }: { sim: Sim }) {
  const g = sim.examples.find((e) => e.wrong) ?? sim.examples[2];
  const q = qOf(g);
  return (
    <div className="space-y-2">
      <TexBlock>{"\\text{BCE} = -\\big[y\\log p + (1-y)\\log(1-p)\\big] = -\\log q"}</TexBlock>
      <TexBlock>{`\\text{example } ${g.id}:\\ q = ${q.toFixed(2)} \\Rightarrow -\\log q = ${bce(q).toFixed(3)},\\ \\ (1-q)^2 = ${mseProb(q).toFixed(3)}`}</TexBlock>
      <TexBlock>{`\\left|\\tfrac{\\partial \\text{BCE}}{\\partial z}\\right| = 1 - q = ${bceGrad(q).toFixed(3)} \\qquad \\left|\\tfrac{\\partial \\text{MSE}}{\\partial z}\\right| = 2q(1-q)^2 = ${mseGrad(q).toFixed(3)}`}</TexBlock>
    </div>
  );
}
