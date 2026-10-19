import { useCallback, useMemo, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { RotateCcw } from "lucide-react";
import { usePlayback } from "../../hooks/usePlayback";
import { PlaybackControls } from "../../components/visualizer/PlaybackControls";
import { EventLog } from "../../components/visualizer/EventLog";
import { StepExplainer, Workspace } from "../../components/visualizer/Workspace";
import { AnimatedNumber } from "../../components/visualizer/AnimatedNumber";
import { Badge, Button, Callout, KV, Panel, Select, Slider, Toggle, cx } from "../../components/ui";
import { TexBlock } from "../../components/Tex";
import type { VisualizerTab } from "../types";
import { EASE, StepCaption } from "../decision-trees/StepCaption";
import { N_ROWS, REF_DATE, TRANSFORMS, ageYears, simulate, type Col, type FeConfig, type FeSim, type FeState, type StageId, type TransformId } from "./simulation";

const ALL_ON: Record<TransformId, boolean> = { date: true, log: true, onehot: true, scale: true, interact: true };
const DEFAULT: FeConfig = { enabled: ALL_ON, leakScaler: false, lambda: 0.02, seed: 2 };
const LAMBDAS = [0, 0.005, 0.02, 0.1, 0.5];
const PRESETS: { label: string; cfg: Partial<FeConfig> }[] = [
  { label: "Raw only (all off)", cfg: { enabled: { date: false, log: false, onehot: false, scale: false, interact: false }, leakScaler: false } },
  { label: "All transforms", cfg: { enabled: ALL_ON, leakScaler: false } },
  { label: "Break it: scaler fit on all rows", cfg: { enabled: ALL_ON, leakScaler: true } },
  { label: "Plain least squares (λ = 0)", cfg: { lambda: 0 } },
];
/** One schedule (seconds at 1×): old column fades, new column fills row by row, then the model refits. */
const T = { drop: 0.2, head: 0.6, cells: 0.85, row: 0.09, refit: 2.0, refitDur: 0.9 };
const DURATION: Record<StageId, number> = { raw: 3400, date: 3800, log: 3800, onehot: 3800, scale: 3800, interact: 3800, summary: 4200 };
const STAGES: StageId[] = ["raw", "date", "log", "onehot", "scale", "interact"];
const STAGE_LABEL: Record<StageId, string> = { raw: "raw", date: "age", log: "log", onehot: "one-hot", scale: "scale", interact: "interact", summary: "" };

export default function FeatureEngineeringVisualizer({ tab }: { tab: VisualizerTab }) {
  const [cfg, setCfg] = useState(DEFAULT);
  const set = (p: Partial<FeConfig>) => setCfg((c) => ({ ...c, ...p }));
  const sim = useMemo(() => simulate(cfg), [cfg]);
  const duration = useCallback((i: number) => DURATION[sim.steps[i]?.id as StageId] ?? 3800, [sim]);
  const pb = usePlayback(sim.steps.length, duration);
  const [replay, setReplay] = useState(0);
  const step = sim.steps[pb.index];
  const prev = pb.index > 0 ? sim.steps[pb.index - 1].state : null;
  const k = 1 / pb.speed;
  const animKey = `${step.id}-${replay}`;
  if (tab === "metrics") return null;
  const st = step.state;
  const shownRows = [...sim.trainIdx.slice(0, 5), ...sim.testIdx.slice(0, 3)];

  return (
    <Workspace
      main={
        <>
          <Panel title="House prices table" actions={<span className="flex gap-1"><Badge>{N_ROWS} rows</Badge><Badge tone={cfg.leakScaler ? "warn" : "neutral"}>{cfg.leakScaler ? "scaler leaks" : "train-only scaler"}</Badge></span>}>
            <StepCaption index={pb.index} total={sim.steps.length} stepId={step.id} headline={st.headline} sub={st.sub} onReplay={() => setReplay((r) => r + 1)} />
            <TableView sim={sim} st={st} prev={prev} rows={shownRows} k={k} animKey={animKey} />
            <R2Meter st={st} k={k} animKey={animKey} />
            <p className="mt-1 text-[11px] text-muted">Showing 8 of {N_ROWS} rows (5 training, 3 held-out). Price is in thousands of dollars. ŷ is the linear model's prediction after this step's refit.</p>
          </Panel>
          <Panel title="What's happening in this step">
            <AnimatePresence mode="wait" initial={false}>
              <motion.div key={animKey} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: 0.2 }}>
                <StepCard sim={sim} index={pb.index} k={k} leak={cfg.leakScaler} />
              </motion.div>
            </AnimatePresence>
          </Panel>
          <PlaybackControls pb={pb} stepTitles={sim.steps.map((s) => s.title)} />
          <Panel title="Parameters" actions={<Button size="sm" variant="ghost" onClick={() => setCfg(DEFAULT)}><RotateCcw size={12} /> Reset</Button>}>
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <div className="text-xs font-medium">Transforms (each one is a step)</div>
                {TRANSFORMS.map((t) => <Toggle key={t.id} label={t.label} checked={cfg.enabled[t.id]} onChange={(v) => set({ enabled: { ...cfg.enabled, [t.id]: v } })} />)}
              </div>
              <div className="space-y-3">
                <Toggle label="Fit the scaler on ALL rows (leak)" checked={cfg.leakScaler} onChange={(v) => set({ leakScaler: v })} hint="The classic mistake: compute μ and σ before splitting, so held-out rows shape the features." />
                <Select label="Ridge penalty λ" value={String(cfg.lambda)} options={LAMBDAS.map((l) => ({ value: String(l), label: l === 0 ? "0 (plain least squares)" : String(l) }))} onChange={(v) => set({ lambda: Number(v) })} />
                <Slider label="Random seed" value={cfg.seed} min={1} max={30} onChange={(v) => set({ seed: v })} hint="New houses and a new train/held-out split." />
              </div>
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
            why="A linear model can only combine the columns you give it. Most of the accuracy here comes from turning strings and categories into numbers with the right shape, not from the model." />
          <Panel title="Calculation"><Calculation sim={sim} st={st} lambda={cfg.lambda} /></Panel>
          <Panel title="Model after this step">
            <KV rows={[
              ["held-out R²", st.r2Test.toFixed(3)],
              ["training R²", st.r2Train.toFixed(3)],
              ["λ", cfg.lambda],
              ["intercept", st.weights[0].toFixed(2)],
              ...st.features.map((f, i): [string, string] => [`w · ${f}`, st.weights[i + 1].toFixed(i === 0 && f === "sqft" ? 4 : 2)]),
            ]} />
          </Panel>
          <Panel title="Events"><EventLog events={step.events} /></Panel>
        </>
      }
    />
  );
}

const fmtCell = (c: Col, v: number) => (c.key.startsWith("is_") ? v.toFixed(0) : c.key === "sqft" && !c.scaled ? v.toFixed(0) : c.key === "age" && !c.scaled ? v.toFixed(1) : v.toFixed(2));

function TableView({ sim, st, prev, rows, k, animKey }: { sim: FeSim; st: FeState; prev: FeState | null; rows: number[]; k: number; animKey: string }) {
  const fresh = st.stage === "raw";
  const prevPreds = prev?.preds;
  return (
    <div className="overflow-x-auto rounded-md border border-line">
      <div className="flex min-w-max">
        <div className="sticky left-0 z-10 border-r border-line bg-surface">
          <div className="h-12 border-b border-line px-2 py-1 text-[11px] font-semibold text-muted">row</div>
          {rows.map((r) => (
            <div key={r} className="flex h-7 items-center gap-1 border-b border-line px-2 font-mono text-[11px] last:border-b-0">
              #{r}{sim.houses[r].test && <Badge tone="warn">held-out</Badge>}
            </div>
          ))}
        </div>
        <AnimatePresence>
          {st.cols.map((c) => {
            const prevCol = prev?.cols.find((p) => p.key === c.key);
            return (
              <motion.div key={c.key} layout="position" initial={{ opacity: 0, width: 0 }} animate={{ opacity: 1, width: "auto" }} exit={{ opacity: 0, width: 0 }}
                transition={{ layout: { duration: 0.5 * k, ease: EASE }, width: { duration: 0.5 * k }, opacity: { duration: 0.4 * k, delay: c.kind === "new" ? T.head * k : 0 } }}
                className={cx("min-w-[86px] overflow-hidden border-r border-line", c.kind === "new" && "bg-ok-soft/40", c.kind === "dropped" && "bg-err-soft/30", c.kind === "changed" && "bg-accent-soft/40")}>
                <div className="flex h-12 flex-col justify-center border-b border-line px-2">
                  <span className={cx("whitespace-nowrap font-mono text-[11px] font-semibold", c.kind === "dropped" && "line-through")}>{c.label}</span>
                  <span className="text-[10px] text-muted">{c.kind === "new" ? "＋ new" : c.kind === "dropped" ? "－ dropped" : c.kind === "changed" ? "↻ rescaled" : c.numeric ? "number · used" : "text · not usable"}</span>
                </div>
                {rows.map((r, j) => {
                  const text = c.text ? c.text[r] : null;
                  const v = c.values?.[r];
                  return (
                    <motion.div key={`${r}-${animKey}`} className={cx("flex h-7 items-center border-b border-line px-2 font-mono text-[11px] whitespace-nowrap last:border-b-0", !c.numeric && "text-muted")}
                      initial={c.kind === "new" || fresh ? { opacity: 0, x: -8 } : false}
                      animate={{ opacity: c.kind === "dropped" ? 0.35 : 1, x: 0 }}
                      transition={{ delay: (c.kind === "dropped" ? T.drop + j * 0.05 : (fresh ? 0.2 : T.cells) + j * T.row) * k, duration: 0.3 * k }}>
                      {text ?? (c.kind === "changed" && prevCol?.values
                        ? <AnimatedNumber key={animKey} value={v!} from={prevCol.values[r]} digits={2} delay={(T.cells + j * T.row) * k} duration={0.5 * k} />
                        : fmtCell(c, v!))}
                    </motion.div>
                  );
                })}
              </motion.div>
            );
          })}
        </AnimatePresence>
          <motion.div key="price" layout="position" transition={{ layout: { duration: 0.5 * k, ease: EASE } }} className="min-w-[70px] border-r border-line bg-surface-2/60">
            <div className="flex h-12 flex-col justify-center border-b border-line px-2"><span className="font-mono text-[11px] font-semibold">price</span><span className="text-[10px] text-muted">target y</span></div>
            {rows.map((r) => <div key={r} className="flex h-7 items-center border-b border-line px-2 font-mono text-[11px] last:border-b-0">{sim.houses[r].price}</div>)}
          </motion.div>
          <motion.div key="pred" layout="position" transition={{ layout: { duration: 0.5 * k, ease: EASE } }} className="min-w-[78px] bg-surface-2/60">
            <div className="flex h-12 flex-col justify-center border-b border-line px-2"><span className="font-mono text-[11px] font-semibold">ŷ</span><span className="text-[10px] text-muted">prediction</span></div>
            {rows.map((r) => {
              const err = Math.abs(st.preds[r] - sim.houses[r].price);
              return (
                <div key={r} className="flex h-7 items-center justify-between gap-1 border-b border-line px-2 font-mono text-[11px] last:border-b-0">
                  <AnimatedNumber key={animKey} value={st.preds[r]} from={prevPreds ? prevPreds[r] : st.preds[r]} digits={0} delay={T.refit * k} duration={T.refitDur * k} />
                  <motion.span key={`e-${animKey}`} className={cx("text-[9px]", err > 40 ? "text-err" : "text-muted")} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: (T.refit + T.refitDur) * k }}>±{err.toFixed(0)}</motion.span>
                </div>
              );
            })}
          </motion.div>
      </div>
    </div>
  );
}

function R2Meter({ st, k, animKey }: { st: FeState; k: number; animKey: string }) {
  const from = Number.isFinite(st.prevR2) ? st.prevR2 : 0;
  const pct = (v: number) => `${Math.max(0, Math.min(1, v)) * 100}%`;
  const d = Number.isFinite(st.prevR2) ? st.r2Test - st.prevR2 : null;
  return (
    <div className="mt-3 grid grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-2 text-xs" key={animKey}>
      <span className="font-semibold">held-out R²</span>
      <div className="relative h-5 rounded bg-surface-2">
        <motion.div className="absolute inset-y-0.5 left-0 rounded bg-accent" initial={{ width: pct(from) }} animate={{ width: pct(st.r2Test) }} transition={{ delay: T.refit * k, duration: T.refitDur * k, ease: EASE }} />
        {Number.isFinite(st.prevR2) && <div className="absolute inset-y-0 w-0.5 bg-fg/60" style={{ left: pct(st.prevR2) }} title="before this step" />}
      </div>
      <span className="flex items-center gap-1 font-mono">
        <AnimatedNumber value={st.r2Test} from={from} digits={3} delay={T.refit * k} duration={T.refitDur * k} />
        {d !== null && st.stage !== "summary" && (
          <motion.span initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: (T.refit + T.refitDur) * k }}>
            <Badge tone={!st.on ? "neutral" : d > 0.001 ? "ok" : d < -0.001 ? "err" : "neutral"}>{!st.on ? "skipped" : `${d >= 0 ? "+" : ""}${d.toFixed(3)}`}</Badge>
          </motion.span>
        )}
      </span>
    </div>
  );
}

/** Waterfall: one bar per stage from the previous R² to the new one. */
function Waterfall({ sim, upTo, k }: { sim: FeSim; upTo: number; k: number }) {
  const W = 520, H = 170, L = 34, B = 26, TOP = 10;
  const ys = sim.steps.slice(0, 6).map((s) => s.state.r2Test);
  const lo = Math.min(0, ...ys);
  const sy = (v: number) => TOP + (1 - (v - lo) / (1 - lo)) * (H - TOP - B);
  const bw = (W - L) / 6;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label="Held-out R² after each transform">
      {[0, 0.5, 1].map((t) => (
        <g key={t}><line x1={L} x2={W} y1={sy(t)} y2={sy(t)} stroke="var(--line)" /><text x={L - 4} y={sy(t) + 3} fontSize={9} textAnchor="end" fill="var(--muted)">{t}</text></g>
      ))}
      {STAGES.map((stage, i) => {
        const s = sim.steps[i].state;
        if (i > upTo) return <text key={stage} x={L + bw * i + bw / 2} y={H - 8} fontSize={10} textAnchor="middle" fill="var(--muted)" opacity={0.5}>{STAGE_LABEL[stage]}</text>;
        const base = i === 0 ? Math.max(lo, 0) : sim.steps[i - 1].state.r2Test;
        const top = Math.max(base, s.r2Test), bot = Math.min(base, s.r2Test);
        const current = i === upTo;
        const d = s.r2Test - base;
        const color = i === 0 ? "var(--muted)" : !s.on || Math.abs(d) < 0.005 ? "var(--line)" : d > 0 ? "var(--ok)" : "var(--err)";
        const x = L + bw * i + 8;
        const delay = (current ? T.refit : 0.1 + i * 0.25) * k;
        return (
          <g key={stage}>
            <motion.rect x={x} width={bw - 16} fill={color} rx={2} initial={{ y: sy(base), height: 0 }} animate={{ y: sy(top), height: Math.max(1.5, sy(bot) - sy(top)) }} transition={{ delay, duration: 0.6 * k, ease: EASE }} />
            {i > 0 && <line x1={x - 16} x2={x} y1={sy(base)} y2={sy(base)} stroke="var(--muted)" strokeDasharray="2 2" />}
            <motion.text x={x + (bw - 16) / 2} y={sy(top) - 4} fontSize={9.5} textAnchor="middle" fontFamily="var(--font-mono)" fill="var(--fg)" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: delay + 0.5 * k }}>
              {i === 0 ? s.r2Test.toFixed(2) : !s.on ? "skip" : Math.abs(d) < 0.005 ? "±0.00" : `${d > 0 ? "+" : ""}${d.toFixed(2)}`}
            </motion.text>
            <text x={x + (bw - 16) / 2} y={H - 8} fontSize={10} textAnchor="middle" fill={current ? "var(--accent)" : "var(--muted)"} fontWeight={current ? 700 : 400}>{STAGE_LABEL[stage]}</text>
          </g>
        );
      })}
    </svg>
  );
}

function StepCard({ sim, index, k, leak }: { sim: FeSim; index: number; k: number; leak: boolean }) {
  const st = sim.steps[index].state;
  const h = sim.houses[0];
  const example: Partial<Record<StageId, string>> = {
    raw: `Row #0: "${h.built}", ${h.sqft} sqft, "${h.hood}" → the model sees only [${h.sqft}]`,
    date: `"${h.built}" → (${REF_DATE} − ${h.built}) / 365.25 = ${ageYears(h.built).toFixed(1)} years`,
    log: `${h.sqft} sqft → ln(${h.sqft}) = ${Math.log(h.sqft).toFixed(3)}`,
    onehot: `"${h.hood}" → is_Hilltop = ${h.hood === "Hilltop" ? 1 : 0}, is_Downtown = ${h.hood === "Downtown" ? 1 : 0}`,
    interact: `is_Downtown = ${h.hood === "Downtown" ? 1 : 0} → size × Downtown = ${h.hood === "Downtown" ? "size" : "0"}`,
  };
  return (
    <div className="grid gap-4 md:grid-cols-[minmax(0,1.1fr)_minmax(0,0.9fr)]">
      <div>
        <h4 className="mb-1 text-sm font-semibold">Held-out R² after each step</h4>
        <Waterfall sim={sim} upTo={Math.min(index, 5)} k={k} />
      </div>
      <div className="space-y-2 text-xs leading-relaxed">
        {!st.on && <Callout tone="neutral" title="Skipped">This transform is toggled off, so the bar stays flat. That flat bar is its contribution, measured by its absence.</Callout>}
        {st.on && example[st.stage] && (
          <motion.div className="rounded-md border border-line bg-surface-2 p-2 font-mono" initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: T.head * k, duration: 0.4 * k }}>{example[st.stage]}</motion.div>
        )}
        {st.stage === "scale" && st.on && (
          <div className="space-y-1">
            <table className="w-full font-mono text-[11px]">
              <thead className="text-left text-muted"><tr><th>column</th><th>μ used</th><th>σ used</th><th>μ train-only</th></tr></thead>
              <tbody>
                {st.scalers.map((s, i) => (
                  <motion.tr key={s.key} className={cx("border-t border-line", leak && "bg-warn-soft")} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: (T.cells + i * 0.2) * k }}>
                    <td>{s.key}</td><td>{s.mu.toFixed(3)}</td><td>{s.sd.toFixed(3)}</td><td>{s.muTrain.toFixed(3)}</td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
            {leak
              ? <Callout tone="warn" title="Leak">μ and σ were computed with the held-out rows included. Here the effect on R² is small (a mean/std scaler leaks little), but the held-out score is no longer an honest estimate. The same mistake with target encoding, imputation or feature selection can inflate scores a lot.</Callout>
              : <p className="text-muted">The scaler is fit on training rows only and then applied to held-out rows unchanged, exactly as it will be applied to new houses in production.</p>}
          </div>
        )}
        {st.stage === "summary" && (
          <p>Raw sqft alone explains {(sim.steps[0].state.r2Test * 100).toFixed(0)}% of the held-out variance; the engineered table explains {(st.r2Test * 100).toFixed(0)}%. The model never changed: still one weight per column. Contributions depend on order and on what else is in the model, so toggle one transform off and replay to measure it alone.</p>
        )}
        {st.stage !== "summary" && st.stage !== "scale" && st.on && (
          <p className="text-muted">{st.stage === "raw" ? "Baseline: one usable column." : `The model is refit on the 28 training rows with the new columns; the bar shows what it gained (or lost) on the 12 held-out rows.`}</p>
        )}
      </div>
    </div>
  );
}

function Calculation({ sim, st, lambda }: { sim: FeSim; st: FeState; lambda: number }) {
  const h = sim.houses[0];
  const ridge = `\\min_{w}\\ \\sum_{i \\in \\text{train}} (y_i - \\hat{y}_i)^2 + \\lambda n \\sum_{j \\ge 1} w_j^2, \\quad \\lambda = ${lambda}`;
  const r2 = `R^2_{\\text{held-out}} = 1 - \\frac{\\sum (y_i - \\hat{y}_i)^2}{\\sum (y_i - \\bar{y})^2} = ${st.r2Test.toFixed(3)}`;
  const specific: Partial<Record<StageId, string>> = {
    raw: `\\hat{y} = ${st.weights[0].toFixed(1)} + ${st.weights[1].toFixed(4)} \\cdot \\text{sqft}`,
    date: `\\text{age} = \\frac{\\text{${REF_DATE}} - \\text{${h.built}}}{365.25} = ${ageYears(h.built).toFixed(2)}`,
    log: `\\text{log\\_sqft} = \\ln(${h.sqft}) = ${Math.log(h.sqft).toFixed(3)}`,
    onehot: `\\text{is\\_Downtown} = [\\text{hood} = \\text{Downtown}] = ${h.hood === "Downtown" ? 1 : 0}`,
    scale: st.scalers[0] ? `z = \\frac{x - \\mu}{\\sigma}, \\quad \\mu_{\\text{${st.scalers[0].key.replace(/_/g, "\\_")}}} = ${st.scalers[0].mu.toFixed(3)},\\ \\sigma = ${st.scalers[0].sd.toFixed(3)}` : undefined,
    interact: `x_{\\text{size}\\times\\text{DT}} = x_{\\text{size}} \\cdot \\text{is\\_Downtown}`,
  };
  return (
    <div className="space-y-2">
      {st.on && specific[st.stage] && <TexBlock>{specific[st.stage]!}</TexBlock>}
      <TexBlock>{ridge}</TexBlock>
      <TexBlock>{r2}</TexBlock>
    </div>
  );
}
