import { useMemo, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { RotateCcw } from "lucide-react";
import { usePlayback } from "../../hooks/usePlayback";
import { PlaybackControls } from "../../components/visualizer/PlaybackControls";
import { EventLog } from "../../components/visualizer/EventLog";
import { StepExplainer, Workspace } from "../../components/visualizer/Workspace";
import { AnimatedNumber } from "../../components/visualizer/AnimatedNumber";
import { Badge, Button, Callout, KV, Panel, SimulatedLabel, Toggle, cx } from "../../components/ui";
import { TexBlock } from "../../components/Tex";
import type { VisualizerTab } from "../types";
import { StepCaption, ease } from "../hallucination-analysis/kit";
import {
  BIAS, BUGS, BUG_OF, DEFAULT_INPUT, DEFS, ENTITIES, FEATURES, STALE_HOURS, differs, featureVectors, fmtF, predict, simulate, skewReport,
  type BugId, type FeatureId, type SkewInput, type SkewState,
} from "./simulation";

/** One schedule (seconds at 1×): canvas dots, cell counters and the step card all read from it. */
const T = { row: (i: number) => 0.2 + i * 0.22, travel: 0.7, count: 0.5 };
const DUR: Record<SkewState["phase"], number> = { raw: 2800, feature: 3400, score: 3400, report: 3000 };
const SHORT: Record<FeatureId, string> = { orders_7d: "orders_7d", avg_order_value: "avg_value $", last_order_hour: "last_hour", avg_rating: "avg_rating", tenure_days: "tenure" };
const digits = (f: FeatureId) => (f === "avg_order_value" || f === "avg_rating" ? 2 : 0);

const RY = (i: number) => 100 + i * 38;
const LX = (h: number) => 196 - (h / 216) * 136; // raw timeline: 216 h ago … now
const CX = (j: number) => 216 + j * 104; // left edge of feature column j
const PX = 744;

export default function SkewVisualizer({ tab }: { tab: VisualizerTab }) {
  const [input, setInput] = useState<SkewInput>(DEFAULT_INPUT);
  const [replay, setReplay] = useState(0);
  const steps = useMemo(() => simulate(input), [input]);
  const vecs = useMemo(() => featureVectors(input), [input]);
  const rep = useMemo(() => skewReport(input), [input]);
  const pb = usePlayback(steps.length, (i) => DUR[steps[i].state.phase]);
  const step = steps[pb.index];
  const s = step.state;
  const k = 1 / pb.speed;
  const animKey = `${step.id}-${replay}`;
  const worst = vecs.reduce((a, b) => (Math.abs(b.pOnline - b.pBatch) > Math.abs(a.pOnline - a.pBatch) ? b : a));
  const zt = predict(worst.batch).z, zs = predict(worst.online).z;
  const toggleBug = (b: BugId, on: boolean) => setInput((x) => ({ ...x, bugs: on ? [...x.bugs, b] : x.bugs.filter((y) => y !== b) }));
  const mismatches = rep.perFeature.reduce((a, x) => a + x.mismatches, 0);
  if (tab === "metrics") return null;

  return (
    <Workspace
      main={
        <>
          <Panel title="Raw events → batch SQL (training) vs online service (serving)" actions={<SimulatedLabel>Simulated pipelines & model</SimulatedLabel>}>
            <StepCaption index={pb.index} total={steps.length} stepKey={step.id} headline={s.headline} sub={s.sub} onReplay={() => setReplay((r) => r + 1)} />
            <div className="overflow-x-auto"><div className="min-w-[760px]"><Canvas s={s} input={input} k={k} animKey={animKey} /></div></div>
            <div className="mt-2 flex flex-wrap gap-3 text-[11px] text-muted">
              <span>each feature column: <b className="text-fg">train</b> = batch SQL value, <b className="text-fg">serve</b> = online service value</span>
              <span><span className="font-semibold text-err">≠ red cell</span> = skewed value</span>
              <span><span className="text-ok">green band</span> = true 7-day window{input.bugs.includes("stale_join") && !input.shared ? " · red dashed = stale window" : ""}</span>
            </div>
          </Panel>
          <Panel title="What's happening in this step">
            <AnimatePresence mode="wait" initial={false}>
              <motion.div key={animKey} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: 0.2 }}>
                <StepCard s={s} input={input} k={k} rep={rep} vecs={vecs} onFix={() => setInput((x) => ({ ...x, shared: !x.shared }))} />
              </motion.div>
            </AnimatePresence>
          </Panel>
          <PlaybackControls pb={pb} stepTitles={steps.map((x) => x.title)} />
          <Panel title="Bugs in the serving path & the fix" actions={<Button size="sm" variant="ghost" onClick={() => { setInput(DEFAULT_INPUT); pb.restart(); }}><RotateCcw size={12} /> Reset</Button>}>
            <div className="grid gap-3 sm:grid-cols-2">
              {BUGS.map((b) => <Toggle key={b.id} label={b.name} checked={input.bugs.includes(b.id)} onChange={(v) => toggleBug(b.id, v)} hint={`Break it: skews ${Object.entries(BUG_OF).find(([, x]) => x === b.id)?.[0]}.`} />)}
              <Toggle label="Fix: shared feature definitions (feature store)" checked={input.shared} onChange={(v) => setInput((x) => ({ ...x, shared: v }))} hint="Both paths call one definition, so every diff disappears." />
            </div>
          </Panel>
        </>
      }
      inspector={
        <>
          <StepExplainer index={pb.index} title={step.title} what={step.description}
            why="The model was validated on training-path features. If serving computes them differently, offline metrics say nothing about production, and nothing errors: predictions are just quietly wrong." />
          <Panel title="Calculation">
            <div className="space-y-2">
              <TexBlock>{`p = \\sigma\\Big(${BIAS} + \\textstyle\\sum_f w_f\\,(x_f - c_f)\\Big)`}</TexBlock>
              <TexBlock>{`\\begin{aligned} \\text{${worst.id.replace("_", "\\_")} train: } & \\sigma(${zt.toFixed(2)}) = ${worst.pBatch.toFixed(3)} \\\\ \\text{serve: } & \\sigma(${zs.toFixed(2)}) = ${worst.pOnline.toFixed(3)} \\\\ |\\Delta p| &= ${Math.abs(worst.pOnline - worst.pBatch).toFixed(3)} \\end{aligned}`}</TexBlock>
              <p className="text-[11px] text-muted">Customer with the largest prediction gap under the current settings.</p>
            </div>
          </Panel>
          <Panel title="Key values">
            <KV rows={[
              ["skewed values", `${mismatches} / ${ENTITIES.length * FEATURES.length}`],
              ["max |Δp|", rep.maxDp.toFixed(3)],
              ["decisions flipped (p ≥ 0.5)", rep.flipped],
              ["active bugs", input.shared ? "none (shared defs)" : input.bugs.length],
            ]} />
          </Panel>
          <Panel title="Events"><EventLog events={step.events} /></Panel>
        </>
      }
    />
  );
}

function Canvas({ s, input, k, animKey }: { s: SkewState; input: SkewInput; k: number; animKey: string }) {
  const stale = input.bugs.includes("stale_join") && !input.shared;
  const scored = s.phase === "score" || s.phase === "report";
  return (
    <svg viewBox="0 0 860 340" className="h-auto w-full" role="img" aria-label="Per-customer feature values from the training path and the serving path, side by side">
      <text x={128} y={40} textAnchor="middle" fontSize={11} fontWeight={600} fill="var(--fg)">raw orders</text>
      <text x={128} y={58} textAnchor="middle" fontSize={9.5} fill="var(--muted)">← 9 days ago · now →</text>
      {FEATURES.map((f, j) => {
        const cur = s.feature === f;
        return (
          <g key={f}>
            {cur && <motion.rect key={`hl-${animKey}`} x={CX(j) - 2} y={26} width={104} height={300} rx={8} fill="var(--accent)" initial={{ opacity: 0 }} animate={{ opacity: 0.08 }} />}
            <text x={CX(j) + 50} y={40} textAnchor="middle" fontSize={11} fontWeight={600} fontFamily="var(--font-mono)" fill={cur ? "var(--accent)" : "var(--fg)"}>{SHORT[f]}</text>
            <text x={CX(j) + 25} y={58} textAnchor="middle" fontSize={9.5} fill="var(--muted)">train</text>
            <text x={CX(j) + 75} y={58} textAnchor="middle" fontSize={9.5} fill="var(--muted)">serve</text>
          </g>
        );
      })}
      <text x={PX + 50} y={40} textAnchor="middle" fontSize={11} fontWeight={600} fill={scored ? "var(--accent)" : "var(--muted)"}>p(buy again)</text>
      <text x={PX + 25} y={58} textAnchor="middle" fontSize={9.5} fill="var(--muted)">train</text>
      <text x={PX + 75} y={58} textAnchor="middle" fontSize={9.5} fill="var(--muted)">serve</text>
      {/* Raw-event timelines with the 7-day window (and the stale window when that bug is on). */}
      <rect x={LX(168)} y={74} width={LX(0) - LX(168)} height={RY(5) - 50} fill="var(--ok)" opacity={0.08} />
      {stale && <rect x={LX(STALE_HOURS + 168)} y={76} width={LX(STALE_HOURS) - LX(STALE_HOURS + 168)} height={RY(5) - 54} fill="none" stroke="var(--err)" strokeDasharray="4 3" />}
      {s.rows.map((r, i) => {
        const e = ENTITIES[i], y = RY(i);
        const p = scored && r.pBatch !== undefined && r.pOnline !== undefined;
        const flip = p && (r.pBatch! >= 0.5) !== (r.pOnline! >= 0.5);
        return (
          <g key={r.id}>
            <text x={8} y={y + 4} fontSize={10.5} fontWeight={600} fontFamily="var(--font-mono)" fill="var(--fg)">{r.id}</text>
            <line x1={LX(216)} x2={LX(0)} y1={y} y2={y} stroke="var(--line)" />
            {e.orders.map((o, n) => (
              <motion.rect key={`${o.hoursAgo}-${s.phase === "raw" ? animKey : "o"}`} x={LX(o.hoursAgo) - 3} y={y - 7} width={6} height={14} rx={1.5} fill="var(--accent)"
                initial={s.phase === "raw" ? { opacity: 0, scaleY: 0 } : false} animate={{ opacity: 1, scaleY: 1 }} transition={{ delay: (T.row(i) + n * 0.08) * k, duration: 0.3 * k }} />
            ))}
            {FEATURES.map((f, j) => {
              const b = r.batch[f], o = r.online[f];
              if (b === undefined || o === undefined) return <rect key={f} x={CX(j)} y={y - 12} width={100} height={24} rx={4} fill="var(--surface-2)" opacity={0.5} />;
              const cur = s.feature === f, bad = differs(b, o);
              const d = (T.row(i) + T.travel) * k;
              return (
                <g key={f}>
                  {cur && [25, 75].map((dx) => (
                    <motion.circle key={`${dx}-${animKey}`} r={4} fill={dx === 75 && bad ? "var(--err)" : "var(--accent)"} initial={{ cx: LX(0), cy: y, opacity: 0 }}
                      animate={{ cx: CX(j) + dx, cy: y, opacity: [0, 1, 1, 0] }} transition={{ delay: T.row(i) * k, duration: T.travel * k, ease }} />
                  ))}
                  {(["batch", "online"] as const).map((side, n) => {
                    const v = side === "batch" ? b : o, red = n === 1 && bad;
                    return (
                      <motion.g key={side + (cur ? animKey : "")} initial={cur ? { opacity: 0 } : false} animate={{ opacity: 1 }} transition={{ delay: cur ? d : 0, duration: 0.2 * k }}>
                        <rect x={CX(j) + n * 50} y={y - 12} width={48} height={24} rx={4} fill={red ? "var(--err-soft)" : "var(--surface-2)"} stroke={red ? "var(--err)" : "var(--line)"} />
                        <text x={CX(j) + n * 50 + 24} y={y + 4} textAnchor="middle" fontSize={Math.abs(v) >= 1000 ? 8.5 : 10} fontFamily="var(--font-mono)" fill={red ? "var(--err)" : "var(--fg)"}>
                          {cur ? <AnimatedNumber key={animKey} svg value={v} from={0} digits={digits(f)} delay={d} duration={T.count * k} /> : fmtF(f, v)}
                        </text>
                      </motion.g>
                    );
                  })}
                  {bad && <text x={CX(j) + 49} y={y - 13} textAnchor="middle" fontSize={11} fontWeight={700} fill="var(--err)">≠</text>}
                </g>
              );
            })}
            {p ? (["pBatch", "pOnline"] as const).map((key, n) => {
              const v = r[key]!, x = PX + n * 50;
              const big = n === 1 && Math.abs(r.pOnline! - r.pBatch!) > 0.1;
              return (
                <g key={key}>
                  <rect x={x} y={y - 12} width={48} height={24} rx={4} fill="var(--surface-2)" stroke={big ? "var(--err)" : "var(--line)"} />
                  <motion.rect key={s.phase === "score" ? animKey : "bar"} x={x + 1} y={y + 6} height={5} rx={1} fill={big ? "var(--err)" : "var(--accent)"} initial={s.phase === "score" ? { width: 0 } : false} animate={{ width: 46 * v }} transition={{ delay: T.row(i) * k, duration: T.travel * k, ease }} />
                  <text x={x + 24} y={y + 2} textAnchor="middle" fontSize={10} fontFamily="var(--font-mono)" fill={big ? "var(--err)" : "var(--fg)"}>
                    {s.phase === "score" ? <AnimatedNumber key={animKey} svg value={v} from={0} digits={2} delay={T.row(i) * k} duration={T.travel * k} /> : v.toFixed(2)}
                  </text>
                </g>
              );
            }) : <rect x={PX} y={y - 12} width={98} height={24} rx={4} fill="var(--surface-2)" opacity={0.5} />}
            {flip && <text x={PX + 100} y={y - 14} textAnchor="end" fontSize={9} fontWeight={700} fill="var(--err)">flips</text>}
          </g>
        );
      })}
    </svg>
  );
}

function StepCard({ s, input, k, rep, vecs, onFix }: { s: SkewState; input: SkewInput; k: number; rep: ReturnType<typeof skewReport>; vecs: ReturnType<typeof featureVectors>; onFix: () => void }) {
  if (s.phase === "raw") {
    const paths: [string, string, string][] = [["Training path", "Nightly batch SQL over the warehouse; built the training set the model was validated on.", "ok"], ["Serving path", input.shared ? "Calls the same shared feature definitions as training." : "A separate Python service re-implements each feature for low-latency requests.", input.shared ? "ok" : "warn"]];
    return (
      <div className="grid gap-2 sm:grid-cols-2">
        {paths.map(([t, d, tone], i) => (
          <motion.div key={t} className="rounded-md border border-line bg-surface-2 p-2 text-xs" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: (0.3 + i * 0.4) * k }}>
            <div className="mb-1 flex items-center gap-2 font-semibold">{t} <Badge tone={tone === "ok" ? "ok" : "warn"}>{tone === "ok" ? "reference" : "re-implemented"}</Badge></div>
            <p className="text-muted">{d}</p>
          </motion.div>
        ))}
      </div>
    );
  }
  if (s.phase === "feature" && s.feature) {
    const f = s.feature, def = DEFS[f], bug = BUG_OF[f];
    const active = !!bug && input.bugs.includes(bug) && !input.shared;
    return (
      <div className="space-y-2 text-xs">
        <div className="grid gap-2 sm:grid-cols-2">
          <div><div className="mb-1 font-semibold">Training: batch SQL</div><pre className="overflow-x-auto rounded-md border border-line bg-surface-2 p-2 font-mono text-[11px]">{def.batch}</pre></div>
          <div><div className="mb-1 font-semibold">Serving: online service {active ? <Badge tone="err">bug</Badge> : <Badge tone="ok">matches</Badge>}</div>
            <pre className={cx("overflow-x-auto rounded-md border p-2 font-mono text-[11px]", active ? "border-err bg-err-soft" : "border-line bg-surface-2")}>{input.shared ? "features.compute(customer, as_of)  # shared" : active ? def.onlineBug : def.online}</pre></div>
        </div>
        <div className="grid grid-cols-3 gap-1 sm:grid-cols-6">
          {vecs.map((v, i) => {
            const bad = differs(v.batch[f], v.online[f]);
            return (
              <motion.div key={v.id} className={cx("rounded border px-1.5 py-1 font-mono text-[10px]", bad ? "border-err text-err" : "border-line text-muted")} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: (T.row(i) + T.travel) * k }}>
                {v.id} {bad ? `≠ Δ ${fmtF(f, v.online[f] - v.batch[f])}` : "✓ same"}
              </motion.div>
            );
          })}
        </div>
      </div>
    );
  }
  if (s.phase === "score") {
    return (
      <div className="space-y-1.5">
        {vecs.map((v, i) => {
          const d = v.pOnline - v.pBatch;
          return (
            <div key={v.id} className="grid grid-cols-[44px_minmax(0,1fr)_70px] items-center gap-2 text-xs">
              <span className="font-mono font-semibold">{v.id}</span>
              <div className="relative h-4 rounded bg-surface-2">
                <motion.div className="absolute inset-y-0 rounded bg-err/60" style={{ left: `${Math.min(v.pBatch, v.pOnline) * 100}%` }} initial={{ width: 0 }} animate={{ width: `${Math.abs(d) * 100}%` }} transition={{ delay: (T.row(i) + T.travel) * k, duration: 0.5 * k, ease }} />
                <div className="absolute inset-y-0 w-0.5 bg-fg" style={{ left: `${v.pBatch * 100}%` }} title="train" />
              </div>
              <span className={cx("text-right font-mono", Math.abs(d) > 0.1 && "text-err")}>Δ {d >= 0 ? "+" : ""}<AnimatedNumber value={d} from={0} digits={3} delay={(T.row(i) + T.travel) * k} duration={0.5 * k} /></span>
            </div>
          );
        })}
        <p className="text-[11px] text-muted">Black tick = p with training features; red band = how far the serving-path prediction moved.</p>
      </div>
    );
  }
  return (
    <div className="space-y-2">
      {rep.perFeature.map((x, i) => (
        <div key={x.f} className="grid grid-cols-[110px_minmax(0,1fr)_40px] items-center gap-2 text-xs">
          <span className="font-mono">{x.f}</span>
          <div className="h-3 rounded bg-surface-2"><motion.div className="h-3 rounded bg-err" initial={{ width: 0 }} animate={{ width: `${(x.mismatches / ENTITIES.length) * 100}%` }} transition={{ delay: (0.2 + i * 0.2) * k, duration: 0.5 * k, ease }} /></div>
          <span className="text-right font-mono">{x.mismatches}/{ENTITIES.length}</span>
        </div>
      ))}
      <Callout tone={input.shared ? "ok" : "warn"} title={input.shared ? "Fixed: one definition, two runtimes" : "Fix: define each feature once"}>
        {input.shared ? "Training and serving call the same feature code (a feature store / shared library), so every diff is 0 and the predictions match." : "Move the feature logic into one shared definition used by both the batch job and the online service, then log serving features and compare them with a batch recompute."}
        <div className="mt-2"><Button size="sm" onClick={onFix}>{input.shared ? "Undo the fix" : "Apply the fix"}</Button></div>
      </Callout>
    </div>
  );
}
