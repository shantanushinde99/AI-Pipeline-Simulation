import { useMemo, useState, type ReactNode } from "react";
import { AnimatePresence, motion } from "motion/react";
import { RotateCcw } from "lucide-react";
import { usePlayback } from "../../hooks/usePlayback";
import { PlaybackControls } from "../../components/visualizer/PlaybackControls";
import { EventLog } from "../../components/visualizer/EventLog";
import { StepExplainer, Workspace } from "../../components/visualizer/Workspace";
import { AnimatedNumber } from "../../components/visualizer/AnimatedNumber";
import { LineChart } from "../../components/visualizer/LineChart";
import { Badge, Button, Callout, KV, Panel, SimulatedLabel, Slider, Toggle, cx } from "../../components/ui";
import { TexBlock } from "../../components/Tex";
import type { VisualizerTab } from "../types";
import { StepCaption, ease, pct } from "../hallucination-analysis/kit";
import { BIAS, CATEGORIES, DEFAULT_CONFIG, simulate, type Config, type SafetyState, type Scored } from "./simulation";

/** One schedule (seconds at 1×) for the dots, the score counters and the step card. */
const T = { appear: (i: number) => 0.1 + i * 0.04, score: (j: number) => 0.3 + j * 0.25, move: 0.9, stagger: 0.02, line: 0.9, rows: (i: number) => 0.4 + i * 0.25 };
const DUR: Record<SafetyState["phase"], number> = { set: 2400, score: 3000, threshold: 3000, confusion: 3000, category: 3000, tradeoff: 3000, lookalikes: 3200 };

const SX = (v: number) => 160 + v * 530;
const catIdx = (p: Scored) => CATEGORIES.findIndex((c) => c.id === p.category);
const rowIn = (p: Scored, all: Scored[]) => all.filter((q) => q.category === p.category).indexOf(p);

function narrate(s: SafetyState, run: ReturnType<typeof simulate>, cfg: Config) {
  const o = run.overall;
  switch (s.phase) {
    case "set": return { headline: `${run.all.length} test prompts, from clearly fine to clearly risky`, sub: "Red marks should be blocked; green ticks should be answered. Lookalikes sound risky but are harmless." };
    case "score": { const c = CATEGORIES.find((x) => x.id === s.category)!; return { headline: `The classifier scores the ${c.name.toLowerCase()}${c.id === "lookalike" ? "" : " prompts"}`, sub: c.id === "lookalike" ? (cfg.contextAware ? "Context words like 'Python' or 'chess' pull the score back down." : "Context rules are off, so 'kill', 'shoot' and 'attack' look dangerous.") : "Risk keywords push the score up; darker dot = higher risk score." }; }
    case "threshold": return { headline: `Every prompt slides onto the risk axis; ≥ ${run.threshold.toFixed(2)} is blocked`, sub: `${o.tp + o.fp} blocked, ${o.tn + o.fn} allowed. Rings mark the mistakes.` };
    case "confusion": return { headline: `${o.fp} benign prompt${o.fp === 1 ? "" : "s"} over-refused, ${o.fn} risky prompt${o.fn === 1 ? "" : "s"} let through`, sub: "Each prompt drops into its cell: rows are the truth, columns are the decision." };
    case "category": return { headline: "One threshold, very different errors per category", sub: "Averages hide which users get refused and which risks slip through." };
    case "tradeoff": return { headline: "Move the threshold and the two error rates trade places", sub: "Lower: more over-refusal. Higher: more under-refusal. Pick the point your policy can defend." };
    default: return { headline: "Without lookalikes, over-refusal is invisible", sub: "An eval set made only of obvious cases can't tell a careful classifier from a keyword filter." };
  }
}

export default function SafetyEvalVisualizer({ tab }: { tab: VisualizerTab }) {
  const [cfg, setCfg] = useState<Config>(DEFAULT_CONFIG);
  const [replay, setReplay] = useState(0);
  const run = useMemo(() => simulate(cfg), [cfg]);
  const pb = usePlayback(run.steps.length, (i) => DUR[run.steps[i].state.phase]);
  const step = run.steps[pb.index];
  const s = step.state;
  const k = 1 / pb.speed;
  const animKey = `${step.id}-${replay}`;
  const story = narrate(s, run, cfg);
  const set = (p: Partial<Config>) => setCfg((c) => ({ ...c, ...p }));
  const o = run.overall;
  if (tab === "metrics") return null;

  return (
    <Workspace
      main={
        <>
          <Panel title="Red-team set → risk score → decision" actions={<SimulatedLabel>Rule-based classifier</SimulatedLabel>}>
            <StepCaption index={pb.index} total={run.steps.length} stepKey={step.id} headline={story.headline} sub={story.sub} onReplay={() => setReplay((r) => r + 1)} />
            <div className="overflow-x-auto"><div className="min-w-[640px]"><Canvas run={run} s={s} cfg={cfg} k={k} animKey={animKey} /></div></div>
            <div className="mt-2 flex flex-wrap gap-3 text-[11px] text-muted">
              <span><span className="font-semibold text-err">■ !</span> should be blocked</span>
              <span><span className="font-semibold text-ok">● ✓</span> should be answered</span>
              <span>darker red dot = higher risk score</span>
              <span><span className="text-warn">dashed ring</span> = over-refusal · <span className="text-err">thick ring</span> = under-refusal</span>
            </div>
          </Panel>
          <Panel title="What's happening in this step">
            <AnimatePresence mode="wait" initial={false}>
              <motion.div key={animKey} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: 0.2 }}>
                <StepCard run={run} s={s} cfg={cfg} k={k} />
              </motion.div>
            </AnimatePresence>
          </Panel>
          <PlaybackControls pb={pb} stepTitles={run.steps.map((x) => x.title)} />
          <Panel title="Classifier & eval-set settings" actions={<Button size="sm" variant="ghost" onClick={() => { setCfg(DEFAULT_CONFIG); pb.restart(); }}><RotateCcw size={12} /> Reset</Button>}>
            <div className="grid gap-3 sm:grid-cols-3">
              <Slider label="Block threshold" value={cfg.threshold} min={0.05} max={0.95} step={0.05} format={(v) => v.toFixed(2)} onChange={(v) => set({ threshold: v })} hint="Block when risk score ≥ threshold." />
              <Toggle label="Context-aware rules" checked={cfg.contextAware} onChange={(v) => set({ contextAware: v })} hint="Break it: off = pure keyword matching ('kill' is always risky)." />
              <Toggle label="Include benign lookalikes in the eval set" checked={cfg.includeLookalikes} onChange={(v) => set({ includeLookalikes: v })} hint="Off = the eval can no longer see keyword over-refusal." />
            </div>
          </Panel>
        </>
      }
      inspector={
        <>
          <StepExplainer index={pb.index} title={step.title} what={step.description}
            why="Over-refusal fails real users with legitimate needs (including people seeking support); under-refusal lets harm through. A safety eval has to measure both." />
          <Panel title="Calculation">
            <div className="space-y-2">
              <TexBlock>{`s = \\sigma\\Big(${BIAS} + \\textstyle\\sum w_{\\text{risk}} + \\sum w_{\\text{context}}\\Big)`}</TexBlock>
              <TexBlock>{`\\begin{aligned} \\text{over-refusal} &= \\tfrac{FP}{FP + TN} = \\tfrac{${o.fp}}{${o.benign}} = ${o.overRefusal.toFixed(3)} \\\\ \\text{under-refusal} &= \\tfrac{FN}{FN + TP} = \\tfrac{${o.fn}}{${o.risky}} = ${o.underRefusal.toFixed(3)} \\end{aligned}`}</TexBlock>
            </div>
          </Panel>
          <Panel title="Key values">
            <KV rows={[
              ["prompts in eval set", run.set.length],
              ["threshold", run.threshold.toFixed(2)],
              ["blocked / allowed", `${o.tp + o.fp} / ${o.tn + o.fn}`],
              ["TP · FN · FP · TN", `${o.tp} · ${o.fn} · ${o.fp} · ${o.tn}`],
              ["context rules", cfg.contextAware ? "on" : "off"],
            ]} />
          </Panel>
          <Panel title="Events"><EventLog events={step.events} /></Panel>
        </>
      }
    />
  );
}

function Canvas({ run, s, cfg, k, animKey }: { run: ReturnType<typeof simulate>; s: SafetyState; cfg: Config; k: number; animKey: string }) {
  const layout = s.phase === "set" || s.phase === "score" ? "grid" : s.phase === "confusion" ? "confusion" : "strip";
  const t = run.threshold;
  const boxes = { tp: [160, 34], fn: [430, 34], fp: [160, 160], tn: [430, 160] } as const;
  const counters: Record<string, number> = { tp: 0, fn: 0, fp: 0, tn: 0 };
  const cell = (p: Scored) => (p.risky ? (p.score >= t ? "tp" : "fn") : p.score >= t ? "fp" : "tn");
  const scoredCats = new Set(CATEGORIES.slice(0, s.scored).map((c) => c.id));
  const decided = layout !== "grid";
  return (
    <svg viewBox="0 0 720 300" className="h-auto w-full" role="img" aria-label="Prompts as dots: grouped by category, then placed on a risk-score axis with a threshold">
      {layout === "grid" && CATEGORIES.map((c, i) => (
        <text key={c.id} x={150 + i * 160} y={24} textAnchor="middle" fontSize={11} fontWeight={600} fill={s.category === c.id ? "var(--accent)" : "var(--fg)"}>{c.name}</text>
      ))}
      {layout === "strip" && (
        <g>
          {CATEGORIES.map((c, i) => <text key={c.id} x={8} y={66 + i * 58} fontSize={11} fontWeight={600} fill={!cfg.includeLookalikes && c.id === "lookalike" ? "var(--muted)" : "var(--fg)"}>{c.short}{!cfg.includeLookalikes && c.id === "lookalike" ? " (excluded)" : ""}</text>)}
          <motion.rect key={`shade-${animKey}`} x={SX(t)} y={34} width={SX(1) - SX(t) + 8} height={232} fill="var(--err)" initial={{ opacity: 0 }} animate={{ opacity: 0.07 }} transition={{ delay: (s.phase === "threshold" ? T.move + 0.3 : 0) * k }} />
          <line x1={SX(0)} x2={SX(1)} y1={272} y2={272} stroke="var(--line)" />
          {[0, 0.25, 0.5, 0.75, 1].map((v) => <text key={v} x={SX(v)} y={288} textAnchor="middle" fontSize={10} fontFamily="var(--font-mono)" fill="var(--muted)">{v}</text>)}
          <text x={SX(0.5)} y={299} textAnchor="middle" fontSize={10} fill="var(--muted)">risk score →</text>
          <motion.line key={`th-${animKey}`} x1={SX(t)} x2={SX(t)} y1={30} y2={272} stroke="var(--fg)" strokeWidth={2} strokeDasharray="5 4"
            initial={{ pathLength: s.phase === "threshold" ? 0 : 1 }} animate={{ pathLength: 1 }} transition={{ delay: T.move * k, duration: 0.5 * k }} />
          <text x={SX(t) + 6} y={30} fontSize={10} fontWeight={600} fill="var(--err)">block ≥ {t.toFixed(2)}</text>
          <text x={SX(t) - 6} y={30} textAnchor="end" fontSize={10} fill="var(--muted)">allow</text>
        </g>
      )}
      {layout === "confusion" && (Object.keys(boxes) as (keyof typeof boxes)[]).map((key) => {
        const [bx, by] = boxes[key];
        const label = { tp: "risky · blocked (TP)", fn: "risky · allowed (FN, under-refusal)", fp: "benign · blocked (FP, over-refusal)", tn: "benign · answered (TN)" }[key];
        const bad = key === "fn" || key === "fp";
        return (
          <g key={key}>
            <rect x={bx} y={by} width={250} height={112} rx={8} fill={bad ? "var(--err-soft)" : "var(--surface-2)"} stroke="var(--line)" />
            <text x={bx + 10} y={by + 18} fontSize={11} fontWeight={600} fill={bad ? "var(--err)" : "var(--fg)"}>{label}</text>
            <text x={bx + 240} y={by + 18} textAnchor="end" fontSize={13} fontWeight={700} fontFamily="var(--font-mono)" fill="var(--fg)">
              <AnimatedNumber key={animKey} svg value={run.overall[key]} from={0} digits={0} delay={T.move * k} duration={0.6 * k} />
            </text>
          </g>
        );
      })}
      {layout === "confusion" && (
        <>
          <text x={150} y={92} textAnchor="end" fontSize={10} fill="var(--muted)">truth: risky</text>
          <text x={150} y={218} textAnchor="end" fontSize={10} fill="var(--muted)">truth: benign</text>
        </>
      )}
      {run.all.map((p, i) => {
        const c = catIdx(p), r = rowIn(p, run.all);
        const excluded = !cfg.includeLookalikes && p.category === "lookalike";
        let x: number, y: number;
        if (layout === "grid") { x = 150 + c * 160 + (r % 2 ? 26 : -26); y = 58 + Math.floor(r / 2) * 70 + (r % 2 ? 18 : 0); }
        else if (layout === "strip") { x = SX(p.score); y = 62 + c * 58 + ((r % 3) - 1) * 13; }
        else if (excluded) { x = 700; y = 290; }
        else { const key = cell(p); const j = counters[key]++; x = boxes[key][0] + 22 + (j % 8) * 29; y = boxes[key][1] + 44 + Math.floor(j / 8) * 32; }
        const scored = scoredCats.has(p.category) || decided;
        const nowScoring = s.phase === "score" && s.category === p.category;
        const j = r;
        const wrong = decided && !excluded && (p.risky ? p.score < t : p.score >= t);
        return (
          <motion.g key={p.id} initial={false} animate={{ x, y, opacity: excluded ? (layout === "confusion" ? 0 : 0.25) : 1 }}
            transition={{ duration: T.move * k, delay: (s.phase === "set" ? 0 : i * T.stagger) * k, ease }}>
            <motion.g key={s.phase === "set" ? animKey : "dot"} initial={s.phase === "set" ? { scale: 0 } : false} animate={{ scale: 1 }} transition={{ delay: T.appear(i) * k, type: "spring", stiffness: 300, damping: 18 }}>
              <circle r={13} fill="var(--surface)" stroke={wrong ? (p.risky ? "var(--err)" : "var(--warn)") : "var(--line)"} strokeWidth={wrong ? (p.risky ? 3.5 : 2.5) : 1} strokeDasharray={wrong && !p.risky ? "4 2" : undefined} />
              <motion.circle key={nowScoring ? animKey : "fill"} r={13} fill="var(--err)" initial={nowScoring ? { fillOpacity: 0 } : false} animate={{ fillOpacity: scored ? 0.08 + 0.75 * p.score : 0 }} transition={{ delay: nowScoring ? T.score(j) * k : 0, duration: 0.4 * k }} />
              <text y={3.5} textAnchor="middle" fontSize={9.5} fontWeight={600} fontFamily="var(--font-mono)" fill="var(--fg)">{p.id}</text>
              {p.risky ? <rect x={7} y={-16} width={10} height={10} rx={2} fill="var(--err)" /> : <circle cx={12} cy={-11} r={5} fill="var(--ok)" />}
              <text x={12} y={-8} textAnchor="middle" fontSize={8} fontWeight={700} fill="var(--surface)">{p.risky ? "!" : "✓"}</text>
              {layout === "grid" && scored && (
                <text y={26} textAnchor="middle" fontSize={9.5} fontFamily="var(--font-mono)" fill="var(--muted)">
                  {nowScoring ? <AnimatedNumber key={animKey} svg value={p.score} from={0} delay={T.score(j) * k} duration={0.4 * k} /> : p.score.toFixed(2)}
                </text>
              )}
            </motion.g>
          </motion.g>
        );
      })}
    </svg>
  );
}

function Marked({ p }: { p: Scored }) {
  const words = [...p.risk.map((w) => [w, "risk"] as const), ...p.context.map((w) => [w, "ctx"] as const)];
  if (!words.length) return <>{p.text}</>;
  const re = new RegExp(`(${words.map(([w]) => w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})`, "gi");
  const kind = (w: string) => words.find(([x]) => x === w.toLowerCase())?.[1];
  return <>{p.text.split(re).map((part, i): ReactNode => {
    const kd = kind(part);
    return kd ? <mark key={i} className={cx("rounded px-0.5", kd === "risk" ? "bg-err-soft text-err" : "bg-ok-soft text-ok")}>{part}</mark> : <span key={i}>{part}</span>;
  })}</>;
}

function StepCard({ run, s, cfg, k }: { run: ReturnType<typeof simulate>; s: SafetyState; cfg: Config; k: number }) {
  const o = run.overall;
  if (s.phase === "set") {
    return (
      <div className="space-y-2">
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          {CATEGORIES.map((c, i) => {
            const items = run.all.filter((p) => p.category === c.id);
            return (
              <motion.div key={c.id} className="rounded-md border border-line bg-surface-2 p-2 text-xs" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: T.rows(i) * k }}>
                <div className="font-semibold">{c.name}</div>
                <div className="mt-1 flex gap-1"><Badge tone="err">! {items.filter((p) => p.risky).length} block</Badge><Badge tone="ok">✓ {items.filter((p) => !p.risky).length} answer</Badge></div>
              </motion.div>
            );
          })}
        </div>
        <Callout title="Risky prompts are described abstractly">They are written as [bracketed descriptions], which is enough for a keyword classifier and keeps this page free of harmful content. For self-harm, &quot;blocked&quot; means switching to a supportive safe completion, never a cold refusal; support-seeking prompts should be answered.</Callout>
      </div>
    );
  }
  if (s.phase === "score") {
    const items = run.all.filter((p) => p.category === s.category);
    return (
      <div className="space-y-1.5">
        {items.map((p, j) => (
          <motion.div key={p.id} className="grid grid-cols-[30px_minmax(0,1fr)_90px] items-center gap-2 text-xs" initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: (T.score(j) - 0.15) * k }}>
            <span className={cx("font-mono font-semibold", p.risky ? "text-err" : "text-ok")}>{p.id}</span>
            <span className="min-w-0"><Marked p={p} /> <span className="font-mono text-[10px] text-muted">z = {p.z.toFixed(1)}</span></span>
            <div className="relative h-4 rounded bg-surface-2">
              <motion.div className="absolute inset-y-0 left-0 rounded bg-err/70" initial={{ width: 0 }} animate={{ width: `${p.score * 100}%` }} transition={{ delay: T.score(j) * k, duration: 0.4 * k, ease }} />
              <span className="absolute inset-0 flex items-center justify-end pr-1 font-mono text-[10px]">{p.score.toFixed(2)}</span>
            </div>
          </motion.div>
        ))}
        <p className="pt-1 text-[11px] text-muted"><mark className="rounded bg-err-soft px-0.5 text-err">red</mark> risk keyword (+), <mark className="rounded bg-ok-soft px-0.5 text-ok">green</mark> context keyword (−){cfg.contextAware ? "" : ", context rules are off"}. z = {BIAS} + weights; score = σ(z).</p>
      </div>
    );
  }
  if (s.phase === "threshold") {
    const tiles: [string, number, string][] = [["blocked", o.tp + o.fp, "text-err"], ["allowed", o.tn + o.fn, "text-ok"], ["mistakes", o.fp + o.fn, "text-warn"]];
    return (
      <div className="grid grid-cols-3 gap-2">
        {tiles.map(([l, v, cls], i) => (
          <motion.div key={l} className="rounded-md border border-line bg-surface-2 p-2 text-center" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: (T.move + i * 0.2) * k }}>
            <div className={cx("font-mono text-2xl", cls)}><AnimatedNumber value={v} from={0} digits={0} delay={(T.move + i * 0.2) * k} duration={0.6 * k} /></div>
            <div className="text-[11px] text-muted">{l}</div>
          </motion.div>
        ))}
      </div>
    );
  }
  if (s.phase === "confusion") {
    const rows: [string, number, string, string][] = [["over-refusal", o.overRefusal, `${o.fp} of ${o.benign} benign prompts blocked`, "bg-warn"], ["under-refusal", o.underRefusal, `${o.fn} of ${o.risky} risky prompts allowed`, "bg-err"]];
    return (
      <div className="space-y-2">
        {rows.map(([l, v, sub, cls], i) => (
          <div key={l} className="grid grid-cols-[100px_minmax(0,1fr)_48px] items-center gap-2 text-xs">
            <span className="font-medium">{l}</span>
            <div className="h-4 rounded bg-surface-2"><motion.div className={cx("h-4 rounded", cls)} initial={{ width: 0 }} animate={{ width: `${v * 100}%` }} transition={{ delay: (T.move + i * 0.3) * k, duration: 0.6 * k, ease }} /></div>
            <span className="text-right font-mono">{pct(v)}</span>
            <span />
            <span className="col-span-2 -mt-1 text-[11px] text-muted">{sub}</span>
          </div>
        ))}
      </div>
    );
  }
  if (s.phase === "category") {
    return (
      <div className="space-y-2">
        {run.perCategory.map((c, i) => (
          <motion.div key={c.id} className={cx("grid grid-cols-[120px_minmax(0,1fr)] items-center gap-2 text-xs", !c.included && "opacity-50")} initial={{ opacity: 0 }} animate={{ opacity: c.included ? 1 : 0.5 }} transition={{ delay: T.rows(i) * k }}>
            <span className="font-medium">{c.name}</span>
            <div className="space-y-0.5">
              {([["over", c.r.overRefusal, c.r.benign, "bg-warn"], ["under", c.r.underRefusal, c.r.risky, "bg-err"]] as const).map(([l, v, n, cls]) => (
                <div key={l} className="grid grid-cols-[40px_minmax(0,1fr)_60px] items-center gap-1.5">
                  <span className="text-[10px] text-muted">{l}</span>
                  <div className="h-2.5 rounded bg-surface-2">{n > 0 && <motion.div className={cx("h-2.5 rounded", cls)} initial={{ width: 0 }} animate={{ width: `${v * 100}%` }} transition={{ delay: (T.rows(i) + 0.1) * k, duration: 0.5 * k, ease }} />}</div>
                  <span className="text-right font-mono text-[10px]">{n ? pct(v) : "n/a"}</span>
                </div>
              ))}
            </div>
          </motion.div>
        ))}
        <p className="text-[11px] text-muted">n/a: the category has no prompts of that kind (lookalikes are all benign). {cfg.includeLookalikes ? "" : "Lookalikes are excluded from the overall numbers."}</p>
      </div>
    );
  }
  if (s.phase === "tradeoff") {
    return (
      <div>
        <LineChart xLabel="threshold" yLabel="rate" markerX={run.threshold} height={200} series={[
          { name: "over-refusal (benign blocked)", color: "var(--warn)", points: run.sweep.map((p) => [p.t, p.overRefusal]) },
          { name: "under-refusal (risky allowed)", color: "var(--err)", points: run.sweep.map((p) => [p.t, p.underRefusal]) },
        ]} />
        <motion.p className="text-xs" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.6 * k }}>
          At {run.threshold.toFixed(2)}: over-refusal {pct(o.overRefusal)}, under-refusal {pct(o.underRefusal)}. Drag the threshold slider and watch the dashed marker and both rates move.
        </motion.p>
      </div>
    );
  }
  const g = run.grid;
  return (
    <div className="space-y-2">
      <div className="overflow-x-auto">
        <table className="w-full min-w-[380px] text-xs">
          <thead className="text-left text-muted"><tr><th className="py-1">measured over-refusal</th><th>lookalikes in eval</th><th>lookalikes left out</th></tr></thead>
          <tbody>
            {g.map((row, i) => (
              <tr key={i} className="border-t border-line">
                <td className="py-1.5">context rules {row[0].contextAware ? "on" : "off (keywords only)"}</td>
                {row.map((c, j) => (
                  <td key={j} className={cx("font-mono", !c.contextAware && !c.includeLookalikes && "text-err")}>
                    <motion.span initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: T.rows(i * 2 + j) * k }}><AnimatedNumber value={c.overRefusal * 100} from={0} digits={0} delay={T.rows(i * 2 + j) * k} duration={0.5 * k} />%</motion.span>
                    {!c.contextAware && !c.includeLookalikes && <span className="ml-1 font-sans">← looks perfect</span>}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: T.rows(4) * k }}>
        <Callout tone="warn" title="The eval set decides what you can see">A keyword filter that refuses &quot;kill a Python process&quot; scores 0% over-refusal on a set without lookalikes. Only prompts that sound risky but are fine reveal it.</Callout>
      </motion.div>
    </div>
  );
}
