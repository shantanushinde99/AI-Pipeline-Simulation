import { useMemo, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { RotateCcw } from "lucide-react";
import { usePlayback } from "../../hooks/usePlayback";
import { PlaybackControls } from "../../components/visualizer/PlaybackControls";
import { EventLog } from "../../components/visualizer/EventLog";
import { StepExplainer, Workspace } from "../../components/visualizer/Workspace";
import { AnimatedNumber } from "../../components/visualizer/AnimatedNumber";
import { Badge, Button, KV, Panel, Select, SimulatedLabel, Slider, Toggle, cx } from "../../components/ui";
import { TexBlock } from "../../components/Tex";
import { EASE, StepCaption } from "../grounded-generation/StepCaption";
import { CONFIGS, DEFAULT_EVAL, METRICS, simulate, type ConfigEval, type ConfigId, type EvalParams, type EvalState, type Metric, type QEval } from "./simulation";

// ---------- timing (seconds at 1×) ----------
const ROW_IN = 0.12, MARK_AT = 0.9, CHIP0 = 1.4, CHIP_GAP = 0.3;
const chipAt = (j: number) => CHIP0 + j * CHIP_GAP;
const TABLE_AT = chipAt(METRICS.length) + 0.2;
const DUR: Record<EvalState["phase"], number> = { intro: 3200, question: (TABLE_AT + 1.3) * 1000, aggregate: 3600, flips: 3600 };

const fmtV = (v: number) => (Number.isInteger(v) ? String(v) : v.toFixed(2));
const toneOf = (v: number) => (v >= 1 ? "var(--ok)" : v <= 0 ? "var(--err)" : "var(--warn)");

export default function RagEvaluationVisualizer() {
  const [params, setParams] = useState<EvalParams>(DEFAULT_EVAL);
  const [replay, setReplay] = useState(0);
  const sim = useMemo(() => simulate(params), [params]);
  const { A, B, steps } = sim;
  const pb = usePlayback(steps.length, (i) => DUR[steps[i]?.state.phase ?? "intro"]);
  const step = steps[pb.index];
  const k = 1 / pb.speed;
  const animKey = `${step.id}-${replay}-${JSON.stringify(params)}`;
  const set = (p: Partial<EvalParams>) => setParams((v) => ({ ...v, ...p }));
  const story = narrate(step.state, sim);
  const configOptions = (Object.keys(CONFIGS) as ConfigId[]).map((id) => ({ value: id, label: CONFIGS[id].label }));

  return (
    <Workspace
      main={
        <>
          <Panel title={B ? "Golden-set evaluation · A/B" : "Golden-set evaluation"} actions={<SimulatedLabel>Simulated RAG pipeline</SimulatedLabel>}>
            <StepCaption index={pb.index} total={steps.length} stepKey={animKey} headline={story.headline} sub={story.sub} onReplay={() => setReplay((r) => r + 1)} />
            <div className="overflow-x-auto">
              <div className="min-w-[680px]">
                <Canvas sim={sim} state={step.state} k={k} animKey={animKey} />
              </div>
            </div>
            <Scoreboard A={A} B={B} state={step.state} k={k} animKey={animKey} />
          </Panel>
          <Panel title="What's happening in this step">
            <AnimatePresence mode="wait" initial={false}>
              <motion.div key={animKey} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: 0.2 }}>
                <StepCard sim={sim} state={step.state} k={k} />
              </motion.div>
            </AnimatePresence>
          </Panel>
          <PlaybackControls pb={pb} stepTitles={steps.map((s) => s.title)} />
          <Panel title="Configurations" actions={<Button size="sm" variant="ghost" onClick={() => { setParams(DEFAULT_EVAL); pb.restart(); }}><RotateCcw size={12} /> Reset</Button>}>
            <div className="grid gap-4 sm:grid-cols-2">
              <Toggle label="A/B mode: compare two configurations" checked={params.mode === "ab"} onChange={(v) => set({ mode: v ? "ab" : "single" })} hint="Runs the same golden set through both and reports per-metric deltas and flipped questions." />
              <Slider label="top-k (both configs)" value={params.topK} min={1} max={6} onChange={(v) => set({ topK: v })} hint="Context uses up to 3 of the retrieved chunks." />
              <Select label={params.mode === "ab" ? "Config A" : "Configuration"} value={params.a} onChange={(v: ConfigId) => set({ a: v })} options={configOptions} />
              {params.mode === "ab" && <Select label="Config B" value={params.b} onChange={(v: ConfigId) => set({ b: v })} options={configOptions} />}
            </div>
            <div className="mt-3 flex flex-wrap items-center gap-1.5 text-xs">
              <span className="text-muted">Presets:</span>
              {([["baseline", "chunk20", "Chunk 80 vs 20"], ["baseline", "norerank", "Rerank on vs off"], ["baseline", "budget60", "Budget 300 vs 60"], ["baseline", "stale", "Healthy vs stale index"], ["baseline", "prompt", "Healthy vs no grounding"]] as const).map(([a, b, l]) => (
                <Button key={l} size="sm" variant={params.mode === "ab" && params.a === a && params.b === b ? "primary" : "secondary"} onClick={() => { setParams({ ...params, mode: "ab", a, b }); pb.restart(); }}>{l}</Button>
              ))}
            </div>
          </Panel>
        </>
      }
      inspector={
        <>
          <StepExplainer index={pb.index} title={step.title} what={step.description} why={WHY[step.state.phase]} />
          <Panel title="Calculation"><Calculation sim={sim} state={step.state} /></Panel>
          <Panel title="Key values">
            <KV rows={[
              ["questions", A.rows.length],
              ["top-k", A.cfg.topK],
              ["A", `chunk ${A.cfg.chunkSize}/${A.cfg.overlap}, rerank ${A.cfg.rerank ? "on" : "off"}`],
              ...(B ? [["B", `chunk ${B.cfg.chunkSize}/${B.cfg.overlap}, rerank ${B.cfg.rerank ? "on" : "off"}`] as [string, string]] : []),
              ["MRR (A)", step.state.phase === "aggregate" || step.state.phase === "flips" ? A.means.rr.toFixed(2) : "—"],
              ...(B ? [["MRR (B)", step.state.phase === "aggregate" || step.state.phase === "flips" ? B.means.rr.toFixed(2) : "—"] as [string, string]] : []),
            ]} />
          </Panel>
          <Panel title="Events"><EventLog events={step.events} /></Panel>
        </>
      }
    />
  );
}

const WHY: Record<EvalState["phase"], string> = {
  intro: "A golden set turns 'it seems to work' into numbers you can compare across changes. Each metric isolates one stage.",
  question: "Scoring each stage separately tells you where a failure starts: not retrieved, retrieved but not in context, in context but not used, used but mis-cited.",
  aggregate: "Averages make configurations comparable, but they compress: one broken question out of six is only a 0.17 drop.",
  flips: "Per-question flips are what you debug. A regression test should fail on any ✓ → ✗, not just on a lower average.",
};

function narrate(s: EvalState, sim: ReturnType<typeof simulate>) {
  const { A, B, cmp } = sim;
  if (s.phase === "intro") return { headline: `${A.rows.length} golden questions, ${B ? "two configurations" : "one configuration"}`, sub: "Each question will be run end to end and scored on six metrics." };
  if (s.phase === "question") {
    const ra = A.rows[s.q!], rb = B?.rows[s.q!];
    const bad = (r: QEval) => METRICS.filter(({ id }) => r.m[id] < 1).map(({ short }) => short);
    if (rb) {
      const changed = METRICS.filter(({ id }) => ra.m[id] !== rb.m[id]);
      return { headline: `Q${s.q! + 1}: ${changed.length ? `${changed.length} metric(s) differ between A and B` : "A and B agree on every metric"}`, sub: ra.question };
    }
    const b = bad(ra);
    return { headline: `Q${s.q! + 1}: ${b.length ? `fails on ${b.join(", ")}` : "passes every metric"}`, sub: ra.question };
  }
  if (s.phase === "aggregate") {
    if (!B) return { headline: `Averages: correctness ${A.means.correct.toFixed(2)}, MRR ${A.means.rr.toFixed(2)}`, sub: "Each column of the table is averaged over the golden set." };
    const worst = METRICS.reduce((a, m) => (cmp!.deltas[m.id] < cmp!.deltas[a.id] ? m : a));
    return { headline: cmp!.deltas[worst.id] < 0 ? `B is worse on ${worst.label} (${cmp!.deltas[worst.id].toFixed(2)})` : "B matches or beats A on every metric", sub: "Bars grow to each configuration's mean; the label is B − A." };
  }
  const qs = new Set(cmp!.flips.map((x) => x.qid)).size;
  return { headline: qs ? `${qs} question(s) flipped between A and B` : "No question flipped", sub: qs ? "Each red cell is a ✓ that became ✗ (or a score that fell)." : "On this golden set the change is invisible; a bigger set might still reveal it." };
}

const W = 800;

function Canvas({ sim, state, k, animKey }: { sim: ReturnType<typeof simulate>; state: EvalState; k: number; animKey: string }) {
  const { A, B, cmp } = sim;
  if (state.phase === "intro") {
    const cw = 252, ch = 66;
    return (
      <svg viewBox={`0 0 ${W} 170`} className="h-auto w-full" role="img" aria-label="Golden set questions with their expected facts">
        {A.rows.map((r, i) => {
          const x = 8 + (i % 3) * (cw + 12), y = 8 + Math.floor(i / 3) * (ch + 12);
          return (
            <motion.g key={`${animKey}-${i}`} initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: (0.15 + i * 0.25) * k, duration: 0.4 * k, ease: EASE }}>
              <rect x={x} y={y} width={cw} height={ch} rx={8} fill="var(--surface-2)" stroke="var(--line)" />
              <text x={x + 10} y={y + 18} fontSize={10} fontFamily="var(--font-mono)" fill="var(--accent)">Q{i + 1} · {r.qid}</text>
              <text x={x + 10} y={y + 35} fontSize={10.5} fill="var(--fg)">{r.question.length > 44 ? `${r.question.slice(0, 43)}…` : r.question}</text>
              <rect x={x + 10} y={y + 43} width={Math.min(cw - 20, 22 + r.expect.length * 6.2)} height={16} rx={8} fill="var(--ok-soft)" stroke="var(--ok)" />
              <text x={x + 18} y={y + 55} fontSize={9.5} fill="var(--ok)">expect “{r.expect}”</text>
            </motion.g>
          );
        })}
      </svg>
    );
  }
  if (state.phase === "aggregate") {
    const rowH = 30, H = 24 + METRICS.length * rowH;
    const L = 150, R = B ? 690 : 740;
    return (
      <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label="Mean of each metric per configuration">
        <text x={L} y={14} fontSize={10} fill="var(--muted)">0</text>
        <text x={R} y={14} textAnchor="end" fontSize={10} fill="var(--muted)">1.0</text>
        {METRICS.map(({ id, label }, i) => {
          const y = 22 + i * rowH;
          const bars: [ConfigEval, string, number][] = B ? [[A, "var(--accent)", 0], [B, "var(--violet)", 12]] : [[A, "var(--accent)", 4]];
          return (
            <g key={id}>
              <text x={L - 8} y={y + 15} textAnchor="end" fontSize={11} fill="var(--fg)">{label}</text>
              <rect x={L} y={y} width={R - L} height={B ? 24 : 18} rx={3} fill="var(--surface-2)" />
              {bars.map(([c, color, dy], bi) => (
                <g key={bi}>
                  <motion.rect key={`${animKey}-${id}-${bi}`} x={L} y={y + dy} height={10} rx={2} fill={color} initial={{ width: 0 }} animate={{ width: (R - L) * c.means[id] }} transition={{ delay: (0.2 + i * 0.2 + bi * 0.1) * k, duration: 0.8 * k, ease: EASE }} />
                </g>
              ))}
              {B && (
                <motion.text key={`${animKey}-d-${id}`} x={R + 8} y={y + 16} fontSize={11} fontWeight={700} fontFamily="var(--font-mono)" fill={cmp!.deltas[id] < -1e-9 ? "var(--err)" : cmp!.deltas[id] > 1e-9 ? "var(--ok)" : "var(--muted)"}
                  initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: (1.1 + i * 0.2) * k }}>
                  {cmp!.deltas[id] >= 0 ? "+" : "−"}{Math.abs(cmp!.deltas[id]).toFixed(2)}
                </motion.text>
              )}
            </g>
          );
        })}
      </svg>
    );
  }
  if (state.phase === "flips") {
    const L = 230, cw = 88, rh = 26, H = 30 + A.rows.length * rh;
    return (
      <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label="Matrix of questions by metrics showing which changed between A and B">
        {METRICS.map(({ short }, j) => <text key={short} x={L + j * cw + cw / 2} y={16} textAnchor="middle" fontSize={10.5} fill="var(--muted)">{short}</text>)}
        {A.rows.map((ra, i) => {
          const rb = B!.rows[i];
          const y = 24 + i * rh;
          return (
            <g key={ra.qid}>
              <text x={L - 10} y={y + 16} textAnchor="end" fontSize={11} fill="var(--fg)">Q{i + 1} {ra.question.length > 30 ? `${ra.question.slice(0, 29)}…` : ra.question}</text>
              {METRICS.map(({ id }, j) => {
                const a = ra.m[id], b = rb.m[id];
                const changed = Math.abs(a - b) > 1e-9;
                return (
                  <motion.g key={`${animKey}-${i}-${j}`} initial={{ opacity: 0, scale: 0.6 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: (0.2 + i * 0.25 + j * 0.04) * k, duration: 0.3 * k }}
                    style={{ transformOrigin: `${L + j * cw + cw / 2}px ${y + 11}px` }}>
                    <rect x={L + j * cw + 3} y={y} width={cw - 6} height={rh - 4} rx={4} fill={changed ? (b < a ? "var(--err-soft)" : "var(--ok-soft)") : "var(--surface-2)"} stroke={changed ? (b < a ? "var(--err)" : "var(--ok)") : "none"} />
                    <text x={L + j * cw + cw / 2} y={y + 15} textAnchor="middle" fontSize={10.5} fontFamily="var(--font-mono)" fill={changed ? (b < a ? "var(--err)" : "var(--ok)") : "var(--muted)"}>
                      {changed ? `${fmtV(a)}→${fmtV(b)}` : fmtV(a)}
                    </text>
                  </motion.g>
                );
              })}
            </g>
          );
        })}
      </svg>
    );
  }
  // question step: the retrieved top-k list per config + metric chips
  const cols: [ConfigEval, string][] = B ? [[A, "A"], [B, "B"]] : [[A, ""]];
  const colW = B ? 388 : 600;
  const maxK = Math.max(...cols.map(([c]) => c.rows[state.q!].retrieved.length));
  const rowH = 28, listTop = 40, chipsTop = listTop + maxK * rowH + 34;
  const H = chipsTop + 40;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label="Top-k retrieved chunks for this question with the relevant chunk marked, and the six metric scores">
      {cols.map(([c, tag], ci) => {
        const r = c.rows[state.q!];
        const x0 = 8 + ci * (colW + 16);
        const maxS = Math.max(...r.retrieved.map((h) => h.score), 1e-9);
        const barX = x0 + 138, barW = colW - 238;
        return (
          <g key={tag || "single"}>
            <text x={x0} y={14} fontSize={11} fontWeight={600} fill={ci ? "var(--violet)" : "var(--accent)"}>{tag ? `${tag} · ` : ""}{c.label}</text>
            <text x={x0} y={30} fontSize={10} fill="var(--muted)">top-{r.k} after {c.cfg.rerank ? "rerank" : "vector search"} · expect “{r.expect}”</text>
            {r.retrieved.map((h, i) => {
              const y = listTop + i * rowH;
              return (
                <motion.g key={`${animKey}-${ci}-${h.id}`} initial={{ opacity: 0, x: -14 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: (0.1 + i * ROW_IN) * k, duration: 0.3 * k }}>
                  <rect x={x0} y={y} width={colW} height={rowH - 5} rx={5} fill="var(--surface-2)" stroke="var(--line)" />
                  <motion.rect x={x0} y={y} width={colW} height={rowH - 5} rx={5} fill="var(--ok-soft)" stroke="var(--ok)" initial={{ opacity: 0 }} animate={{ opacity: h.relevant ? 1 : 0 }} transition={{ delay: MARK_AT * k, duration: 0.3 * k }} />
                  <text x={x0 + 10} y={y + 16} fontSize={11} fontFamily="var(--font-mono)" fill="var(--muted)">#{i + 1}</text>
                  <text x={x0 + 36} y={y + 16} fontSize={11} fontFamily="var(--font-mono)" fill="var(--fg)">{h.id}</text>
                  <rect x={barX} y={y + 8} width={barW} height={7} rx={3.5} fill="var(--line)" />
                  <motion.rect x={barX} y={y + 8} height={7} rx={3.5} fill={ci ? "var(--violet)" : "var(--accent)"} initial={{ width: 0 }} animate={{ width: (barW * Math.max(0, h.score)) / maxS }} transition={{ delay: (0.2 + i * ROW_IN) * k, duration: 0.5 * k, ease: EASE }} />
                  <text x={barX + barW + 8} y={y + 16} fontSize={10} fontFamily="var(--font-mono)" fill="var(--muted)">{h.score.toFixed(2)}</text>
                  {h.relevant && <motion.text x={x0 + colW - 8} y={y + 16} textAnchor="end" fontSize={10} fontWeight={700} fill="var(--ok)" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: MARK_AT * k }}>✓ fact</motion.text>}
                </motion.g>
              );
            })}
            {r.firstRelRank ? (
              <motion.g key={`${animKey}-rr-${ci}`} initial={{ opacity: 0, y: listTop }} animate={{ opacity: 1, y: listTop + (r.firstRelRank - 1) * rowH }} transition={{ delay: MARK_AT * k, duration: 0.5 * k, ease: EASE }}>
                <path d={`M ${x0 - 7} 4 l 6 7 l -6 7 z`} fill="var(--ok)" />
              </motion.g>
            ) : (
              <motion.text key={`${animKey}-miss-${ci}`} x={x0} y={listTop + r.retrieved.length * rowH + 12} fontSize={10.5} fontWeight={600} fill="var(--err)" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: MARK_AT * k }}>
                ✗ fact not in top-{r.k}{r.indexRank ? ` (its chunk ranks #${r.indexRank} in the index)` : ": no chunk in the index contains it"}
              </motion.text>
            )}
            {METRICS.map(({ id, short }, j) => {
              const v = r.m[id];
              const cw = (colW - 5 * 6) / 6;
              const x = x0 + j * (cw + 6);
              return (
                <motion.g key={`${animKey}-chip-${ci}-${id}`} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: chipAt(j) * k, duration: 0.3 * k }}>
                  <rect x={x} y={chipsTop} width={cw} height={34} rx={6} fill="var(--surface)" stroke={toneOf(v)} strokeWidth={1.5} />
                  <text x={x + cw / 2} y={chipsTop + 13} textAnchor="middle" fontSize={9} fill="var(--muted)">{short}</text>
                  <text x={x + cw / 2} y={chipsTop + 28} textAnchor="middle" fontSize={12} fontWeight={700} fontFamily="var(--font-mono)" fill={toneOf(v)}>{v >= 1 ? "✓ 1" : v <= 0 ? "✗ 0" : fmtV(v)}</text>
                </motion.g>
              );
            })}
          </g>
        );
      })}
    </svg>
  );
}

function Scoreboard({ A, B, state, k, animKey }: { A: ConfigEval; B: ConfigEval | null; state: EvalState; k: number; animKey: string }) {
  const showMeans = state.phase === "aggregate" || state.phase === "flips";
  const cell = (a: number, b: number | undefined) => {
    if (b === undefined) return <span style={{ color: toneOf(a) }}>{a >= 1 ? "✓" : a <= 0 ? "✗" : ""} {fmtV(a)}</span>;
    const changed = Math.abs(a - b) > 1e-9;
    return <span className={cx(changed && "rounded px-1 font-semibold", changed && (b < a ? "bg-err-soft text-err" : "bg-ok-soft text-ok"))}>{changed ? `${fmtV(a)}→${fmtV(b)}` : fmtV(a)}</span>;
  };
  return (
    <div className="mt-3 overflow-x-auto">
      <table className="w-full min-w-[560px] text-xs">
        <thead className="text-left text-muted">
          <tr><th className="py-1">question</th>{METRICS.map((m) => <th key={m.id} className="text-center">{m.short}</th>)}</tr>
        </thead>
        <tbody className="font-mono">
          {A.rows.map((r, i) => {
            const filled = i < state.filled;
            const isNew = state.phase === "question" && state.q === i;
            return (
              <tr key={r.qid} className={cx("border-t border-line", isNew && "bg-accent-soft/60")}>
                <td className="py-1 pr-2 font-sans">Q{i + 1} <span className="text-muted">{r.qid}</span></td>
                {METRICS.map(({ id }, j) => (
                  <td key={id} className="text-center">
                    {filled ? (
                      <motion.span key={isNew ? `${animKey}-${j}` : "s"} initial={isNew ? { opacity: 0, scale: 0.5 } : false} animate={{ opacity: 1, scale: 1 }} transition={{ delay: (isNew ? TABLE_AT + j * 0.08 : 0) * k, duration: 0.25 * k }} className="inline-block">
                        {cell(r.m[id], B?.rows[i].m[id])}
                      </motion.span>
                    ) : <span className="text-muted">·</span>}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
        <tfoot>
          <tr className="border-t-2 border-line font-mono">
            <td className="py-1 font-sans font-semibold">mean{B ? " (A → B)" : ""}</td>
            {METRICS.map(({ id }, j) => (
              <td key={id} className="text-center">
                {showMeans ? (
                  <span key={state.phase === "aggregate" ? animKey : "s"}>
                    <AnimatedNumber value={A.means[id]} from={0} digits={2} delay={(0.2 + j * 0.2) * k} duration={0.8 * k} />
                    {B && <>→<AnimatedNumber value={B.means[id]} from={0} digits={2} delay={(0.3 + j * 0.2) * k} duration={0.8 * k} /></>}
                  </span>
                ) : <span className="text-muted">—</span>}
              </td>
            ))}
          </tr>
        </tfoot>
      </table>
    </div>
  );
}

const DEF: Record<Metric, string> = {
  recall: "share of chunks containing the expected fact that made it into the top-k",
  rr: "1 / rank of the first relevant chunk in the final order (0 if none in top-k)",
  contextHit: "the expected fact survived context assembly (budget, chunk limit)",
  faithful: "the answer sentence appears in the context (abstaining counts as faithful)",
  correct: "the answer contains the expected fact",
  citation: "the cited passage contains the answer sentence",
};

function StepCard({ sim, state, k }: { sim: ReturnType<typeof simulate>; state: EvalState; k: number }) {
  const { A, B, cmp } = sim;
  if (state.phase === "intro") {
    return (
      <table className="w-full text-xs">
        <tbody>{METRICS.map(({ id, label }, i) => (
          <motion.tr key={id} className="border-t border-line" initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: (0.2 + i * 0.2) * k }}>
            <td className="py-1 pr-3 font-semibold">{label}</td><td className="text-muted">{DEF[id]}</td>
          </motion.tr>
        ))}</tbody>
      </table>
    );
  }
  if (state.phase === "question") {
    const cols: [ConfigEval, string][] = B ? [[A, "A"], [B, "B"]] : [[A, "result"]];
    return (
      <div className="space-y-2 text-xs">
        <table className="w-full">
          <thead className="text-left text-muted"><tr><th className="py-1">metric</th><th>how it is computed here</th>{cols.map(([, t]) => <th key={t} className="text-center">{t}</th>)}</tr></thead>
          <tbody>{METRICS.map(({ id, label }, j) => (
            <motion.tr key={id} className="border-t border-line align-top" initial={{ opacity: 0.2 }} animate={{ opacity: 1 }} transition={{ delay: chipAt(j) * k }}>
              <td className="py-1 pr-2 font-semibold">{label}</td>
              <td className="pr-2 text-muted">{explain(id, A.rows[state.q!])}</td>
              {cols.map(([c, t]) => <td key={t} className="text-center font-mono" style={{ color: toneOf(c.rows[state.q!].m[id]) }}>{fmtV(c.rows[state.q!].m[id])}</td>)}
            </motion.tr>
          ))}</tbody>
        </table>
        <p><span className="text-muted">Answer{B ? " (A)" : ""}:</span> {A.rows[state.q!].answer}</p>
        {B && B.rows[state.q!].answer !== A.rows[state.q!].answer && <p><span className="text-muted">Answer (B):</span> {B.rows[state.q!].answer}</p>}
      </div>
    );
  }
  if (state.phase === "aggregate") {
    return (
      <div className="space-y-1 text-xs">
        <p>Every metric here is a per-question score averaged over the golden set; for RR that average is the MRR.</p>
        <div className="grid gap-1 sm:grid-cols-2">
          {METRICS.map(({ id, label }, i) => (
            <motion.div key={id} className="flex justify-between rounded-md border border-line px-2 py-1" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: (0.2 + i * 0.2) * k }}>
              <span>{label === "reciprocal rank" ? "MRR" : label}</span>
              <span className="font-mono">{A.rows.map((r) => fmtV(r.m[id])).join("+")} / {A.rows.length} = {A.means[id].toFixed(2)}{B && <> → {B.means[id].toFixed(2)}</>}</span>
            </motion.div>
          ))}
        </div>
      </div>
    );
  }
  const byQ = [...new Set(cmp!.flips.map((f) => f.qid))];
  if (!byQ.length) return <p className="text-xs text-muted">Nothing flipped. {A.id === B!.id ? "A and B are the same configuration." : "This change doesn't affect any question in this small golden set. That is not proof it is harmless: add questions that exercise it."}</p>;
  return (
    <ul className="space-y-1.5 text-xs">
      {byQ.map((qid, i) => {
        const fl = cmp!.flips.filter((f) => f.qid === qid);
        return (
          <motion.li key={qid} className="rounded-md border border-line p-2" initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: (0.3 + i * 0.3) * k }}>
            <div className="font-medium">{fl[0].question}</div>
            <div className="mt-1 flex flex-wrap gap-1">{fl.map((f) => <Badge key={f.metric} tone={f.to < f.from ? "err" : "ok"}>{METRICS.find((m) => m.id === f.metric)!.short}: {fmtV(f.from)} → {fmtV(f.to)}</Badge>)}</div>
          </motion.li>
        );
      })}
    </ul>
  );
}

function explain(id: Metric, r: QEval) {
  const hits = r.retrieved.filter((h) => h.relevant).length;
  switch (id) {
    case "recall": return r.relevant.length ? `${hits} of ${r.relevant.length} relevant chunk(s) in top-${r.k}` : "no chunk contains the fact";
    case "rr": return r.firstRelRank ? `first relevant at rank ${r.firstRelRank} → 1/${r.firstRelRank}` : `none in top-${r.k} → 0`;
    case "contextHit": return `“${r.expect}” in the assembled context?`;
    case "faithful": return r.source === "none" ? "abstained → nothing unfaithful" : r.source === "prior" ? "answered from model memory, not context" : "answer sentence found in context";
    case "correct": return `answer contains “${r.expect}”?`;
    case "citation": return "cited passage contains the answer sentence?";
  }
}

function Calculation({ sim, state }: { sim: ReturnType<typeof simulate>; state: EvalState }) {
  const { A, B } = sim;
  if (state.phase === "question") {
    const r = A.rows[state.q!];
    const hits = r.retrieved.filter((h) => h.relevant).length;
    return (
      <div className="space-y-2">
        <TexBlock>{`\\text{recall@}${r.k} = \\frac{|\\text{Rel} \\cap \\text{Top}_{${r.k}}|}{|\\text{Rel}|} = ${r.relevant.length ? `\\frac{${hits}}{${r.relevant.length}} = ${fmtV(r.m.recall)}` : "0"}`}</TexBlock>
        <TexBlock>{`\\text{RR} = \\frac{1}{\\text{rank}_{\\text{first rel}}} = ${r.firstRelRank ? `\\frac{1}{${r.firstRelRank}} = ${fmtV(r.m.rr)}` : "0"}`}</TexBlock>
        {B && <p className="text-[11px] text-muted">Shown for A. B: recall {fmtV(B.rows[state.q!].m.recall)}, RR {fmtV(B.rows[state.q!].m.rr)}.</p>}
      </div>
    );
  }
  if (state.phase === "aggregate" || state.phase === "flips") {
    return (
      <div className="space-y-2">
        <TexBlock>{`\\text{MRR} = \\frac{1}{|Q|}\\sum_{q \\in Q} \\text{RR}_q = \\frac{${A.rows.map((r) => fmtV(r.m.rr)).join("+")}}{${A.rows.length}} = ${A.means.rr.toFixed(2)}`}</TexBlock>
        {B && <TexBlock>{`\\Delta_{\\text{correct}} = ${B.means.correct.toFixed(2)} - ${A.means.correct.toFixed(2)} = ${(B.means.correct - A.means.correct).toFixed(2)}`}</TexBlock>}
      </div>
    );
  }
  return <TexBlock>{"\\text{recall@}k = \\frac{|\\text{Rel} \\cap \\text{Top}_k|}{|\\text{Rel}|} \\qquad \\text{MRR} = \\frac{1}{|Q|}\\sum_q \\frac{1}{\\text{rank}_q}"}</TexBlock>;
}
